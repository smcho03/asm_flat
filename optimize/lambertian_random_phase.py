"""
lambertian_random_phase.py
--------------------------
Simulate a *rough* (Lambertian-like) membrane by applying a random phase
on top of h(x,y). This mimics speckle illumination from a diffuser, a
common trick in holographic imaging to linearize / decorrelate the signal.
Cf. Popoff et al. (wavefront shaping through random media), and
Wang et al. (computational ghost imaging w/ random phase).

We compare three forward models:
  (a) specular   : U0 = exp(i 4pi h/lam)                 (default)
  (b) diffuser-S : U0 = exp(i 4pi h/lam) * exp(i psi(x,y))  (static speckle)
  (c) diffuser-avg : (b) averaged over N realizations      (incoherent avg)

And show reconstruction quality for each.  The static-random-phase model
makes the intensity approximately *quadratic* in h (like Lambertian
reflectance), decoupling it from interferometric fringes.

Output:
  output/lambertian_random_phase/
    lambertian_summary.png
    lambertian_metrics.txt
"""

import sys, time, json, functools
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print = functools.partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).parent.parent))
from sensor_model  import HolographicSensor, gaussian_bump
from sensor_params import wavelength, mem_pitch
from sensor_utils  import STYLE

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "output" / "lambertian_random_phase"
OUT.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

MEM   = 128
CMOS  = 256
GRID  = 384
DIST  = 5e-3
AMP   = 200e-9
SIGMA = 150e-6
H_SCALE = 500e-9

# -----------------------------------------------------------------------
# A thin wrapper around the sensor that injects a random phase screen
# -----------------------------------------------------------------------
base = HolographicSensor(
    wavelength=wavelength, mem_res=MEM, mem_pitch=mem_pitch,
    cmos_res=CMOS, grid_res=GRID, distance=DIST, device=device,
).to(device)

def forward_spec(h):
    return base(h)

def _random_phase(seed, N=MEM):
    g = torch.Generator(device=device).manual_seed(seed)
    return 2*np.pi * torch.rand(N, N, generator=g, dtype=torch.float32,
                                 device=device)

def _forward_complex_A(h, A_cplx):
    pad = (base._ms, base.grid_res - base._me,
           base._ms, base.grid_res - base._me)
    h_grid = F.pad(h, pad)
    A_grid = F.pad(A_cplx, pad)
    phi = -(4.0*np.pi/base.lam) * h_grid
    U0  = A_grid * torch.exp(1j * phi)
    Ud  = torch.fft.ifft2(torch.fft.fft2(U0) * base.H)
    I   = torch.abs(Ud).pow(2)
    return I[base._s:base._e, base._s:base._e]

def forward_diffuser(h, seed=0):
    psi = _random_phase(seed)
    A_cplx = torch.exp(1j * psi)
    return _forward_complex_A(h, A_cplx)

def forward_diffuser_avg(h, n_real=10, base_seed=0):
    acc = None
    for k in range(n_real):
        I = forward_diffuser(h, seed=base_seed + k)
        acc = I if acc is None else acc + I
    return acc / n_real

# -----------------------------------------------------------------------
# Targets
# -----------------------------------------------------------------------
h_true = gaussian_bump(N=MEM, dx=mem_pitch, amplitude=AMP, sigma=SIGMA,
                       device=device)

with torch.no_grad():
    I_spec    = forward_spec(h_true)
    I_diff    = forward_diffuser(h_true, seed=0)
    I_diffavg = forward_diffuser_avg(h_true, n_real=20)

# -----------------------------------------------------------------------
# Reconstruction (uses matching forward model)
# -----------------------------------------------------------------------
def reconstruct(forward_fn, I_tgt, lr=1e-3, n_iter=1500):
    h_var = torch.zeros(MEM, MEM, dtype=torch.float32, device=device,
                        requires_grad=True)
    opt = torch.optim.Adam([h_var], lr=lr)
    losses = []
    for _ in range(n_iter):
        opt.zero_grad()
        I = forward_fn(h_var * H_SCALE)
        loss = torch.mean((I - I_tgt)**2)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return (h_var.detach() * H_SCALE), losses

def psnr_db(a, b):
    mse_v = float(torch.mean((a - b)**2))
    peak  = float(b.abs().max())
    if mse_v < 1e-30: return 100.0
    if peak < 1e-30:  return 0.0
    return 10.0 * np.log10(peak**2 / mse_v)

print("\nReconstructing (specular) ...")
t0 = time.time()
h_spec, L_spec = reconstruct(forward_spec, I_spec)
p_spec = psnr_db(h_spec, h_true); dt_s = time.time()-t0

print("Reconstructing (diffuser static) ...")
t0 = time.time()
h_diff, L_diff = reconstruct(lambda h: forward_diffuser(h, seed=0), I_diff)
p_diff = psnr_db(h_diff, h_true); dt_d = time.time()-t0

print("Reconstructing (diffuser avg, N=20) ...")
t0 = time.time()
h_davg, L_davg = reconstruct(lambda h: forward_diffuser_avg(h, n_real=20),
                              I_diffavg, n_iter=800)
p_davg = psnr_db(h_davg, h_true); dt_a = time.time()-t0

print(f"\n  specular       PSNR={p_spec:.2f} dB  ({dt_s:.1f}s)")
print(f"  diffuser-S     PSNR={p_diff:.2f} dB  ({dt_d:.1f}s)")
print(f"  diffuser-avg   PSNR={p_davg:.2f} dB  ({dt_a:.1f}s)")

# -----------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------
plt.rcParams.update(STYLE)
fig, axes = plt.subplots(3, 4, figsize=(22, 15))
fig.suptitle("Lambertian / random-phase diffuser simulation",
             fontsize=13, color="#e6edf3")
fig.patch.set_facecolor("#0d1117")

def _style(ax):
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e", labelsize=7)
    ax.grid(color="#30363d", lw=0.4, ls=":")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

def _im(ax, data, cmap, vmin, vmax, title, label=""):
    im = ax.imshow(data.T, cmap=cmap, origin="lower", aspect="equal",
                   vmin=vmin, vmax=vmax)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(label, color="#8b949e", fontsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="#8b949e")
    ax.set_title(title, fontsize=9, color="#e6edf3")
    _style(ax)

# Row 0: intensity images
I_spec_np    = I_spec.cpu().numpy()
I_diff_np    = I_diff.cpu().numpy()
I_diffavg_np = I_diffavg.cpu().numpy()
vmax = max(I_spec_np.max(), I_diff_np.max(), I_diffavg_np.max())

_im(axes[0,0], I_spec_np, "inferno", 0, vmax, "(a) Specular I", "I")
_im(axes[0,1], I_diff_np, "inferno", 0, vmax, "(b) Diffuser static I", "I")
_im(axes[0,2], I_diffavg_np, "inferno", 0, vmax,
    "(c) Diffuser avg N=20", "I")
# Mean of I (more Lambertian-like under averaging)
_im(axes[0,3], np.abs(I_diffavg_np - I_diffavg_np.mean()), "inferno", 0,
    np.abs(I_diffavg_np-I_diffavg_np.mean()).max(),
    "avg deviation from mean", "|dI|")

# Row 1: h_true vs each reconstruction
h_true_np = h_true.cpu().numpy()*1e9
vm = max(abs(h_true_np).max(), abs(h_spec.cpu().numpy()*1e9).max(),
         abs(h_diff.cpu().numpy()*1e9).max(),
         abs(h_davg.cpu().numpy()*1e9).max(), 1.0)

_im(axes[1,0], h_true_np, "RdBu", -vm, vm, "h_true", "h [nm]")
_im(axes[1,1], h_spec.cpu().numpy()*1e9, "RdBu", -vm, vm,
    f"h_rec specular  PSNR={p_spec:.1f}", "h [nm]")
_im(axes[1,2], h_diff.cpu().numpy()*1e9, "RdBu", -vm, vm,
    f"h_rec diff-S  PSNR={p_diff:.1f}", "h [nm]")
_im(axes[1,3], h_davg.cpu().numpy()*1e9, "RdBu", -vm, vm,
    f"h_rec diff-avg  PSNR={p_davg:.1f}", "h [nm]")

# Row 2: centre cross-sections + loss curves
m_um = (np.arange(MEM) - (MEM-1)*0.5) * mem_pitch * 1e6
mid = MEM // 2
ax = axes[2,0]
ax.plot(m_um, h_true_np[mid,:], color="#8b949e", lw=2, ls=":", label="GT")
ax.plot(m_um, h_spec.cpu().numpy()[mid,:]*1e9, color="#58a6ff", lw=1.0, label="specular")
ax.plot(m_um, h_diff.cpu().numpy()[mid,:]*1e9, color="#f78166", lw=1.0, label="diff-S")
ax.plot(m_um, h_davg.cpu().numpy()[mid,:]*1e9, color="#3fb950", lw=1.0, label="diff-avg")
ax.set_xlabel("x [um]", fontsize=9, color="#8b949e")
ax.set_ylabel("h [nm]", fontsize=9, color="#8b949e")
ax.set_title("Centre cross-sections", fontsize=10, color="#e6edf3")
ax.legend(fontsize=7, labelcolor="#e6edf3", framealpha=0.3)
_style(ax)

# I(mid) profiles
ax = axes[2,1]
ax.plot(I_spec_np[I_spec_np.shape[0]//2, :], color="#58a6ff", lw=1.0, label="specular")
ax.plot(I_diff_np[I_diff_np.shape[0]//2, :], color="#f78166", lw=1.0, alpha=0.8, label="diff-S")
ax.plot(I_diffavg_np[I_diffavg_np.shape[0]//2, :], color="#3fb950", lw=1.2, label="diff-avg")
ax.set_xlabel("cmos pixel", fontsize=9, color="#8b949e")
ax.set_ylabel("I", fontsize=9, color="#8b949e")
ax.set_title("CMOS centre row intensity", fontsize=10, color="#e6edf3")
ax.legend(fontsize=7, labelcolor="#e6edf3", framealpha=0.3)
_style(ax)

# Loss curves
ax = axes[2,2]
ax.plot(L_spec, color="#58a6ff", lw=1.0, label=f"specular ({p_spec:.1f}dB)")
ax.plot(L_diff, color="#f78166", lw=1.0, label=f"diff-S ({p_diff:.1f}dB)")
ax.plot(L_davg, color="#3fb950", lw=1.0, label=f"diff-avg ({p_davg:.1f}dB)")
ax.set_yscale("log")
ax.set_xlabel("iteration", fontsize=9, color="#8b949e")
ax.set_ylabel("loss", fontsize=9, color="#8b949e")
ax.set_title("Loss curves", fontsize=10, color="#e6edf3")
ax.legend(fontsize=7, labelcolor="#e6edf3", framealpha=0.3)
_style(ax)

# Summary text
ax = axes[2,3]
ax.axis("off")
summary = (
    "Lambertian / random-phase diffuser\n"
    "===================================\n"
    f"GT: Gaussian, A={AMP*1e9:.0f} nm, sigma={SIGMA*1e6:.0f} um\n"
    f"Distance = {DIST*1e3} mm, mem = {MEM}, cmos = {CMOS}\n\n"
    f"(a) specular  PSNR = {p_spec:.2f} dB\n"
    f"(b) diff-S    PSNR = {p_diff:.2f} dB\n"
    f"(c) diff-avg  PSNR = {p_davg:.2f} dB  (N=20)\n\n"
    "Notes\n"
    " * Static diffuser creates speckle but\n"
    "   is still deterministic (inversion works).\n"
    " * Averaging realizations -> Lambertian-\n"
    "   like incoherent model; inversion becomes\n"
    "   ill-posed (only intensity envelope).\n"
    " * Random phase is the trick used in\n"
    "   ghost-imaging + wavefront-shaping papers\n"
    "   to decorrelate the forward operator.\n"
)
ax.text(0.02, 0.98, summary, transform=ax.transAxes, va="top", ha="left",
        fontsize=8, color="#e6edf3", family="monospace")

plt.tight_layout()
plt.savefig(OUT / "lambertian_summary.png", dpi=150, bbox_inches="tight",
            facecolor="#0d1117")
plt.close()

(OUT / "lambertian_metrics.txt").write_text(summary, encoding="utf-8")
(OUT / "lambertian_metrics.json").write_text(json.dumps(dict(
    psnr_specular=p_spec, psnr_diff_static=p_diff, psnr_diff_avg=p_davg,
    amp_nm=AMP*1e9, sigma_um=SIGMA*1e6, distance_mm=DIST*1e3,
    n_realizations_avg=20,
), indent=2))
print(f"Saved -> {OUT/'lambertian_summary.png'}")
