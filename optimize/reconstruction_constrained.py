"""
reconstruction_constrained.py
-----------------------------
Test whether enforcing h >= 0 (membrane pressed inward only)
removes the in-line holography sign / twin-image ambiguity.

Background: the unconstrained run (reconstruction_perfectness.py)
hit loss ~1e-12 but PSNR ~15 dB across all amplitudes -> the Adam
optimum is a *valid* solution of |U|^2 = I_target but is the
twin-image / sign-flipped of the true h.  Convention says membrane
deformation is unidirectional, so h >= 0 is physically justified.

Three reparameterizations are compared:
  (a) baseline (no constraint, signed h)
  (b) ReLU      h = relu(h_var) * H_SCALE
  (c) Softplus  h = softplus(h_var) * H_SCALE
  (d) Squared   h = h_var^2 * H_SCALE

For each amplitude in pre-wrap and phase-wrap regimes, run Adam
with the chosen reparam and report PSNR + RMSE.

Output:
  output/reconstruction_constrained/
    constrained_summary.png
    constrained_metrics.txt
    constrained_metrics.json
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
OUT  = ROOT / "output" / "reconstruction_constrained"
OUT.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

# ---------------- sensor / config ----------------
MEM, CMOS, GRID, DIST = 128, 256, 384, 5e-3
SIGMA = 200e-6
LR    = 3e-3
N_ITER = 3000
H_SCALE_FACTOR = 2.0

lam = wavelength
dx  = mem_pitch
PHASE_WRAP_NM = lam / 4.0 * 1e9   # 158.20 nm

sensor = HolographicSensor(
    wavelength=lam, mem_res=MEM, mem_pitch=dx,
    cmos_res=CMOS, grid_res=GRID, distance=DIST, device=device,
).to(device)

A_nm_all = np.array([5, 20, 50, 100, 150, 200, 300, 500])

# ---------------- reparam variants ----------------
def parameterize(name, raw):
    if name == "baseline":
        return raw
    if name == "relu":
        return F.relu(raw)
    if name == "softplus":
        return F.softplus(raw)
    if name == "squared":
        return raw * raw
    raise ValueError(name)

def init_raw(name, mean_norm=0.05):
    """A small positive initial state ensures gradient flows for all
    reparameterizations from the first step."""
    if name == "baseline":
        return torch.zeros(MEM, MEM, device=device, requires_grad=True)
    if name == "relu":
        # start at +0.05 so relu output is nonzero -> active gradient
        x = torch.full((MEM, MEM), mean_norm, device=device)
        return x.clone().requires_grad_(True)
    if name == "softplus":
        # softplus(0)=ln2 ~ 0.693 — start at -2 so softplus is small positive
        x = torch.full((MEM, MEM), -2.0, device=device)
        return x.clone().requires_grad_(True)
    if name == "squared":
        x = torch.full((MEM, MEM), 0.2, device=device)  # h0 = 0.04 * H_SCALE
        return x.clone().requires_grad_(True)
    raise ValueError(name)

# ---------------- run ----------------
def reconstruct(name, h_true, h_scale, n_iter=N_ITER, lr=LR):
    with torch.no_grad():
        I_tgt = sensor(h_true)
    raw = init_raw(name)
    opt = torch.optim.Adam([raw], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iter)
    losses = []
    for _ in range(n_iter):
        opt.zero_grad()
        h = parameterize(name, raw) * h_scale
        loss = torch.mean((sensor(h) - I_tgt)**2)
        loss.backward(); opt.step(); sched.step()
        losses.append(float(loss.detach()))
    with torch.no_grad():
        h_final = parameterize(name, raw) * h_scale
    return h_final, losses

def psnr_db(a, b):
    mse = float(torch.mean((a - b)**2))
    peak = float(b.abs().max())
    if mse < 1e-30: return 200.0
    if peak < 1e-30: return 0.0
    return 10.0 * np.log10(peak**2 / mse)

variants = ["baseline", "relu", "softplus", "squared"]

results = []
recon_keep = {}     # for plotting, keep one pre-wrap and one wrap example
print(f"\nphase-wrap pixel limit lam/4 = {PHASE_WRAP_NM:.2f} nm")
print(f"\n{'A[nm]':>6}  {'variant':>10}  {'PSNR':>8}  {'RMSE[nm]':>10}  "
      f"{'loss':>10}  {'time[s]':>7}")
for A_nm in A_nm_all:
    A_m = float(A_nm) * 1e-9
    h_scale = max(H_SCALE_FACTOR * A_m, 50e-9)
    h_true = gaussian_bump(N=MEM, dx=dx, amplitude=A_m, sigma=SIGMA, device=device)
    regime = "pre-wrap" if A_nm < PHASE_WRAP_NM else "wrap"
    row = dict(A_nm=float(A_nm), regime=regime)
    keep = dict(h_true=h_true.cpu().numpy()*1e9)
    for v in variants:
        t0 = time.time()
        h_rec, _ = reconstruct(v, h_true, h_scale)
        p = psnr_db(h_rec, h_true)
        rmse = float(torch.sqrt(torch.mean((h_rec - h_true)**2))) * 1e9
        loss_final = float(torch.mean((sensor(h_rec) - sensor(h_true))**2))
        dt = time.time() - t0
        print(f"{A_nm:>6.0f}  {v:>10s}  {p:>8.2f}  {rmse:>10.3f}  "
              f"{loss_final:>10.3e}  {dt:>7.1f}")
        row[v] = dict(psnr=p, rmse_nm=rmse, loss=loss_final, t=dt)
        keep[v] = h_rec.cpu().numpy() * 1e9
    results.append(row)
    recon_keep[float(A_nm)] = keep

# ---------------- plot ----------------
plt.rcParams.update(STYLE)
fig = plt.figure(figsize=(16, 11))
gs = fig.add_gridspec(3, len(variants))
fig.suptitle("Reconstruction with positivity constraint  (h >= 0)",
             fontsize=13, color="#e6edf3")
fig.patch.set_facecolor("#0d1117")

def _style(ax):
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e", labelsize=8)
    ax.grid(color="#30363d", lw=0.4, ls=":")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

# (Top row) PSNR vs amplitude per variant
ax = fig.add_subplot(gs[0, :])
A_axis = np.array([r["A_nm"] for r in results])
colors = {"baseline":"#8b949e", "relu":"#58a6ff",
          "softplus":"#3fb950", "squared":"#f78166"}
for v in variants:
    ax.plot(A_axis, [r[v]["psnr"] for r in results], "o-",
            color=colors[v], lw=1.4, label=v)
ax.axvline(PHASE_WRAP_NM, color="#f85149", ls="--", lw=1,
           label=f"phase-wrap lam/4 = {PHASE_WRAP_NM:.0f} nm")
ax.axhline(100, color="#8b949e", ls=":", lw=0.8, label="100 dB target")
ax.set_xscale("log")
ax.set_xlabel("amplitude A [nm]", fontsize=10, color="#8b949e")
ax.set_ylabel("PSNR [dB]", fontsize=10, color="#8b949e")
ax.set_title("PSNR per reparameterization", fontsize=10, color="#e6edf3")
ax.legend(fontsize=8, labelcolor="#e6edf3", framealpha=0.3, ncol=3)
_style(ax)

# (Middle row) cross-section at small pre-wrap A (e.g. 50 nm)
A_small = float(A_nm_all[2])  # 50 nm
mid = MEM // 2
m_um = (np.arange(MEM) - (MEM-1)*0.5) * dx * 1e6
for j, v in enumerate(variants):
    ax = fig.add_subplot(gs[1, j])
    keep = recon_keep[A_small]
    ax.plot(m_um, keep["h_true"][mid,:], color="#8b949e", lw=2, ls=":",
            label="GT")
    ax.plot(m_um, keep[v][mid,:], color=colors[v], lw=1.2, label=v)
    p = next(r for r in results if r["A_nm"] == A_small)[v]["psnr"]
    ax.set_title(f"A={A_small:.0f} nm  {v}\nPSNR={p:.1f} dB",
                 fontsize=9, color="#e6edf3")
    ax.set_xlabel("x [um]", fontsize=8, color="#8b949e")
    ax.set_ylabel("h [nm]", fontsize=8, color="#8b949e")
    ax.legend(fontsize=7, labelcolor="#e6edf3", framealpha=0.3)
    _style(ax)

# (Bottom row) cross-section at phase-wrap A
A_big = float(A_nm_all[-2])  # 300 nm
for j, v in enumerate(variants):
    ax = fig.add_subplot(gs[2, j])
    keep = recon_keep[A_big]
    ax.plot(m_um, keep["h_true"][mid,:], color="#8b949e", lw=2, ls=":",
            label="GT")
    ax.plot(m_um, keep[v][mid,:], color=colors[v], lw=1.2, label=v)
    p = next(r for r in results if r["A_nm"] == A_big)[v]["psnr"]
    ax.set_title(f"A={A_big:.0f} nm  {v}\nPSNR={p:.1f} dB",
                 fontsize=9, color="#e6edf3")
    ax.set_xlabel("x [um]", fontsize=8, color="#8b949e")
    ax.set_ylabel("h [nm]", fontsize=8, color="#8b949e")
    ax.legend(fontsize=7, labelcolor="#e6edf3", framealpha=0.3)
    _style(ax)

plt.tight_layout()
plt.savefig(OUT / "constrained_summary.png", dpi=150, bbox_inches="tight",
            facecolor="#0d1117")
plt.close()

# ---------------- text + json ----------------
lines = ["reconstruction_constrained.py — h>=0 ambiguity test",
         "=" * 60,
         f"  wavelength  = {lam*1e9:.1f} nm",
         f"  mem pitch   = {dx*1e6:.1f} um",
         f"  distance    = {DIST*1e3:.1f} mm",
         f"  sigma bump  = {SIGMA*1e6:.0f} um",
         f"  phase-wrap  = {PHASE_WRAP_NM:.2f} nm",
         f"  algo: Adam lr={LR}, {N_ITER} iters, cosine decay",
         "",
         f"  {'A[nm]':>6}  {'regime':>10}  " +
         "  ".join(f"{v:>10}" for v in variants) + "   PSNR [dB]"]
for r in results:
    lines.append(f"  {r['A_nm']:>6.0f}  {r['regime']:>10s}  " +
                 "  ".join(f"{r[v]['psnr']:>10.2f}" for v in variants))
lines += ["",
          "Interpretation",
          "  * Baseline (signed h) shows the in-line twin-image ambiguity:",
          "    loss converges to ~0 but reconstructed h is sign-flipped /",
          "    twinned, capping PSNR around 15 dB regardless of amplitude.",
          "  * relu / softplus / squared all force h >= 0; whichever achieves",
          "    PSNR > 60 dB in the pre-wrap regime (A < lam/4) is the right",
          "    constraint -- algorithm is correct, ambiguity was the bottleneck.",
          "  * In the wrap regime (A > lam/4) PSNR is expected to plateau even",
          "    with positivity, because per-pixel modulo-lam/2 ambiguity is a",
          "    second, independent ill-posedness.  That is addressed only by",
          "    finer spatial sampling (smaller mem_pitch -> NVIDIA setup)."]
txt = "\n".join(lines)
(OUT / "constrained_metrics.txt").write_text(txt, encoding="utf-8")
print("\n" + txt)

(OUT / "constrained_metrics.json").write_text(json.dumps(dict(
    config=dict(wavelength_nm=lam*1e9, mem_pitch_um=dx*1e6, distance_mm=DIST*1e3,
                mem=MEM, cmos=CMOS, grid=GRID, sigma_um=SIGMA*1e6,
                lr=LR, n_iter=N_ITER, phase_wrap_pixel_nm=PHASE_WRAP_NM),
    results=results,
), indent=2))
print(f"\nSaved -> {OUT/'constrained_summary.png'}")
