"""
reconstruction_perfectness.py
-----------------------------
Two-stage diagnostic of the reconstruction algorithm in a noiseless
simulation:

  Stage 1  Pre-wrap regime (h_max < lam/4 = 158 nm)
           Forward operator is approximately linear and the inverse
           problem is well-posed.  In a noiseless sim the algorithm
           SHOULD reach near machine-precision PSNR (~100 dB).

  Stage 2  Phase-wrap regime (h_max > lam/4)
           Forward is genuinely non-injective due to modulo-lam/2
           per-pixel ambiguity in I = |U|^2.  Algorithm performance
           will plateau or degrade unless we add disambiguation.

For both stages we test three reconstruction strategies of increasing
strength to separate algorithm failure from physical ill-posedness:

  (a) Adam baseline              (current default, lr=3e-3, 1500 iter)
  (b) Adam long                  (lr=3e-3, 10000 iter, cosine LR decay)
  (c) LBFGS refine               (Adam warm-start -> LBFGS polish)

Output:
  output/reconstruction_perfectness/
    perfectness_summary.png
    perfectness_metrics.txt
    perfectness_metrics.json
"""

import sys, time, json, functools
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print = functools.partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).parent.parent))
from sensor_model  import HolographicSensor, gaussian_bump
from sensor_params import wavelength, mem_pitch
from sensor_utils  import STYLE

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "output" / "reconstruction_perfectness"
OUT.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

# ---------------- sensor / config ----------------
MEM   = 128
CMOS  = 256
GRID  = 384
DIST  = 5e-3
SIGMA = 200e-6                 # smooth bump → gradient stays small
H_SCALE_FACTOR = 2.0           # per-amplitude scaling
LR    = 3e-3

lam = wavelength
dx  = mem_pitch
PHASE_WRAP_NM = lam / 4.0 * 1e9   # 158.20 nm

sensor = HolographicSensor(
    wavelength=lam, mem_res=MEM, mem_pitch=dx,
    cmos_res=CMOS, grid_res=GRID, distance=DIST, device=device,
).to(device)

# ---------------- amplitude grid spanning both regimes ----------------
A_nm_pre  = np.array([5, 20, 50, 100, 150])           # below lam/4
A_nm_post = np.array([200, 300, 500, 800, 1500])      # above lam/4
A_nm_all  = np.concatenate([A_nm_pre, A_nm_post])

def psnr_db(a, b):
    mse = float(torch.mean((a - b)**2))
    peak = float(b.abs().max())
    if mse < 1e-30: return 200.0
    if peak < 1e-30: return 0.0
    return 10.0 * np.log10(peak**2 / mse)

# ---------------- algorithms ----------------
def adam_run(I_tgt, h_scale, lr=LR, n_iter=1500, decay=False):
    h_var = torch.zeros(MEM, MEM, dtype=torch.float32, device=device,
                        requires_grad=True)
    opt = torch.optim.Adam([h_var], lr=lr)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iter)
             if decay else None)
    for _ in range(n_iter):
        opt.zero_grad()
        loss = torch.mean((sensor(h_var * h_scale) - I_tgt)**2)
        loss.backward(); opt.step()
        if sched is not None: sched.step()
    return h_var.detach() * h_scale, float(loss.detach())

def lbfgs_polish(h_init, I_tgt, h_scale, max_iter=200):
    h_var = (h_init / h_scale).clone().detach().requires_grad_(True)
    opt = torch.optim.LBFGS([h_var], lr=1.0, max_iter=max_iter,
                             tolerance_grad=1e-12, tolerance_change=1e-12,
                             line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        loss = torch.mean((sensor(h_var * h_scale) - I_tgt)**2)
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        loss = torch.mean((sensor(h_var * h_scale) - I_tgt)**2).item()
    return h_var.detach() * h_scale, loss

# ---------------- main sweep ----------------
results = []
print(f"\nphase-wrap pixel limit lam/4 = {PHASE_WRAP_NM:.2f} nm")
print(f"\n{'A[nm]':>6} {'regime':>10} {'algo':>10} "
      f"{'PSNR':>8} {'RMSE[nm]':>10} {'final loss':>11} {'time[s]':>7}")
recon_traces = {}
for A_nm in A_nm_all:
    A_m = float(A_nm) * 1e-9
    h_scale = max(H_SCALE_FACTOR * A_m, 50e-9)
    h_true = gaussian_bump(N=MEM, dx=dx, amplitude=A_m, sigma=SIGMA, device=device)
    with torch.no_grad():
        I_tgt = sensor(h_true)
    regime = "pre-wrap" if A_nm < PHASE_WRAP_NM else "wrap"

    # (a) Adam baseline
    t0 = time.time()
    h_a, La = adam_run(I_tgt, h_scale, lr=LR, n_iter=1500, decay=False)
    pa = psnr_db(h_a, h_true); rmsa = float(torch.sqrt(torch.mean((h_a-h_true)**2)))*1e9
    dta = time.time()-t0

    # (b) Adam long + cosine decay
    t0 = time.time()
    h_b, Lb = adam_run(I_tgt, h_scale, lr=LR, n_iter=10000, decay=True)
    pb = psnr_db(h_b, h_true); rmsb = float(torch.sqrt(torch.mean((h_b-h_true)**2)))*1e9
    dtb = time.time()-t0

    # (c) LBFGS polish (warm-start from b)
    t0 = time.time()
    h_c, Lc = lbfgs_polish(h_b, I_tgt, h_scale, max_iter=300)
    pc = psnr_db(h_c, h_true); rmsc = float(torch.sqrt(torch.mean((h_c-h_true)**2)))*1e9
    dtc = time.time()-t0

    print(f"{A_nm:>6.0f} {regime:>10s} {'adam-1500':>10s} "
          f"{pa:>8.2f} {rmsa:>10.3f} {La:>11.3e} {dta:>7.1f}")
    print(f"{A_nm:>6.0f} {regime:>10s} {'adam-10k':>10s} "
          f"{pb:>8.2f} {rmsb:>10.3f} {Lb:>11.3e} {dtb:>7.1f}")
    print(f"{A_nm:>6.0f} {regime:>10s} {'lbfgs':>10s} "
          f"{pc:>8.2f} {rmsc:>10.3f} {Lc:>11.3e} {dtc:>7.1f}")

    results.append(dict(
        A_nm=float(A_nm), regime=regime,
        adam1500=dict(psnr=pa, rmse_nm=rmsa, loss=La, t=dta),
        adam10k =dict(psnr=pb, rmse_nm=rmsb, loss=Lb, t=dtb),
        lbfgs   =dict(psnr=pc, rmse_nm=rmsc, loss=Lc, t=dtc),
    ))
    recon_traces[float(A_nm)] = dict(
        h_true = h_true.cpu().numpy()*1e9,
        h_a    = h_a.cpu().numpy()*1e9,
        h_b    = h_b.cpu().numpy()*1e9,
        h_c    = h_c.cpu().numpy()*1e9,
    )

# ---------------- plots ----------------
plt.rcParams.update(STYLE)
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("Reconstruction perfectness — pre-wrap vs phase-wrap regimes",
             fontsize=13, color="#e6edf3")
fig.patch.set_facecolor("#0d1117")

def _style(ax):
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e", labelsize=8)
    ax.grid(color="#30363d", lw=0.4, ls=":")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

# (A) PSNR vs amplitude  (per algorithm)
ax = axes[0,0]
A_axis = np.array([r["A_nm"] for r in results])
ax.plot(A_axis, [r["adam1500"]["psnr"] for r in results], "o-",
        color="#58a6ff", lw=1.2, label="Adam 1500")
ax.plot(A_axis, [r["adam10k" ]["psnr"] for r in results], "s-",
        color="#3fb950", lw=1.2, label="Adam 10k + cosine")
ax.plot(A_axis, [r["lbfgs"   ]["psnr"] for r in results], "^-",
        color="#f78166", lw=1.4, label="Adam→LBFGS")
ax.axvline(PHASE_WRAP_NM, color="#f85149", ls="--", lw=1,
           label=f"phase-wrap limit lam/4 = {PHASE_WRAP_NM:.0f} nm")
ax.axhline(100, color="#8b949e", ls=":", lw=0.8, label="100 dB target")
ax.set_xscale("log")
ax.set_xlabel("amplitude A [nm]", fontsize=10, color="#8b949e")
ax.set_ylabel("PSNR [dB]", fontsize=10, color="#8b949e")
ax.set_title("(A) PSNR vs amplitude", fontsize=10, color="#e6edf3")
ax.legend(fontsize=8, labelcolor="#e6edf3", framealpha=0.3)
_style(ax)

# (B) loss
ax = axes[0,1]
ax.plot(A_axis, [r["adam1500"]["loss"] for r in results], "o-",
        color="#58a6ff", lw=1.2, label="Adam 1500")
ax.plot(A_axis, [r["adam10k" ]["loss"] for r in results], "s-",
        color="#3fb950", lw=1.2, label="Adam 10k")
ax.plot(A_axis, [r["lbfgs"   ]["loss"] for r in results], "^-",
        color="#f78166", lw=1.4, label="LBFGS")
ax.axvline(PHASE_WRAP_NM, color="#f85149", ls="--", lw=1)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("amplitude A [nm]", fontsize=10, color="#8b949e")
ax.set_ylabel("final loss (MSE on I)", fontsize=10, color="#8b949e")
ax.set_title("(B) Final loss vs amplitude", fontsize=10, color="#e6edf3")
ax.legend(fontsize=8, labelcolor="#e6edf3", framealpha=0.3)
_style(ax)

# (C) representative pre-wrap recon  (smallest A)
ax = axes[1,0]
A_show_pre = float(A_nm_pre[2])  # 50 nm
tr = recon_traces[A_show_pre]
mid = MEM // 2
m_um = (np.arange(MEM) - (MEM-1)*0.5) * dx * 1e6
ax.plot(m_um, tr["h_true"][mid,:], color="#8b949e", lw=2, ls=":", label="GT")
ax.plot(m_um, tr["h_a"][mid,:],    color="#58a6ff", lw=1.0, label="Adam 1500")
ax.plot(m_um, tr["h_b"][mid,:],    color="#3fb950", lw=1.0, label="Adam 10k")
ax.plot(m_um, tr["h_c"][mid,:],    color="#f78166", lw=1.0, label="LBFGS")
ax.set_xlabel("x [um]", fontsize=10, color="#8b949e")
ax.set_ylabel("h [nm]", fontsize=10, color="#8b949e")
ax.set_title(f"(C) Pre-wrap example (A={A_show_pre:.0f} nm)",
             fontsize=10, color="#e6edf3")
ax.legend(fontsize=8, labelcolor="#e6edf3", framealpha=0.3)
_style(ax)

# (D) representative phase-wrap recon  (largest A)
ax = axes[1,1]
A_show_post = float(A_nm_post[-1])
tr = recon_traces[A_show_post]
ax.plot(m_um, tr["h_true"][mid,:], color="#8b949e", lw=2, ls=":", label="GT")
ax.plot(m_um, tr["h_a"][mid,:],    color="#58a6ff", lw=1.0, label="Adam 1500")
ax.plot(m_um, tr["h_b"][mid,:],    color="#3fb950", lw=1.0, label="Adam 10k")
ax.plot(m_um, tr["h_c"][mid,:],    color="#f78166", lw=1.0, label="LBFGS")
ax.set_xlabel("x [um]", fontsize=10, color="#8b949e")
ax.set_ylabel("h [nm]", fontsize=10, color="#8b949e")
ax.set_title(f"(D) Phase-wrap example (A={A_show_post:.0f} nm)",
             fontsize=10, color="#e6edf3")
ax.legend(fontsize=8, labelcolor="#e6edf3", framealpha=0.3)
_style(ax)

plt.tight_layout()
plt.savefig(OUT / "perfectness_summary.png", dpi=150, bbox_inches="tight",
            facecolor="#0d1117")
plt.close()

# ---------------- text + json ----------------
lines = ["reconstruction_perfectness.py — algorithm vs physics diagnostic",
         "=" * 64,
         f"  wavelength       = {lam*1e9:.1f} nm",
         f"  membrane pitch   = {dx*1e6:.1f} um",
         f"  distance         = {DIST*1e3:.1f} mm",
         f"  Gaussian sigma   = {SIGMA*1e6:.0f} um (smooth, low gradient)",
         f"  phase-wrap limit = {PHASE_WRAP_NM:.2f} nm",
         "",
         f"  {'A[nm]':>6} {'regime':>10} "
         f"{'adam1500':>10} {'adam10k':>10} {'lbfgs':>10}   PSNR [dB]"]
for r in results:
    lines.append(f"  {r['A_nm']:>6.0f} {r['regime']:>10s} "
                 f"{r['adam1500']['psnr']:>10.2f} "
                 f"{r['adam10k' ]['psnr']:>10.2f} "
                 f"{r['lbfgs'   ]['psnr']:>10.2f}")
lines += ["",
          "Interpretation",
          "  * In the pre-wrap regime the strongest algorithm should achieve",
          "    PSNR >> 100 dB.  If not, the bottleneck is the algorithm.",
          "  * In the phase-wrap regime the per-pixel modulo-lam/2 ambiguity",
          "    in I = |U|^2 makes the inverse genuinely non-unique.  No",
          "    single-shot algorithm can reach 100 dB without disambiguation",
          "    (multi-distance, multi-wavelength / synthetic-wavelength,",
          "     phase-shifting, or temporal tracking)."]
txt = "\n".join(lines)
(OUT / "perfectness_metrics.txt").write_text(txt, encoding="utf-8")
print("\n" + txt)

(OUT / "perfectness_metrics.json").write_text(json.dumps(dict(
    config=dict(wavelength_nm=lam*1e9, mem_pitch_um=dx*1e6, distance_mm=DIST*1e3,
                mem=MEM, cmos=CMOS, grid=GRID, sigma_um=SIGMA*1e6,
                lr=LR, phase_wrap_pixel_nm=PHASE_WRAP_NM),
    results=results,
), indent=2))
print(f"\nSaved -> {OUT/'perfectness_summary.png'}")
