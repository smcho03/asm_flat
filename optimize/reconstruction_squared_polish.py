"""
reconstruction_squared_polish.py
--------------------------------
Follow-up to reconstruction_constrained.py.

The squared reparameterization (h = h_var^2 * H_SCALE, h >= 0) raised
the ceiling from 15 dB to ~34 dB but did not reach 100 dB even in the
pre-wrap regime.  This script tests whether longer optimization +
LBFGS polish closes that gap.

Three configurations are run for each amplitude:
  (a) Adam 3k     (baseline from constrained run)
  (b) Adam 15k    (long run, cosine decay)
  (c) Adam 15k -> LBFGS polish (line-search Newton on top)

Output:
  output/reconstruction_squared_polish/
    polish_summary.png
    polish_metrics.txt
    polish_metrics.json
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
OUT  = ROOT / "output" / "reconstruction_squared_polish"
OUT.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

MEM, CMOS, GRID, DIST = 128, 256, 384, 5e-3
SIGMA = 200e-6
LR    = 3e-3
H_SCALE_FACTOR = 2.0

lam = wavelength
dx  = mem_pitch
PHASE_WRAP_NM = lam / 4.0 * 1e9

sensor = HolographicSensor(
    wavelength=lam, mem_res=MEM, mem_pitch=dx,
    cmos_res=CMOS, grid_res=GRID, distance=DIST, device=device,
).to(device)

A_nm_all = np.array([20, 50, 100, 150, 200, 300, 500])

def adam_run(I_tgt, h_scale, n_iter, lr, raw_init=0.2):
    raw = torch.full((MEM, MEM), raw_init, device=device,
                     dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([raw], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iter)
    for _ in range(n_iter):
        opt.zero_grad()
        h = (raw * raw) * h_scale
        loss = torch.mean((sensor(h) - I_tgt)**2)
        loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        h_final = (raw * raw) * h_scale
    return raw.detach(), h_final, float(loss.detach())

def lbfgs_polish(raw_init, I_tgt, h_scale, max_iter=400):
    raw = raw_init.clone().detach().requires_grad_(True)
    opt = torch.optim.LBFGS([raw], lr=1.0, max_iter=max_iter,
                             tolerance_grad=1e-14,
                             tolerance_change=1e-14,
                             line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        h = (raw * raw) * h_scale
        loss = torch.mean((sensor(h) - I_tgt)**2)
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        h_final = (raw * raw) * h_scale
        loss = torch.mean((sensor(h_final) - I_tgt)**2).item()
    return h_final, loss

def psnr_db(a, b):
    mse = float(torch.mean((a - b)**2))
    peak = float(b.abs().max())
    if mse < 1e-30: return 200.0
    if peak < 1e-30: return 0.0
    return 10.0 * np.log10(peak**2 / mse)

results = []
recon_keep = {}
print(f"\nphase-wrap pixel limit lam/4 = {PHASE_WRAP_NM:.2f} nm")
print(f"\n{'A[nm]':>6}  {'algo':>14}  {'PSNR':>8}  {'RMSE[nm]':>10}  "
      f"{'loss':>10}  {'time[s]':>7}")

for A_nm in A_nm_all:
    A_m = float(A_nm) * 1e-9
    h_scale = max(H_SCALE_FACTOR * A_m, 50e-9)
    h_true = gaussian_bump(N=MEM, dx=dx, amplitude=A_m, sigma=SIGMA, device=device)
    with torch.no_grad():
        I_tgt = sensor(h_true)
    regime = "pre-wrap" if A_nm < PHASE_WRAP_NM else "wrap"
    row = dict(A_nm=float(A_nm), regime=regime)
    keep = dict(h_true=h_true.cpu().numpy()*1e9)

    # (a) Adam 3k
    t0 = time.time()
    raw_a, h_a, _ = adam_run(I_tgt, h_scale, n_iter=3000, lr=LR)
    pa = psnr_db(h_a, h_true); rmsa = float(torch.sqrt(torch.mean((h_a-h_true)**2)))*1e9
    la = float(torch.mean((sensor(h_a) - I_tgt)**2))
    print(f"{A_nm:>6.0f}  {'adam-3k':>14s}  {pa:>8.2f}  {rmsa:>10.3f}  "
          f"{la:>10.3e}  {time.time()-t0:>7.1f}")
    row["adam3k"] = dict(psnr=pa, rmse_nm=rmsa, loss=la, t=time.time()-t0)
    keep["adam3k"] = h_a.cpu().numpy()*1e9

    # (b) Adam 15k
    t0 = time.time()
    raw_b, h_b, _ = adam_run(I_tgt, h_scale, n_iter=15000, lr=LR)
    pb = psnr_db(h_b, h_true); rmsb = float(torch.sqrt(torch.mean((h_b-h_true)**2)))*1e9
    lb = float(torch.mean((sensor(h_b) - I_tgt)**2))
    print(f"{A_nm:>6.0f}  {'adam-15k':>14s}  {pb:>8.2f}  {rmsb:>10.3f}  "
          f"{lb:>10.3e}  {time.time()-t0:>7.1f}")
    row["adam15k"] = dict(psnr=pb, rmse_nm=rmsb, loss=lb, t=time.time()-t0)
    keep["adam15k"] = h_b.cpu().numpy()*1e9

    # (c) Adam 15k + LBFGS polish
    t0 = time.time()
    h_c, lc = lbfgs_polish(raw_b, I_tgt, h_scale, max_iter=400)
    pc = psnr_db(h_c, h_true); rmsc = float(torch.sqrt(torch.mean((h_c-h_true)**2)))*1e9
    print(f"{A_nm:>6.0f}  {'adam15k+lbfgs':>14s}  {pc:>8.2f}  {rmsc:>10.3f}  "
          f"{lc:>10.3e}  {time.time()-t0:>7.1f}")
    row["polish"] = dict(psnr=pc, rmse_nm=rmsc, loss=lc, t=time.time()-t0)
    keep["polish"] = h_c.cpu().numpy()*1e9

    results.append(row)
    recon_keep[float(A_nm)] = keep

# ---------------- plot ----------------
plt.rcParams.update(STYLE)
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("Squared reparam + long Adam + LBFGS polish",
             fontsize=13, color="#e6edf3")
fig.patch.set_facecolor("#0d1117")

def _style(ax):
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e", labelsize=8)
    ax.grid(color="#30363d", lw=0.4, ls=":")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

A_axis = np.array([r["A_nm"] for r in results])
ax = axes[0,0]
ax.plot(A_axis, [r["adam3k"]["psnr"] for r in results], "o-",
        color="#58a6ff", lw=1.4, label="Adam 3k")
ax.plot(A_axis, [r["adam15k"]["psnr"] for r in results], "s-",
        color="#3fb950", lw=1.4, label="Adam 15k")
ax.plot(A_axis, [r["polish"]["psnr"] for r in results], "^-",
        color="#f78166", lw=1.6, label="Adam 15k + LBFGS")
ax.axvline(PHASE_WRAP_NM, color="#f85149", ls="--", lw=1,
           label=f"phase-wrap {PHASE_WRAP_NM:.0f} nm")
ax.axhline(100, color="#8b949e", ls=":", lw=0.8, label="100 dB target")
ax.set_xscale("log")
ax.set_xlabel("amplitude A [nm]", fontsize=10, color="#8b949e")
ax.set_ylabel("PSNR [dB]", fontsize=10, color="#8b949e")
ax.set_title("(A) PSNR vs amplitude", fontsize=10, color="#e6edf3")
ax.legend(fontsize=8, labelcolor="#e6edf3", framealpha=0.3)
_style(ax)

ax = axes[0,1]
ax.plot(A_axis, [r["adam3k"]["loss"] for r in results], "o-",
        color="#58a6ff", lw=1.4, label="Adam 3k")
ax.plot(A_axis, [r["adam15k"]["loss"] for r in results], "s-",
        color="#3fb950", lw=1.4, label="Adam 15k")
ax.plot(A_axis, [r["polish"]["loss"] for r in results], "^-",
        color="#f78166", lw=1.6, label="LBFGS polish")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("amplitude A [nm]", fontsize=10, color="#8b949e")
ax.set_ylabel("final loss", fontsize=10, color="#8b949e")
ax.set_title("(B) Final loss vs amplitude", fontsize=10, color="#e6edf3")
ax.legend(fontsize=8, labelcolor="#e6edf3", framealpha=0.3)
_style(ax)

# pre-wrap small bump cross-section
A_show = float(A_nm_all[2])  # 100 nm
mid = MEM // 2
m_um = (np.arange(MEM) - (MEM-1)*0.5) * dx * 1e6
ax = axes[1,0]
keep = recon_keep[A_show]
ax.plot(m_um, keep["h_true"][mid,:], color="#8b949e", lw=2, ls=":", label="GT")
ax.plot(m_um, keep["adam3k"][mid,:],  color="#58a6ff", lw=1.0, label="Adam 3k")
ax.plot(m_um, keep["adam15k"][mid,:], color="#3fb950", lw=1.0, label="Adam 15k")
ax.plot(m_um, keep["polish"][mid,:],  color="#f78166", lw=1.2, label="LBFGS polish")
ax.set_xlabel("x [um]", fontsize=10, color="#8b949e")
ax.set_ylabel("h [nm]", fontsize=10, color="#8b949e")
ax.set_title(f"(C) Pre-wrap (A={A_show:.0f} nm) cross-section",
             fontsize=10, color="#e6edf3")
ax.legend(fontsize=8, labelcolor="#e6edf3", framealpha=0.3)
_style(ax)

A_show2 = float(A_nm_all[-1])
ax = axes[1,1]
keep = recon_keep[A_show2]
ax.plot(m_um, keep["h_true"][mid,:], color="#8b949e", lw=2, ls=":", label="GT")
ax.plot(m_um, keep["adam3k"][mid,:],  color="#58a6ff", lw=1.0, label="Adam 3k")
ax.plot(m_um, keep["adam15k"][mid,:], color="#3fb950", lw=1.0, label="Adam 15k")
ax.plot(m_um, keep["polish"][mid,:],  color="#f78166", lw=1.2, label="LBFGS polish")
ax.set_xlabel("x [um]", fontsize=10, color="#8b949e")
ax.set_ylabel("h [nm]", fontsize=10, color="#8b949e")
ax.set_title(f"(D) Phase-wrap (A={A_show2:.0f} nm) cross-section",
             fontsize=10, color="#e6edf3")
ax.legend(fontsize=8, labelcolor="#e6edf3", framealpha=0.3)
_style(ax)

plt.tight_layout()
plt.savefig(OUT / "polish_summary.png", dpi=150, bbox_inches="tight",
            facecolor="#0d1117")
plt.close()

lines = ["reconstruction_squared_polish.py — push toward 100 dB",
         "=" * 60,
         f"  wavelength  = {lam*1e9:.1f} nm",
         f"  mem pitch   = {dx*1e6:.1f} um",
         f"  distance    = {DIST*1e3:.1f} mm",
         f"  sigma bump  = {SIGMA*1e6:.0f} um",
         f"  phase-wrap  = {PHASE_WRAP_NM:.2f} nm",
         f"  reparam     = h = (h_var)^2 * H_SCALE  (h >= 0)",
         "",
         f"  {'A[nm]':>6}  {'regime':>10}  {'adam3k':>10}  "
         f"{'adam15k':>10}  {'polish':>10}   PSNR [dB]"]
for r in results:
    lines.append(f"  {r['A_nm']:>6.0f}  {r['regime']:>10s}  "
                 f"{r['adam3k']['psnr']:>10.2f}  "
                 f"{r['adam15k']['psnr']:>10.2f}  "
                 f"{r['polish']['psnr']:>10.2f}")
lines += ["",
          "Take-aways",
          "  * If LBFGS polish drives PSNR -> 100 dB in pre-wrap regime,",
          "    the squared-reparam optimizer is the right algorithm and",
          "    no additional physics (multi-distance / synthetic-lambda)",
          "    is needed for that regime.",
          "  * If PSNR plateaus below 100 dB even after polish, there is a",
          "    residual ambiguity beyond sign flip (likely constant-offset",
          "    or local twin patches) that needs further investigation.",
          "  * Phase-wrap regime is expected to plateau regardless;",
          "    the answer there is finer dx (NVIDIA setup)."]
txt = "\n".join(lines)
(OUT / "polish_metrics.txt").write_text(txt, encoding="utf-8")
print("\n" + txt)

(OUT / "polish_metrics.json").write_text(json.dumps(dict(
    config=dict(wavelength_nm=lam*1e9, mem_pitch_um=dx*1e6, distance_mm=DIST*1e3,
                mem=MEM, cmos=CMOS, grid=GRID, sigma_um=SIGMA*1e6,
                lr=LR, phase_wrap_pixel_nm=PHASE_WRAP_NM),
    results=results,
), indent=2))
print(f"\nSaved -> {OUT/'polish_summary.png'}")
