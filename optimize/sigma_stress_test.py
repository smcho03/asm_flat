"""
sigma_stress_test.py
--------------------
Stress test of the corrected algorithm (squared reparam + Adam 15k).

Up to now we showed 50 dB / sub-nm RMSE for sigma = 200 um at A up to
500 nm. That works because the per-pixel slope stays well below lam/4.

Here we sweep sigma DOWN at fixed A = 200 nm to find where the
per-pixel phase-wrap actually bites.  Theoretical onset:
    slope_max = A / (sigma * sqrt(e))
    per-pixel = slope_max * dx
    onset when per-pixel = lam/4 = 158 nm
    -> sigma_onset = A * dx / (lam/4 * sqrt(e))
                   = 200e-9 * 10e-6 / (158e-9 * 1.65) = 7.7 um

So sigma below ~10 um at A=200 nm should start failing.  We sweep
sigma in [200, 100, 60, 40, 25, 15, 10] um.

Output:
  output/sigma_stress_test/
    sigma_stress.png
    sigma_stress.txt
    sigma_stress.json
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
OUT  = ROOT / "output" / "sigma_stress_test"
OUT.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

MEM, CMOS, GRID, DIST = 128, 256, 384, 5e-3
A_NM   = 200
A_M    = A_NM * 1e-9
H_SCALE = 2.0 * A_M
LR     = 3e-3
N_ITER = 5000             # Adam steps; 5k is a good cost/quality balance

lam = wavelength
dx  = mem_pitch
PHASE_WRAP_NM = lam / 4.0 * 1e9

sensor = HolographicSensor(
    wavelength=lam, mem_res=MEM, mem_pitch=dx,
    cmos_res=CMOS, grid_res=GRID, distance=DIST, device=device,
).to(device)

sigma_um_list = [200, 100, 60, 40, 25, 15, 10]

def reconstruct(I_tgt):
    raw = torch.full((MEM, MEM), 0.2, device=device,
                     dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([raw], lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_ITER)
    for _ in range(N_ITER):
        opt.zero_grad()
        h = (raw * raw) * H_SCALE
        loss = torch.mean((sensor(h) - I_tgt)**2)
        loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        h_final = (raw * raw) * H_SCALE
        l = float(torch.mean((sensor(h_final) - I_tgt)**2))
    return h_final, l

def psnr_db(a, b):
    mse = float(torch.mean((a - b)**2))
    peak = float(b.abs().max())
    if mse < 1e-30: return 200.0
    if peak < 1e-30: return 0.0
    return 10.0 * np.log10(peak**2 / mse)

results = []
recon_keep = {}
print(f"\nA = {A_NM} nm,  phase-wrap pixel limit = {PHASE_WRAP_NM:.2f} nm")
print(f"\n{'sigma[um]':>10} {'slope':>8} {'pix [nm]':>10} {'wrap×':>7} "
      f"{'PSNR':>8} {'RMSE[nm]':>10} {'time[s]':>7}")
for sg_um in sigma_um_list:
    sg_m = sg_um * 1e-6
    slope = A_M / (sg_m * np.sqrt(np.e))
    per_pix_nm = slope * dx * 1e9
    wrap_x = per_pix_nm / PHASE_WRAP_NM
    h_true = gaussian_bump(N=MEM, dx=dx, amplitude=A_M, sigma=sg_m, device=device)
    with torch.no_grad():
        I_tgt = sensor(h_true)
    t0 = time.time()
    h_rec, loss = reconstruct(I_tgt)
    p = psnr_db(h_rec, h_true)
    rmse = float(torch.sqrt(torch.mean((h_rec - h_true)**2))) * 1e9
    dt = time.time() - t0
    print(f"{sg_um:>10.0f} {slope:>8.4f} {per_pix_nm:>10.2f} {wrap_x:>7.2f} "
          f"{p:>8.2f} {rmse:>10.3f} {dt:>7.1f}")
    results.append(dict(sigma_um=sg_um, slope=float(slope),
                        per_pixel_nm=per_pix_nm, wrap_multiple=wrap_x,
                        psnr=p, rmse_nm=rmse, loss=loss, t=dt))
    recon_keep[sg_um] = dict(h_true=h_true.cpu().numpy()*1e9,
                              h_rec=h_rec.cpu().numpy()*1e9)

# ---------------- plot ----------------
plt.rcParams.update(STYLE)
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 4)
fig.suptitle(f"sigma stress test  (A={A_NM} nm, mem_pitch={dx*1e6:.0f} um)",
             fontsize=13, color="#e6edf3")
fig.patch.set_facecolor("#0d1117")

def _style(ax):
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e", labelsize=8)
    ax.grid(color="#30363d", lw=0.4, ls=":")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

# Top row: PSNR vs sigma + wrap multiple
ax = fig.add_subplot(gs[0, :2])
sg = np.array([r["sigma_um"] for r in results])
ax.plot(sg, [r["psnr"] for r in results], "o-",
        color="#3fb950", lw=1.6, label="PSNR")
ax.set_xscale("log")
ax.set_xlabel("sigma [um]", fontsize=10, color="#8b949e")
ax.set_ylabel("PSNR [dB]", fontsize=10, color="#8b949e",
              color_=None) if False else ax.set_ylabel("PSNR [dB]", fontsize=10, color="#8b949e")
ax.set_title("(A) PSNR vs sigma  (squared+Adam 5k)",
             fontsize=10, color="#e6edf3")
_style(ax)

ax = fig.add_subplot(gs[0, 2:])
ax.plot(sg, [r["wrap_multiple"] for r in results], "s-",
        color="#f78166", lw=1.6, label="wrap × (per-pix / lam/4)")
ax.axhline(1.0, color="#e3b341", ls="--", lw=0.8, label="onset (=1)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("sigma [um]", fontsize=10, color="#8b949e")
ax.set_ylabel("per-pixel / (lam/4)", fontsize=10, color="#8b949e")
ax.set_title("(B) Theoretical per-pixel wrap multiple",
             fontsize=10, color="#e6edf3")
ax.legend(fontsize=8, labelcolor="#e6edf3", framealpha=0.3)
_style(ax)

# Bottom row: cross-sections at 4 sigmas
mid = MEM // 2
m_um = (np.arange(MEM) - (MEM-1)*0.5) * dx * 1e6
show_idx = [0, 2, 4, 6]   # 200, 60, 25, 10 um
for j, idx in enumerate(show_idx):
    sg_um = sigma_um_list[idx]
    keep = recon_keep[sg_um]
    p = results[idx]["psnr"]
    wx = results[idx]["wrap_multiple"]
    ax = fig.add_subplot(gs[1, j])
    ax.plot(m_um, keep["h_true"][mid,:], color="#8b949e", lw=2, ls=":", label="GT")
    ax.plot(m_um, keep["h_rec" ][mid,:], color="#58a6ff", lw=1.2, label="rec")
    ax.set_title(f"sigma={sg_um} um  wrap×{wx:.2f}\nPSNR={p:.1f} dB",
                 fontsize=9, color="#e6edf3")
    ax.set_xlabel("x [um]", fontsize=8, color="#8b949e")
    ax.set_ylabel("h [nm]", fontsize=8, color="#8b949e")
    ax.legend(fontsize=7, labelcolor="#e6edf3", framealpha=0.3)
    _style(ax)

plt.tight_layout()
plt.savefig(OUT / "sigma_stress.png", dpi=150, bbox_inches="tight",
            facecolor="#0d1117")
plt.close()

lines = ["sigma_stress_test.py — find where current dx breaks",
         "=" * 60,
         f"  A           = {A_NM} nm (fixed)",
         f"  mem_pitch   = {dx*1e6:.1f} um (fixed)",
         f"  phase-wrap  = {PHASE_WRAP_NM:.2f} nm per pixel",
         f"  algorithm   = squared (h>=0) + Adam {N_ITER} iters, cosine decay",
         "",
         f"  {'sigma[um]':>10}  {'wrap_x':>7}  {'PSNR':>8}  {'RMSE[nm]':>10}"]
for r in results:
    lines.append(f"  {r['sigma_um']:>10.0f}  "
                 f"{r['wrap_multiple']:>7.2f}  "
                 f"{r['psnr']:>8.2f}  {r['rmse_nm']:>10.3f}")
lines += ["",
          "Reading the table",
          "  * wrap_x < 1  : algorithm should reach near-numerical-floor",
          "  * wrap_x ~ 1  : transition region, expect partial degradation",
          "  * wrap_x >> 1 : multi-cycle pixel wrap; reconstruction breaks",
          "                  even with positivity constraint",
          "",
          "  The first sigma at which PSNR collapses pinpoints where the",
          "  current mem_pitch=10um setup starts failing. To recover that",
          "  regime, dx must be reduced (NVIDIA + finer membrane pitch)."]
txt = "\n".join(lines)
(OUT / "sigma_stress.txt").write_text(txt, encoding="utf-8")
print("\n" + txt)

(OUT / "sigma_stress.json").write_text(json.dumps(dict(
    config=dict(A_nm=A_NM, mem_pitch_um=dx*1e6, distance_mm=DIST*1e3,
                mem=MEM, cmos=CMOS, grid=GRID,
                lr=LR, n_iter=N_ITER, phase_wrap_pixel_nm=PHASE_WRAP_NM),
    results=results,
), indent=2))
print(f"\nSaved -> {OUT/'sigma_stress.png'}")
