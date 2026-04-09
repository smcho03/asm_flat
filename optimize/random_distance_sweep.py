"""
random_distance_sweep.py
------------------------
Distance sweep for random surface reconstruction.
Adam lr=5e-10, MSE, no reg, zeros, 2500 iter
amp=50nm, sigma=150um, 3 seeds
"""

import sys, time, functools
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

print = functools.partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).parent.parent))
from sensor_model import HolographicSensor, make_h_random
from sensor_params import wavelength, mem_pitch
from sensor_utils import STYLE

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

MEM_RES  = 128
CMOS_RES = 256
GRID_RES = 384
AMP      = 50e-9
SIGMA    = 150e-6
LR       = 5e-10
N_ITER   = 2500
N_SEEDS  = 3

DISTANCES = [2e-3, 4e-3, 6e-3, 8e-3, 12e-3]

def rmse_nm(a, b):
    return float(torch.sqrt(torch.mean((a - b)**2))) * 1e9

def psnr_db(a, b):
    mse  = float(torch.mean((a - b)**2))
    peak = float(b.abs().max())
    if mse < 1e-30: return 100.0
    return 10.0 * np.log10(peak**2 / mse)

def ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    denom = torch.sqrt(torch.sum(a**2) * torch.sum(b**2))
    return float((torch.sum(a * b) / denom.clamp(min=1e-30)).item())

def run_one(distance, seed):
    sensor = HolographicSensor(
        wavelength=wavelength, mem_res=MEM_RES, mem_pitch=mem_pitch,
        cmos_res=CMOS_RES, grid_res=GRID_RES, distance=distance, device=device,
    ).to(device)
    h_true = make_h_random(N=MEM_RES, dx=mem_pitch, device=device,
                           seed=seed, amplitude=AMP, sigma_filter=SIGMA)
    with torch.no_grad():
        I_target = sensor(h_true)
    h_pred = torch.zeros(MEM_RES, MEM_RES, dtype=torch.float32,
                         device=device, requires_grad=True)
    opt = torch.optim.Adam([h_pred], lr=LR)
    for _ in range(N_ITER):
        opt.zero_grad()
        loss = torch.mean((sensor(h_pred) - I_target)**2)
        loss.backward()
        opt.step()
    return psnr_db(h_pred.detach(), h_true), rmse_nm(h_pred.detach(), h_true), ncc(h_pred.detach(), h_true)

print(f"\nRandom surface distance sweep")
print(f"amp={AMP*1e9:.0f}nm  sigma={SIGMA*1e6:.0f}um  lr={LR:.0e}  {N_ITER} iter  {N_SEEDS} seeds")
print(f"\n{'Distance':>10}  {'PSNR mean':>10}  {'std':>6}  {'RMSE mean':>10}  {'std':>6}  {'NCC':>7}")
print("-" * 58)

results = []
for d in DISTANCES:
    t0 = time.time()
    psnrs, rmses, nccs = [], [], []
    for seed in range(N_SEEDS):
        p, r, n = run_one(d, seed)
        psnrs.append(p); rmses.append(r); nccs.append(n)
    dt = time.time() - t0
    res = dict(d=d,
               psnr_mean=np.mean(psnrs), psnr_std=np.std(psnrs),
               rmse_mean=np.mean(rmses), rmse_std=np.std(rmses),
               ncc_mean=np.mean(nccs))
    results.append(res)
    print(f"{d*1e3:>8.1f}mm  {res['psnr_mean']:>10.1f}  {res['psnr_std']:>6.1f}"
          f"  {res['rmse_mean']:>10.1f}  {res['rmse_std']:>6.1f}"
          f"  {res['ncc_mean']:>7.3f}  ({dt:.0f}s)")

best = max(results, key=lambda r: r['psnr_mean'])
print(f"\nBest distance: {best['d']*1e3:.0f}mm  PSNR={best['psnr_mean']:.1f}+/-{best['psnr_std']:.1f}dB"
      f"  RMSE={best['rmse_mean']:.1f}+/-{best['rmse_std']:.1f}nm")

# -----------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------
plt.rcParams.update(STYLE)

dists_mm = [r['d']*1e3 for r in results]
psnr_m   = [r['psnr_mean'] for r in results]
psnr_s   = [r['psnr_std']  for r in results]
rmse_m   = [r['rmse_mean'] for r in results]
rmse_s   = [r['rmse_std']  for r in results]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    f"Distance Sweep - Random Surface  lr={LR:.0e}  amp={AMP*1e9:.0f}nm  sigma={SIGMA*1e6:.0f}um\n"
    f"Adam, MSE, no reg, zeros, {N_ITER} iter, {N_SEEDS} seeds (mean +/- std)",
    fontsize=10, color="#e6edf3"
)
fig.patch.set_facecolor("#0d1117")

def _style(ax):
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e", labelsize=7)
    ax.grid(color="#30363d", lw=0.4, ls=":")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

for ax, means, stds, ylabel, color in [
    (axes[0], psnr_m, psnr_s, "PSNR [dB]", "#58a6ff"),
    (axes[1], rmse_m, rmse_s, "RMSE [nm]", "#f78166"),
]:
    ax.plot(dists_mm, means, "o-", color=color, lw=1.5, markersize=5, zorder=3)
    ax.fill_between(dists_mm,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    color=color, alpha=0.2)
    ax.axvline(best['d']*1e3, color="#3fb950", lw=1.0, ls="--", alpha=0.8)
    ax.text(best['d']*1e3 + 0.3, min(means) if "PSNR" in ylabel else max(means),
            f"best: {best['d']*1e3:.0f}mm", fontsize=7, color="#3fb950",
            va="bottom" if "PSNR" in ylabel else "top")
    ax.set_xlabel("Distance [mm]", fontsize=8, color="#8b949e")
    ax.set_ylabel(ylabel, fontsize=8, color="#8b949e")
    ax.set_title(f"{ylabel} vs Distance (random surface)", fontsize=9, color="#e6edf3")
    _style(ax)

plt.tight_layout()
out_path = OUT / "random_distance_sweep.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"Saved -> {out_path}")
