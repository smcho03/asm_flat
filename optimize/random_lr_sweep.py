"""
random_lr_sweep.py
------------------
LR sweep for RANDOM SURFACE reconstruction.
Test case: amp=50nm, sigma=150um, d=5mm (sub-wavelength regime)
Adam, MSE, no reg, zeros init, 2500 iter
3 seeds per LR -> report mean +/- std
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
DISTANCE = 5e-3
AMP      = 50e-9
SIGMA    = 150e-6
N_ITER   = 2500
N_SEEDS  = 3

LRS = [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 2e-8, 3e-8, 5e-8, 1e-7]

sensor = HolographicSensor(
    wavelength=wavelength, mem_res=MEM_RES, mem_pitch=mem_pitch,
    cmos_res=CMOS_RES, grid_res=GRID_RES, distance=DISTANCE, device=device,
).to(device)

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

def run_one(lr, seed):
    h_true = make_h_random(N=MEM_RES, dx=mem_pitch, device=device,
                           seed=seed, amplitude=AMP, sigma_filter=SIGMA)
    with torch.no_grad():
        I_target = sensor(h_true)

    h_pred = torch.zeros(MEM_RES, MEM_RES, dtype=torch.float32,
                         device=device, requires_grad=True)
    opt = torch.optim.Adam([h_pred], lr=lr)

    for _ in range(N_ITER):
        opt.zero_grad()
        loss = torch.mean((sensor(h_pred) - I_target)**2)
        loss.backward()
        opt.step()

    return (psnr_db(h_pred.detach(), h_true),
            rmse_nm(h_pred.detach(), h_true),
            ncc(h_pred.detach(), h_true))

print(f"\nRandom surface LR sweep")
print(f"amp={AMP*1e9:.0f}nm  sigma={SIGMA*1e6:.0f}um  d={DISTANCE*1e3:.0f}mm")
print(f"{N_ITER} iter  {N_SEEDS} seeds per LR")
print(f"\n{'LR':>8}  {'PSNR mean':>10}  {'PSNR std':>9}  {'RMSE mean':>10}  {'RMSE std':>9}  {'NCC mean':>9}")
print("-" * 65)

results = []
for lr in LRS:
    t0 = time.time()
    psnrs, rmses, nccs = [], [], []
    for seed in range(N_SEEDS):
        p, r, n = run_one(lr, seed)
        psnrs.append(p); rmses.append(r); nccs.append(n)
    dt = time.time() - t0
    res = dict(lr=lr,
               psnr_mean=np.mean(psnrs), psnr_std=np.std(psnrs),
               rmse_mean=np.mean(rmses), rmse_std=np.std(rmses),
               ncc_mean=np.mean(nccs))
    results.append(res)
    diverged = not all(np.isfinite(p) for p in psnrs)
    flag = "  DIVERGED" if diverged else ""
    print(f"{lr:>8.0e}  {res['psnr_mean']:>10.1f}  {res['psnr_std']:>9.1f}"
          f"  {res['rmse_mean']:>10.1f}  {res['rmse_std']:>9.1f}"
          f"  {res['ncc_mean']:>9.3f}  ({dt:.0f}s){flag}")

# best
valid = [r for r in results if np.isfinite(r['psnr_mean'])]
best = max(valid, key=lambda r: r['psnr_mean'])
print(f"\nBest LR: {best['lr']:.0e}  PSNR={best['psnr_mean']:.1f}+/-{best['psnr_std']:.1f}dB"
      f"  RMSE={best['rmse_mean']:.1f}+/-{best['rmse_std']:.1f}nm"
      f"  NCC={best['ncc_mean']:.3f}")

# -----------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------
plt.rcParams.update(STYLE)

valid_lrs   = [r['lr']        for r in valid]
valid_psnrs = [r['psnr_mean'] for r in valid]
valid_rmses = [r['rmse_mean'] for r in valid]
psnr_stds   = [r['psnr_std']  for r in valid]
rmse_stds   = [r['rmse_std']  for r in valid]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    f"LR Sweep - Random Surface  amp={AMP*1e9:.0f}nm  sigma={SIGMA*1e6:.0f}um  d={DISTANCE*1e3:.0f}mm\n"
    f"Adam, MSE, no reg, zeros, {N_ITER} iter, {N_SEEDS} seeds",
    fontsize=10, color="#e6edf3"
)
fig.patch.set_facecolor("#0d1117")

def _style(ax):
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e", labelsize=7)
    ax.grid(color="#30363d", lw=0.4, ls=":")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

for ax, vals, stds, ylabel, color in [
    (axes[0], valid_psnrs, psnr_stds, "PSNR [dB]", "#58a6ff"),
    (axes[1], valid_rmses, rmse_stds, "RMSE [nm]", "#f78166"),
]:
    ax.semilogx(valid_lrs, vals, "o-", color=color, lw=1.5, markersize=5, zorder=3)
    ax.fill_between(valid_lrs,
                    [v - s for v, s in zip(vals, stds)],
                    [v + s for v, s in zip(vals, stds)],
                    color=color, alpha=0.2)
    ax.axvline(best['lr'], color="#3fb950", lw=1.0, ls="--", alpha=0.8)
    ax.text(best['lr'] * 1.2,
            ax.get_ylim()[0] if ylabel.startswith("PSNR") else ax.get_ylim()[1],
            f"best\n{best['lr']:.0e}",
            fontsize=7, color="#3fb950",
            va="bottom" if ylabel.startswith("PSNR") else "top")
    ax.set_xlabel("Learning Rate", fontsize=8, color="#8b949e")
    ax.set_ylabel(ylabel, fontsize=8, color="#8b949e")
    ax.set_title(f"{ylabel} vs LR (random surface)", fontsize=9, color="#e6edf3")
    _style(ax)

plt.tight_layout()
out_path = OUT / "random_lr_sweep.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"Saved -> {out_path}")
