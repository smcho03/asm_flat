"""
random_reg_sweep.py
-------------------
Regularization sweep for random surface reconstruction.
Adam lr=5e-10, MSE, zeros, 2500 iter
amp=50nm, sigma=150um, d=5mm, 3 seeds
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
LR       = 5e-10
N_ITER   = 2500
N_SEEDS  = 3

def tv_loss(h):
    return torch.mean(torch.abs(h[1:, :] - h[:-1, :])) + \
           torch.mean(torch.abs(h[:, 1:] - h[:, :-1]))

def laplacian_loss(h):
    lap = (h[:-2, 1:-1] + h[2:, 1:-1] + h[1:-1, :-2] + h[1:-1, 2:] - 4*h[1:-1, 1:-1])
    return torch.mean(lap**2)

CONFIGS = [
    dict(name="none",           lam_tv=0,    lam_lap=0),
    dict(name="TV 1e-3",        lam_tv=1e-3, lam_lap=0),
    dict(name="TV 1e-2",        lam_tv=1e-2, lam_lap=0),
    dict(name="TV 1e-1",        lam_tv=1e-1, lam_lap=0),
    dict(name="Laplacian 1e10", lam_tv=0,    lam_lap=1e10),
    dict(name="Laplacian 1e12", lam_tv=0,    lam_lap=1e12),
    dict(name="TV+Lap",         lam_tv=1e-2, lam_lap=1e10),
]

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

def run_one(cfg, seed):
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
        if cfg['lam_tv']  > 0: loss = loss + cfg['lam_tv']  * tv_loss(h_pred)
        if cfg['lam_lap'] > 0: loss = loss + cfg['lam_lap'] * laplacian_loss(h_pred)
        loss.backward()
        opt.step()
    return psnr_db(h_pred.detach(), h_true), rmse_nm(h_pred.detach(), h_true), ncc(h_pred.detach(), h_true)

print(f"\nRandom surface regularization sweep")
print(f"amp={AMP*1e9:.0f}nm  sigma={SIGMA*1e6:.0f}um  d={DISTANCE*1e3:.0f}mm  lr={LR:.0e}  {N_ITER} iter  {N_SEEDS} seeds")
print(f"\n{'Config':>18}  {'PSNR mean':>10}  {'std':>6}  {'RMSE mean':>10}  {'std':>6}  {'NCC':>7}")
print("-" * 65)

results = []
for cfg in CONFIGS:
    t0 = time.time()
    psnrs, rmses, nccs = [], [], []
    for seed in range(N_SEEDS):
        p, r, n = run_one(cfg, seed)
        psnrs.append(p); rmses.append(r); nccs.append(n)
    dt = time.time() - t0
    res = dict(name=cfg['name'],
               psnr_mean=np.mean(psnrs), psnr_std=np.std(psnrs),
               rmse_mean=np.mean(rmses), rmse_std=np.std(rmses),
               ncc_mean=np.mean(nccs))
    results.append(res)
    print(f"{cfg['name']:>18}  {res['psnr_mean']:>10.1f}  {res['psnr_std']:>6.1f}"
          f"  {res['rmse_mean']:>10.1f}  {res['rmse_std']:>6.1f}"
          f"  {res['ncc_mean']:>7.3f}  ({dt:.0f}s)")

best = max(results, key=lambda r: r['psnr_mean'])
print(f"\nBest: {best['name']}  PSNR={best['psnr_mean']:.1f}+/-{best['psnr_std']:.1f}dB"
      f"  RMSE={best['rmse_mean']:.1f}+/-{best['rmse_std']:.1f}nm")

# -----------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------
plt.rcParams.update(STYLE)

names  = [r['name']      for r in results]
psnr_m = [r['psnr_mean'] for r in results]
psnr_s = [r['psnr_std']  for r in results]
rmse_m = [r['rmse_mean'] for r in results]
rmse_s = [r['rmse_std']  for r in results]
best_i = results.index(best)

x = np.arange(len(results))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    f"Regularization Sweep - Random Surface  lr={LR:.0e}  amp={AMP*1e9:.0f}nm  sigma={SIGMA*1e6:.0f}um\n"
    f"Adam, MSE, zeros, {N_ITER} iter, {N_SEEDS} seeds (mean +/- std)",
    fontsize=10, color="#e6edf3"
)
fig.patch.set_facecolor("#0d1117")

def _style(ax):
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e", labelsize=7)
    ax.grid(color="#30363d", lw=0.4, ls=":", axis="y")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

for ax, means, stds, ylabel, base_color in [
    (axes[0], psnr_m, psnr_s, "PSNR [dB]", "#58a6ff"),
    (axes[1], rmse_m, rmse_s, "RMSE [nm]", "#f78166"),
]:
    colors = [base_color if i == best_i else "#30363d" for i in range(len(results))]
    bars = ax.bar(x, means, color=colors, edgecolor="#8b949e", linewidth=0.5)
    ax.errorbar(x, means, yerr=stds, fmt="none", color="#e6edf3",
                capsize=4, capthick=1.2, elinewidth=1.0)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (0.2 if "PSNR" in ylabel else 0.3),
                f"{val:.1f}", ha="center", va="bottom",
                fontsize=6.5, color="#e6edf3")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=6.5, color="#8b949e", rotation=15, ha="right")
    ax.set_ylabel(ylabel, fontsize=8, color="#8b949e")
    ax.set_title(f"{ylabel} vs Regularization (random surface)", fontsize=9, color="#e6edf3")
    _style(ax)

plt.tight_layout()
out_path = OUT / "random_reg_sweep.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"Saved -> {out_path}")
