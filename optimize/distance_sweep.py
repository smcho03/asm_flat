"""
distance_sweep.py
-----------------
Sweep propagation distance with fine granularity.
Optimal config: Adam lr=2e-8, MSE, no reg, zeros, 2000 iter
Test case: single_bump amp=200nm sigma=150um
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
from sensor_model import HolographicSensor, gaussian_bump
from sensor_params import wavelength, mem_pitch
from sensor_utils import STYLE

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

MEM_RES  = 128
CMOS_RES = 256
GRID_RES = 384
AMP      = 200e-9
SIGMA    = 150e-6
LR       = 2e-8
N_ITER   = 2000

DISTANCES = [1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 10e-3, 12e-3, 15e-3, 20e-3]

def rmse_nm(a, b):
    return float(torch.sqrt(torch.mean((a - b)**2))) * 1e9

def psnr_db(a, b):
    mse  = float(torch.mean((a - b)**2))
    peak = float(b.abs().max())
    if mse < 1e-30: return 100.0
    return 10.0 * np.log10(peak**2 / mse)

def run(distance):
    sensor = HolographicSensor(
        wavelength=wavelength, mem_res=MEM_RES, mem_pitch=mem_pitch,
        cmos_res=CMOS_RES, grid_res=GRID_RES, distance=distance, device=device,
    ).to(device)

    h_true = gaussian_bump(N=MEM_RES, dx=mem_pitch, amplitude=AMP,
                           sigma=SIGMA, device=device)
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

    return psnr_db(h_pred.detach(), h_true), rmse_nm(h_pred.detach(), h_true)

print(f"\nDistance sweep - Adam lr={LR:.0e}, MSE, no reg, zeros, {N_ITER} iter")
print(f"single_bump amp={AMP*1e9:.0f}nm  sigma={SIGMA*1e6:.0f}um")
print(f"\n{'Distance':>10}  {'PSNR [dB]':>10}  {'RMSE [nm]':>10}")
print("-" * 35)

results = []
for d in DISTANCES:
    t0 = time.time()
    p, r = run(d)
    dt = time.time() - t0
    print(f"{d*1e3:>8.1f}mm  {p:>10.1f}  {r:>10.1f}  ({dt:.0f}s)")
    results.append((d, p, r))

# -----------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------
plt.rcParams.update(STYLE)

dists_mm = [r[0]*1e3 for r in results]
psnrs    = [r[1] for r in results]
rmses    = [r[2] for r in results]

best_idx = psnrs.index(max(psnrs))
best_d   = dists_mm[best_idx]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    f"Distance Sweep — Adam lr={LR:.0e}, MSE, no reg, zeros, {N_ITER} iter\n"
    f"single_bump  amp={AMP*1e9:.0f}nm  sigma={SIGMA*1e6:.0f}um",
    fontsize=10, color="#e6edf3"
)
fig.patch.set_facecolor("#0d1117")

def _style(ax):
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e", labelsize=7)
    ax.grid(color="#30363d", lw=0.4, ls=":")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

for ax, vals, ylabel, color in [
    (axes[0], psnrs, "PSNR [dB]", "#58a6ff"),
    (axes[1], rmses, "RMSE [nm]", "#f78166"),
]:
    ax.plot(dists_mm, vals, "o-", color=color, lw=1.5, markersize=5, zorder=3)
    ax.axvline(best_d, color="#3fb950", lw=1.0, ls="--", alpha=0.8)
    ax.text(best_d + 0.2, ax.get_ylim()[0] if ylabel.startswith("PSNR") else ax.get_ylim()[1],
            f"best: {best_d:.0f}mm", fontsize=7, color="#3fb950",
            va="bottom" if ylabel.startswith("PSNR") else "top")

    # Annotate each point
    for x, y in zip(dists_mm, vals):
        ax.text(x, y + (0.3 if ylabel.startswith("PSNR") else -0.5),
                f"{y:.1f}", ha="center", va="bottom" if ylabel.startswith("PSNR") else "top",
                fontsize=6, color="#8b949e")

    ax.set_xlabel("Distance [mm]", fontsize=8, color="#8b949e")
    ax.set_ylabel(ylabel, fontsize=8, color="#8b949e")
    ax.set_title(f"{ylabel} vs Distance", fontsize=9, color="#e6edf3")
    _style(ax)

plt.tight_layout()
out_path = OUT / "distance_sweep.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"\nBest distance: {best_d:.0f}mm  PSNR={psnrs[best_idx]:.1f}dB  RMSE={rmses[best_idx]:.1f}nm")
print(f"Saved -> {out_path}")
