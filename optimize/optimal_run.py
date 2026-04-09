"""
optimal_run.py
--------------
Run reconstruction under the best hyperparameter configuration found so far:
  Adam, lr=2e-8, MSE, no reg, zeros init, 5000 iter
  single_bump  amp=200nm  sigma=150um  d=5mm

Tracks per-iteration: intensity loss, RMSE [nm], PSNR [dB]
Saves:
  output/optimal_loss_curve.png   -- loss + RMSE + PSNR vs iteration
  output/optimal_reconstruction.png -- h_true vs h_pred side-by-side
"""

import sys, functools
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

# -----------------------------------------------------------------------
# Optimal hyperparameters
# -----------------------------------------------------------------------
MEM_RES  = 128
CMOS_RES = 256
GRID_RES = 384
DISTANCE = 5e-3
AMP      = 200e-9
SIGMA    = 150e-6
LR       = 2e-8
N_ITER   = 5000

print(f"\nOptimal configuration:")
print(f"  Optimizer : Adam  lr={LR}")
print(f"  Loss      : MSE")
print(f"  Init      : zeros")
print(f"  Reg       : none")
print(f"  Iterations: {N_ITER}")
print(f"  Test case : single_bump  amp={AMP*1e9:.0f}nm  sigma={SIGMA*1e6:.0f}um  d={DISTANCE*1e3:.0f}mm")

# -----------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------
sensor = HolographicSensor(
    wavelength=wavelength, mem_res=MEM_RES, mem_pitch=mem_pitch,
    cmos_res=CMOS_RES, grid_res=GRID_RES, distance=DISTANCE, device=device,
).to(device)

h_true = gaussian_bump(N=MEM_RES, dx=mem_pitch, amplitude=AMP,
                       sigma=SIGMA, device=device)
with torch.no_grad():
    I_target = sensor(h_true)

# -----------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------
def rmse_nm(a, b):
    return float(torch.sqrt(torch.mean((a - b)**2))) * 1e9

def psnr_db(a, b):
    mse  = float(torch.mean((a - b)**2))
    peak = float(b.abs().max())
    if mse < 1e-30: return 100.0
    return 10.0 * np.log10(peak**2 / mse)

# -----------------------------------------------------------------------
# Reconstruction
# -----------------------------------------------------------------------
h_pred = torch.zeros(MEM_RES, MEM_RES, dtype=torch.float32,
                     device=device, requires_grad=True)
opt = torch.optim.Adam([h_pred], lr=LR)

losses, rmses, psnrs = [], [], []

LOG_EVERY = 100
print(f"\n{'iter':>6}  {'loss':>12}  {'RMSE [nm]':>10}  {'PSNR [dB]':>10}")
print("-" * 46)

for i in range(N_ITER):
    opt.zero_grad()
    I_pred = sensor(h_pred)
    loss = torch.mean((I_pred - I_target)**2)
    loss.backward()
    opt.step()

    l = float(loss.detach())
    r = rmse_nm(h_pred.detach(), h_true)
    p = psnr_db(h_pred.detach(), h_true)

    losses.append(l)
    rmses.append(r)
    psnrs.append(p)

    if (i + 1) % LOG_EVERY == 0 or i == 0:
        print(f"{i+1:>6}  {l:>12.6e}  {r:>10.2f}  {p:>10.2f}")

print(f"\nFinal:  PSNR={psnrs[-1]:.2f} dB   RMSE={rmses[-1]:.2f} nm")

# -----------------------------------------------------------------------
# Plot 1: Loss curves
# -----------------------------------------------------------------------
plt.rcParams.update(STYLE)

iters = np.arange(1, N_ITER + 1)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle(
    f"Optimal Run — Adam lr={LR:.0e}, MSE, no reg, zeros, {N_ITER} iter\n"
    f"single_bump amp={AMP*1e9:.0f}nm  sigma={SIGMA*1e6:.0f}um  d={DISTANCE*1e3:.0f}mm",
    fontsize=10, color="#e6edf3"
)
fig.patch.set_facecolor("#0d1117")

def _style(ax):
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e", labelsize=7)
    ax.grid(color="#30363d", lw=0.4, ls=":")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

# Intensity loss
ax = axes[0]
ax.semilogy(iters, losses, color="#58a6ff", lw=1.2)
ax.set_xlabel("Iteration", fontsize=8, color="#8b949e")
ax.set_ylabel("Intensity MSE", fontsize=8, color="#8b949e")
ax.set_title("Loss (Intensity MSE)", fontsize=9, color="#e6edf3")
_style(ax)

# RMSE
ax = axes[1]
ax.plot(iters, rmses, color="#f78166", lw=1.2)
ax.axhline(rmses[-1], color="#f78166", lw=0.7, ls="--", alpha=0.6)
ax.text(N_ITER * 0.02, rmses[-1] * 1.05,
        f"Final: {rmses[-1]:.1f} nm", fontsize=7, color="#f78166")
ax.set_xlabel("Iteration", fontsize=8, color="#8b949e")
ax.set_ylabel("RMSE [nm]", fontsize=8, color="#8b949e")
ax.set_title("Height RMSE", fontsize=9, color="#e6edf3")
_style(ax)

# PSNR
ax = axes[2]
ax.plot(iters, psnrs, color="#3fb950", lw=1.2)
ax.axhline(psnrs[-1], color="#3fb950", lw=0.7, ls="--", alpha=0.6)
ax.text(N_ITER * 0.02, psnrs[-1] * 0.98,
        f"Final: {psnrs[-1]:.1f} dB", fontsize=7, color="#3fb950")
ax.set_xlabel("Iteration", fontsize=8, color="#8b949e")
ax.set_ylabel("PSNR [dB]", fontsize=8, color="#8b949e")
ax.set_title("Height PSNR", fontsize=9, color="#e6edf3")
_style(ax)

plt.tight_layout()
p1 = OUT / "optimal_loss_curve.png"
plt.savefig(p1, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"Saved -> {p1}")

# -----------------------------------------------------------------------
# Plot 2: h_true vs h_pred
# -----------------------------------------------------------------------
h_np   = h_true.cpu().numpy() * 1e9      # nm
hp_np  = h_pred.detach().cpu().numpy() * 1e9
diff   = hp_np - h_np

dx_um  = mem_pitch * 1e6
extent = [0, MEM_RES * dx_um, 0, MEM_RES * dx_um]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle(
    f"Reconstruction Result — PSNR={psnrs[-1]:.1f} dB   RMSE={rmses[-1]:.1f} nm",
    fontsize=10, color="#e6edf3"
)
fig.patch.set_facecolor("#0d1117")

vmax = np.abs(h_np).max()

for ax, arr, title, cmap in zip(
    axes,
    [h_np, hp_np, diff],
    ["h_true [nm]", "h_pred [nm]", "Residual (pred - true) [nm]"],
    ["viridis", "viridis", "RdBu_r"],
):
    if "Residual" in title:
        v = np.abs(diff).max()
        im = ax.imshow(arr, extent=extent, origin="lower", cmap=cmap,
                       vmin=-v, vmax=v, aspect="equal")
    else:
        im = ax.imshow(arr, extent=extent, origin="lower", cmap=cmap,
                       vmin=0, vmax=vmax, aspect="equal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=9, color="#e6edf3")
    ax.set_xlabel("x [um]", fontsize=7, color="#8b949e")
    ax.set_ylabel("y [um]", fontsize=7, color="#8b949e")
    ax.tick_params(colors="#8b949e", labelsize=6)
    ax.set_facecolor("#0d1117")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

plt.tight_layout()
p2 = OUT / "optimal_reconstruction.png"
plt.savefig(p2, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"Saved -> {p2}")
