"""
convergence_plot.py
-------------------
Shows iteration count convergence for the optimal configuration.
Detects the plateau point and shades the "unnecessary" region.

Optimal config: Adam lr=2e-8, MSE, no reg, zeros, 5000 iter
Test case: single_bump amp=200nm sigma=150um d=5mm
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
# Setup
# -----------------------------------------------------------------------
MEM_RES  = 128
CMOS_RES = 256
GRID_RES = 384
DISTANCE = 5e-3
AMP      = 200e-9
SIGMA    = 150e-6
LR       = 2e-8
N_ITER   = 5000

sensor = HolographicSensor(
    wavelength=wavelength, mem_res=MEM_RES, mem_pitch=mem_pitch,
    cmos_res=CMOS_RES, grid_res=GRID_RES, distance=DISTANCE, device=device,
).to(device)

h_true = gaussian_bump(N=MEM_RES, dx=mem_pitch, amplitude=AMP,
                       sigma=SIGMA, device=device)
with torch.no_grad():
    I_target = sensor(h_true)

def rmse_nm(a, b):
    return float(torch.sqrt(torch.mean((a - b)**2))) * 1e9

def psnr_db(a, b):
    mse  = float(torch.mean((a - b)**2))
    peak = float(b.abs().max())
    if mse < 1e-30: return 100.0
    return 10.0 * np.log10(peak**2 / mse)

# -----------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------
h_pred = torch.zeros(MEM_RES, MEM_RES, dtype=torch.float32,
                     device=device, requires_grad=True)
opt = torch.optim.Adam([h_pred], lr=LR)

rmses, psnrs = [], []

print(f"Running {N_ITER} iterations...")
for i in range(N_ITER):
    opt.zero_grad()
    I_pred = sensor(h_pred)
    loss = torch.mean((I_pred - I_target)**2)
    loss.backward()
    opt.step()
    rmses.append(rmse_nm(h_pred.detach(), h_true))
    psnrs.append(psnr_db(h_pred.detach(), h_true))
    if (i + 1) % 500 == 0:
        print(f"  iter {i+1:5d}  RMSE={rmses[-1]:.2f} nm  PSNR={psnrs[-1]:.2f} dB")

rmses = np.array(rmses)
psnrs = np.array(psnrs)
iters = np.arange(1, N_ITER + 1)

# -----------------------------------------------------------------------
# Convergence detection
# Criterion: improvement over a window < threshold
# -----------------------------------------------------------------------
WINDOW    = 200   # look back over 200 iters
THRESHOLD = 0.05  # nm improvement threshold

converge_iter = N_ITER  # default: never converged
for i in range(WINDOW, N_ITER):
    improvement = rmses[i - WINDOW] - rmses[i]   # positive = still improving
    if improvement < THRESHOLD:
        converge_iter = i - WINDOW
        break

print(f"\nConvergence detected at iter ~{converge_iter}")
print(f"  RMSE at convergence : {rmses[converge_iter]:.2f} nm")
print(f"  RMSE at {N_ITER} iter   : {rmses[-1]:.2f} nm")
print(f"  Further gain        : {rmses[converge_iter] - rmses[-1]:.2f} nm")

# -----------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------
plt.rcParams.update(STYLE)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    f"Convergence Analysis — Adam lr={LR:.0e}, MSE, no reg, zeros\n"
    f"single_bump amp={AMP*1e9:.0f}nm  sigma={SIGMA*1e6:.0f}um  d={DISTANCE*1e3:.0f}mm",
    fontsize=10, color="#e6edf3"
)
fig.patch.set_facecolor("#0d1117")

def _style(ax):
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e", labelsize=7)
    ax.grid(color="#30363d", lw=0.4, ls=":")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

for ax, vals, ylabel, color, final_val in [
    (axes[0], rmses, "RMSE [nm]",  "#f78166", rmses[-1]),
    (axes[1], psnrs, "PSNR [dB]",  "#3fb950", psnrs[-1]),
]:
    # Main curve
    ax.plot(iters, vals, color=color, lw=1.4, zorder=3)

    # Shade plateau region
    ax.axvspan(converge_iter, N_ITER, color="#30363d", alpha=0.5,
               label=f"plateau (>{converge_iter} iter)")

    # Convergence marker
    ax.axvline(converge_iter, color="#e3b341", lw=1.2, ls="--", zorder=4)
    ax.text(converge_iter + N_ITER * 0.01,
            vals.max() * 0.97 if ylabel.startswith("RMSE") else vals.min() * 1.02,
            f"converged\n~{converge_iter} iter",
            fontsize=7, color="#e3b341", va="top" if ylabel.startswith("RMSE") else "bottom")

    # Final value line
    ax.axhline(final_val, color=color, lw=0.8, ls=":", alpha=0.6)
    ax.text(N_ITER * 0.02,
            final_val * (1.04 if ylabel.startswith("RMSE") else 0.997),
            f"final: {final_val:.1f}", fontsize=7, color=color)

    ax.set_xlabel("Iteration", fontsize=8, color="#8b949e")
    ax.set_ylabel(ylabel, fontsize=8, color="#8b949e")
    ax.set_title(ylabel + " vs Iteration", fontsize=9, color="#e6edf3")
    ax.legend(fontsize=7, labelcolor="#e6edf3", framealpha=0.3,
              loc="upper right" if ylabel.startswith("RMSE") else "lower right")
    _style(ax)

plt.tight_layout()
out_path = OUT / "convergence_plot.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"\nSaved -> {out_path}")
