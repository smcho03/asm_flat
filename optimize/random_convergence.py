"""
random_convergence.py
---------------------
Convergence analysis for random surface reconstruction.
Optimal config so far: Adam lr=5e-10, MSE, no reg, zeros
Test case: amp=50nm, sigma=150um, d=5mm, seed=0
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
from sensor_model import HolographicSensor, make_h_random_pressed
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
N_ITER   = 8000
SEED     = 0

sensor = HolographicSensor(
    wavelength=wavelength, mem_res=MEM_RES, mem_pitch=mem_pitch,
    cmos_res=CMOS_RES, grid_res=GRID_RES, distance=DISTANCE, device=device,
).to(device)

h_true = make_h_random_pressed(N=MEM_RES, dx=mem_pitch, device=device,
                               seed=SEED, amplitude=AMP, sigma_filter=SIGMA)
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

losses, rmses, psnrs = [], [], []

LOG_EVERY = 500
print(f"\nRunning {N_ITER} iterations (random surface, lr={LR:.0e})...")
print(f"{'iter':>6}  {'loss':>12}  {'RMSE [nm]':>10}  {'PSNR [dB]':>10}")
print("-" * 46)

for i in range(N_ITER):
    opt.zero_grad()
    loss = torch.mean((sensor(h_pred) - I_target)**2)
    loss.backward()
    opt.step()

    l = float(loss.detach())
    r = rmse_nm(h_pred.detach(), h_true)
    p = psnr_db(h_pred.detach(), h_true)
    losses.append(l); rmses.append(r); psnrs.append(p)

    if (i + 1) % LOG_EVERY == 0 or i == 0:
        print(f"{i+1:>6}  {l:>12.6e}  {r:>10.2f}  {p:>10.2f}")

print(f"\nFinal: PSNR={psnrs[-1]:.2f} dB  RMSE={rmses[-1]:.2f} nm")

# -----------------------------------------------------------------------
# Convergence detection
# -----------------------------------------------------------------------
WINDOW    = 200
THRESHOLD = 0.02  # nm

rmses_arr = np.array(rmses)
converge_iter = None
for i in range(WINDOW, N_ITER):
    if rmses_arr[i - WINDOW] - rmses_arr[i] < THRESHOLD:
        converge_iter = i - WINDOW
        break

if converge_iter is not None:
    print(f"Convergence at iter ~{converge_iter}")
    print(f"  RMSE at convergence: {rmses_arr[converge_iter]:.2f} nm")
    print(f"  Further gain: {rmses_arr[converge_iter] - rmses_arr[-1]:.2f} nm")
else:
    converge_iter = N_ITER - 1
    print(f"No clear convergence detected within {N_ITER} iter")
    print(f"  Final RMSE: {rmses_arr[-1]:.2f} nm (still improving)")

# -----------------------------------------------------------------------
# Plot 1: convergence curves
# -----------------------------------------------------------------------
plt.rcParams.update(STYLE)
iters = np.arange(1, N_ITER + 1)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle(
    f"Convergence - Random Surface  lr={LR:.0e}  amp={AMP*1e9:.0f}nm  sigma={SIGMA*1e6:.0f}um  d={DISTANCE*1e3:.0f}mm",
    fontsize=10, color="#e6edf3"
)
fig.patch.set_facecolor("#0d1117")

def _style(ax):
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e", labelsize=7)
    ax.grid(color="#30363d", lw=0.4, ls=":")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

for ax, vals, ylabel, color, log in [
    (axes[0], losses, "Intensity MSE (loss)", "#58a6ff", True),
    (axes[1], rmses,  "RMSE [nm]",            "#f78166", False),
    (axes[2], psnrs,  "PSNR [dB]",            "#3fb950", False),
]:
    if log:
        ax.semilogy(iters, vals, color=color, lw=1.2)
    else:
        ax.plot(iters, vals, color=color, lw=1.2)
        ax.axvspan(converge_iter, N_ITER, color="#30363d", alpha=0.5,
                   label=f"plateau (>{converge_iter})")
        ax.axvline(converge_iter, color="#e3b341", lw=1.2, ls="--")
        ax.text(converge_iter + N_ITER*0.01,
                max(vals)*0.97 if "RMSE" in ylabel else min(vals)*1.02,
                f"~{converge_iter} iter",
                fontsize=7, color="#e3b341",
                va="top" if "RMSE" in ylabel else "bottom")
        ax.axhline(vals[-1], color=color, lw=0.8, ls=":", alpha=0.6)
        ax.text(N_ITER*0.02, vals[-1]*(1.04 if "RMSE" in ylabel else 0.997),
                f"final: {vals[-1]:.1f}", fontsize=7, color=color)
        ax.legend(fontsize=7, labelcolor="#e6edf3", framealpha=0.3,
                  loc="upper right" if "RMSE" in ylabel else "lower right")

    ax.set_xlabel("Iteration", fontsize=8, color="#8b949e")
    ax.set_ylabel(ylabel, fontsize=8, color="#8b949e")
    ax.set_title(ylabel + " vs Iteration", fontsize=9, color="#e6edf3")
    _style(ax)

plt.tight_layout()
p1 = OUT / "random_convergence.png"
plt.savefig(p1, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"Saved -> {p1}")

# -----------------------------------------------------------------------
# Plot 2: h_true vs h_pred vs residual
# -----------------------------------------------------------------------
h_np  = h_true.cpu().numpy() * 1e9
hp_np = h_pred.detach().cpu().numpy() * 1e9
diff  = hp_np - h_np
dx_um = mem_pitch * 1e6
extent = [0, MEM_RES*dx_um, 0, MEM_RES*dx_um]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle(
    f"Reconstruction - Random Surface  PSNR={psnrs[-1]:.1f}dB  RMSE={rmses[-1]:.1f}nm",
    fontsize=10, color="#e6edf3"
)
fig.patch.set_facecolor("#0d1117")

vmax = np.abs(h_np).max()
dmax = np.abs(diff).max()
for ax, arr, title, cmap, vmin, vmax_val in zip(
    axes,
    [h_np, hp_np, diff],
    ["h_true [nm]", "h_pred [nm]", "Residual (pred-true) [nm]"],
    ["viridis", "viridis", "RdBu_r"],
    [0,    0,    -dmax],
    [vmax, vmax,  dmax],
):
    im = ax.imshow(arr, extent=extent, origin="lower", cmap=cmap,
                   vmin=vmin, vmax=vmax_val, aspect="equal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="nm")
    ax.set_title(title, fontsize=9, color="#e6edf3")
    ax.set_xlabel("x [um]", fontsize=7, color="#8b949e")
    ax.set_ylabel("y [um]", fontsize=7, color="#8b949e")
    ax.tick_params(colors="#8b949e", labelsize=6)
    ax.set_facecolor("#0d1117")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

plt.tight_layout()
p2 = OUT / "random_reconstruction.png"
plt.savefig(p2, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"Saved -> {p2}")
