"""
random_iter_snapshots.py
------------------------
Visualize how h_pred evolves over iterations for random surface reconstruction.
Saves snapshots at key iterations and plots as a grid.
Adam lr=5e-10, MSE, no reg, zeros, 8000 iter
amp=50nm, sigma=150um, d=5mm, seed=0
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

# Iterations at which to save snapshots
SNAPSHOTS = [0, 50, 200, 500, 1000, 2000, 4000, 8000]

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
# Run and collect snapshots
# -----------------------------------------------------------------------
h_pred = torch.zeros(MEM_RES, MEM_RES, dtype=torch.float32,
                     device=device, requires_grad=True)
opt = torch.optim.Adam([h_pred], lr=LR)

snap_iters  = []
snap_arrays = []  # h_pred in nm
snap_rmse   = []
snap_psnr   = []

snap_set = set(SNAPSHOTS)

# iter=0 snapshot (before any update)
snap_iters.append(0)
snap_arrays.append(h_pred.detach().cpu().numpy() * 1e9)
snap_rmse.append(rmse_nm(h_pred.detach(), h_true))
snap_psnr.append(psnr_db(h_pred.detach(), h_true))

print(f"Running {N_ITER} iterations...")
for i in range(1, N_ITER + 1):
    opt.zero_grad()
    loss = torch.mean((sensor(h_pred) - I_target)**2)
    loss.backward()
    opt.step()

    if i in snap_set:
        snap_iters.append(i)
        snap_arrays.append(h_pred.detach().cpu().numpy() * 1e9)
        snap_rmse.append(rmse_nm(h_pred.detach(), h_true))
        snap_psnr.append(psnr_db(h_pred.detach(), h_true))
        print(f"  iter {i:5d}  RMSE={snap_rmse[-1]:.2f}nm  PSNR={snap_psnr[-1]:.2f}dB")

h_true_np = h_true.cpu().numpy() * 1e9

# -----------------------------------------------------------------------
# Plot: h_true + snapshots grid
# -----------------------------------------------------------------------
plt.rcParams.update(STYLE)

n_snaps = len(snap_iters)
n_cols  = n_snaps + 1  # +1 for h_true
fig, axes = plt.subplots(1, n_cols, figsize=(2.8 * n_cols, 3.5))
fig.suptitle(
    f"h_pred evolution over iterations  (random surface  amp={AMP*1e9:.0f}nm  sigma={SIGMA*1e6:.0f}um  lr={LR:.0e})",
    fontsize=10, color="#e6edf3", y=1.02
)
fig.patch.set_facecolor("#0d1117")

dx_um  = mem_pitch * 1e6
extent = [0, MEM_RES * dx_um, 0, MEM_RES * dx_um]
vmax = np.abs(h_true_np).max()

def _style_ax(ax):
    ax.tick_params(colors="#8b949e", labelsize=5)
    ax.set_facecolor("#0d1117")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    ax.set_xlabel("x [um]", fontsize=5, color="#8b949e")
    ax.set_ylabel("y [um]", fontsize=5, color="#8b949e")

# h_true
ax = axes[0]
im = ax.imshow(h_true_np, extent=extent, origin="lower", cmap="viridis",
               vmin=0, vmax=vmax, aspect="equal")
ax.set_title("h_true", fontsize=8, color="#e6edf3")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=5, colors="#8b949e")
_style_ax(ax)

# snapshots
for col, (it, arr, r, p) in enumerate(zip(snap_iters, snap_arrays, snap_rmse, snap_psnr)):
    ax = axes[col + 1]
    im = ax.imshow(arr, extent=extent, origin="lower", cmap="viridis",
                   vmin=0, vmax=vmax, aspect="equal")
    ax.set_title(f"iter {it}\nRMSE={r:.1f}nm\nPSNR={p:.1f}dB",
                 fontsize=7, color="#e6edf3", linespacing=1.4)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=5, colors="#8b949e")
    _style_ax(ax)

plt.tight_layout()
out_path = OUT / "random_iter_snapshots_pressed.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"\nSaved -> {out_path}")

# -----------------------------------------------------------------------
# Plot: residual (h_pred - h_true) at each snapshot
# -----------------------------------------------------------------------
fig, axes = plt.subplots(1, n_snaps, figsize=(2.8 * n_snaps, 3.5))
fig.suptitle(
    f"Residual (h_pred - h_true) evolution  [nm]",
    fontsize=10, color="#e6edf3", y=1.02
)
fig.patch.set_facecolor("#0d1117")

for col, (it, arr, r, p) in enumerate(zip(snap_iters, snap_arrays, snap_rmse, snap_psnr)):
    ax = axes[col]
    diff = arr - h_true_np
    v = max(np.abs(diff).max(), 1e-3)
    im = ax.imshow(diff, extent=extent, origin="lower", cmap="RdBu_r",
                   vmin=-v, vmax=v, aspect="equal")  # residual: still symmetric
    ax.set_title(f"iter {it}\nbias={diff.mean():.1f}nm",
                 fontsize=7, color="#e6edf3", linespacing=1.4)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=5, colors="#8b949e")
    _style_ax(ax)

plt.tight_layout()
out_path2 = OUT / "random_iter_residuals_pressed.png"
plt.savefig(out_path2, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"Saved -> {out_path2}")
