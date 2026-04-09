"""
optimize_h.py
-------------
Inverse problem: recover membrane height h from CMOS intensity.

Pipeline:
  1. Generate ground-truth h (Gaussian bump)
  2. Run forward model -> target I_cmos
  3. Initialize h_pred = 0, optimize with Adam to minimize MSE(I_pred, I_target)
  4. Visualize: h_true vs h_pred, I_true vs I_pred, loss curve
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from sensor_model  import HolographicSensor, gaussian_bump
from sensor_params import wavelength, mem_res, mem_pitch, cmos_res, grid_res, distance
from sensor_utils  import STYLE

os.makedirs("output", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

# -----------------------------------------------------------------------
# 1. Sensor model
# -----------------------------------------------------------------------
sensor = HolographicSensor(
    wavelength=wavelength, mem_res=mem_res, mem_pitch=mem_pitch,
    cmos_res=cmos_res, grid_res=grid_res, distance=distance, device=device,
).to(device)

# -----------------------------------------------------------------------
# 2. Ground-truth h & target I
# -----------------------------------------------------------------------
h_true = gaussian_bump(
    N=mem_res, dx=mem_pitch,
    amplitude=200e-9,   # 200 nm (< lambda/2 -> single-valued regime)
    sigma=150e-6,
    device=device,
)

with torch.no_grad():
    I_target = sensor(h_true)   # [cmos_res, cmos_res]

print(f"I_target  min/max: {I_target.min():.4f} / {I_target.max():.4f}")

# -----------------------------------------------------------------------
# 3. Optimizable h
# -----------------------------------------------------------------------
h_pred = torch.zeros(mem_res, mem_res, dtype=torch.float32,
                     device=device, requires_grad=True)

optimizer = torch.optim.Adam([h_pred], lr=5e-10)

N_iter    = 3000
log_every = 300
lambda_tv  = 1e-20   # TV (1차 미분)
lambda_lap = 1e-40   # Laplacian (2차 미분)
losses        = []
losses_mse    = []
losses_tv     = []
losses_lap    = []

print(f"\nOptimizing for {N_iter} iterations ...  "
      f"(lambda_tv={lambda_tv:.0e}, lambda_lap={lambda_lap:.0e})")
for i in range(N_iter):
    optimizer.zero_grad()

    I_pred = sensor(h_pred)
    mse    = torch.mean((I_pred - I_target) ** 2)

    tv  = (torch.mean(torch.abs(h_pred[1:, :] - h_pred[:-1, :])) +
           torch.mean(torch.abs(h_pred[:, 1:] - h_pred[:, :-1])))

    lap = (h_pred[2:, 1:-1] + h_pred[:-2, 1:-1] +
           h_pred[1:-1, 2:] + h_pred[1:-1, :-2] -
           4.0 * h_pred[1:-1, 1:-1])
    reg_lap = torch.mean(lap ** 2)

    loss = mse + lambda_tv * tv + lambda_lap * reg_lap

    loss.backward()
    optimizer.step()

    losses.append(float(loss.item()))
    losses_mse.append(float(mse.item()))
    losses_tv.append(float(tv.item()))
    losses_lap.append(float(reg_lap.item()))

    if (i + 1) % log_every == 0:
        print(f"  iter {i+1:4d}  loss={loss.item():.4e}  "
              f"mse={mse.item():.4e}  tv={tv.item():.4e}  lap={reg_lap.item():.4e}  "
              f"h_pred max={h_pred.abs().max().item()*1e9:.2f} nm", flush=True)

print("Done.")

# -----------------------------------------------------------------------
# 4. Visualize
# -----------------------------------------------------------------------
mem_coords  = (np.arange(mem_res)  - (mem_res  - 1) * 0.5) * mem_pitch
cmos_coords = (np.arange(cmos_res) - (cmos_res - 1) * 0.5) * mem_pitch
m_um = mem_coords  * 1e6
c_um = cmos_coords * 1e6
ext_m = [m_um.min(), m_um.max(), m_um.min(), m_um.max()]
ext_c = [c_um.min(), c_um.max(), c_um.min(), c_um.max()]

h_true_nm = h_true.detach().cpu().numpy() * 1e9
h_pred_nm = h_pred.detach().cpu().numpy() * 1e9
I_true_np = I_target.cpu().numpy()
I_pred_np = I_pred.detach().cpu().numpy()
diff_h_nm = h_pred_nm - h_true_nm
diff_I    = I_pred_np - I_true_np

# zoom CMOS to centre
zoom_um = 500
mask = np.abs(c_um) <= zoom_um
idx  = np.where(mask)[0]
s, e = idx[0], idx[-1] + 1
ext_cz = [c_um[s], c_um[e-1], c_um[s], c_um[e-1]]

plt.rcParams.update(STYLE)
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
fig.suptitle(
    f"Inverse problem: recover h from CMOS intensity\n"
    f"Adam lr=5e-10,  {N_iter} iters,  tv={lambda_tv:.0e}  lap={lambda_lap:.0e},  h_true amp=200nm sigma=150um",
    fontsize=11, color="#e6edf3",
)
fig.patch.set_facecolor("#0d1117")

def _im(ax, data, ext, cmap, vmin, vmax, label, title):
    im = ax.imshow(data.T, extent=ext, origin="lower", aspect="equal",
                   cmap=cmap, vmin=vmin, vmax=vmax)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(label, color="#8b949e", fontsize=7)
    cb.ax.yaxis.set_tick_params(labelsize=6, color="#8b949e")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="#8b949e")
    ax.set_title(title, fontsize=9, color="#e6edf3")
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e", labelsize=6)
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

h_max = max(h_true_nm.max(), abs(h_pred_nm).max(), 1.0)
I_max = max(I_true_np.max(), I_pred_np.max())
dh_abs = max(abs(diff_h_nm).max(), 1e-3)
dI_abs = max(abs(diff_I).max(), 1e-10)

# row 0: h
_im(axes[0,0], h_true_nm, ext_m, "RdBu",   -h_max,  h_max,  "h [nm]",   "h_true")
_im(axes[0,1], h_pred_nm, ext_m, "RdBu",   -h_max,  h_max,  "h [nm]",   "h_pred")
_im(axes[0,2], diff_h_nm, ext_m, "RdBu",   -dh_abs, dh_abs, "dh [nm]",  "h_pred - h_true")

# loss curve
ax_l = axes[0,3]
ax_l.set_facecolor("#0d1117")
ax_l.plot(losses,     color="#58a6ff", lw=1.2, label="total")
ax_l.plot(losses_mse, color="#f78166", lw=1.0, linestyle="--", label="MSE")
ax_l.plot(losses_tv,  color="#3fb950", lw=1.0, linestyle=":",  label="TV")
ax_l.plot(losses_lap, color="#e3b341", lw=1.0, linestyle="-.", label="Laplacian")
ax_l.legend(fontsize=7, labelcolor="#e6edf3", framealpha=0.3)
ax_l.set_xlabel("iteration", color="#8b949e", fontsize=8)
ax_l.set_ylabel("loss", color="#8b949e", fontsize=8)
ax_l.set_title("Loss curve", fontsize=9, color="#e6edf3")
ax_l.set_yscale("log")
ax_l.tick_params(colors="#8b949e", labelsize=7)
ax_l.grid(color="#30363d", linewidth=0.4, linestyle=":")
for sp in ax_l.spines.values(): sp.set_edgecolor("#30363d")

# row 1: I (zoomed)
_im(axes[1,0], I_true_np[s:e,s:e], ext_cz, "inferno", 0, I_max, "I [a.u.]", "I_target (zoom)")
_im(axes[1,1], I_pred_np[s:e,s:e], ext_cz, "inferno", 0, I_max, "I [a.u.]", "I_pred (zoom)")
_im(axes[1,2], diff_I[s:e,s:e],    ext_cz, "RdBu", -dI_abs, dI_abs, "dI [a.u.]", "I_pred - I_target (zoom)")

# center profile comparison
ax_p = axes[1,3]
ax_p.set_facecolor("#0d1117")
ax_p.plot(m_um, h_true_nm[mem_res//2, :], color="#58a6ff", lw=1.5, label="h_true")
ax_p.plot(m_um, h_pred_nm[mem_res//2, :], color="#f78166", lw=1.5, linestyle="--", label="h_pred")
ax_p.set_xlabel("x [um]", color="#8b949e", fontsize=8)
ax_p.set_ylabel("h [nm]", color="#8b949e", fontsize=8)
ax_p.set_title("Centre cross-section", fontsize=9, color="#e6edf3")
ax_p.tick_params(colors="#8b949e", labelsize=7)
ax_p.legend(fontsize=8, labelcolor="#e6edf3", framealpha=0.3)
ax_p.grid(color="#30363d", linewidth=0.4, linestyle=":")
for sp in ax_p.spines.values(): sp.set_edgecolor("#30363d")

plt.tight_layout()
out = "output/optimize_h.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"Saved -> {out}")
