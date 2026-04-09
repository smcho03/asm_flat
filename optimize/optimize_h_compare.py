"""
optimize_h_compare.py
---------------------
Compare three regularization strategies for the inverse problem
(recovering membrane height h from CMOS intensity):

  Config A:  MSE only                    (no regularization)
  Config B:  MSE + TV  (1st derivative)  (promotes piecewise-flat h)
  Config C:  MSE + Laplacian (2nd deriv) (promotes smooth h)

Output:
  optimize_h_compare.png  – 3-col comparison:
    row 0 : loss curves  (total, mse, reg)
    row 1 : h_pred vs h_true  cross-section
    row 2 : I_pred vs I_target  (centre crop)
"""

import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sensor_model  import HolographicSensor, gaussian_bump
from sensor_params import wavelength, mem_res, mem_pitch, cmos_res, grid_res, distance
from sensor_utils  import STYLE

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

# -----------------------------------------------------------------------
# Ground-truth
# -----------------------------------------------------------------------
sensor = HolographicSensor(
    wavelength=wavelength, mem_res=mem_res, mem_pitch=mem_pitch,
    cmos_res=cmos_res, grid_res=grid_res, distance=distance, device=device,
).to(device)

h_true = gaussian_bump(
    N=mem_res, dx=mem_pitch,
    amplitude=200e-9,
    sigma=150e-6,
    device=device,
)
with torch.no_grad():
    I_target = sensor(h_true)
print(f"I_target  min={I_target.min():.4f}  max={I_target.max():.4f}")

# -----------------------------------------------------------------------
# Regularization strength calibration
# -----------------------------------------------------------------------
# Estimate TV and Laplacian magnitudes for a zero h (start of optimization):
# We set lambda so that reg_term ~ 0.1 * mse_initial.
#
# From a quick estimate with Gaussian h:
#   TV         ~ 1e-9  m  (mean absolute gradient step)
#   reg_lap    ~ 1e-28     (mean squared Laplacian step)
#   mse_init   ~ 0.01-0.1  (intensity difference squared)
#
# So:
#   lambda_tv  ~ 0.01 / 1e-9  = 1e7   (strong TV)
#   lambda_lap ~ 0.01 / 1e-28 = 1e26  (strong Laplacian)
#
# We run both moderate and strong, but show one set.

N_ITER    = 500
LOG_EVERY = 100

CONFIGS = [
    dict(name="MSE only",        color="#58a6ff", lambda_tv=0.0,   lambda_lap=0.0  ),
    dict(name="MSE + TV",        color="#3fb950", lambda_tv=5e3,   lambda_lap=0.0  ),
    dict(name="MSE + Laplacian", color="#e3b341", lambda_tv=0.0,   lambda_lap=5e14 ),
]

results = []

for cfg in CONFIGS:
    name       = cfg["name"]
    lambda_tv  = cfg["lambda_tv"]
    lambda_lap = cfg["lambda_lap"]
    print(f"\n--- {name}  (lambda_tv={lambda_tv:.0e}, lambda_lap={lambda_lap:.0e}) ---")

    h_pred = torch.zeros(mem_res, mem_res, dtype=torch.float32,
                         device=device, requires_grad=True)
    opt    = torch.optim.Adam([h_pred], lr=5e-10)

    log_loss     = []
    log_mse      = []
    log_reg      = []

    for i in range(N_ITER):
        opt.zero_grad()

        I_pred = sensor(h_pred)
        mse    = torch.mean((I_pred - I_target) ** 2)

        # 1st derivative: Total Variation
        tv  = (torch.mean(torch.abs(h_pred[1:, :] - h_pred[:-1, :])) +
               torch.mean(torch.abs(h_pred[:, 1:] - h_pred[:, :-1])))

        # 2nd derivative: Laplacian
        lap     = (h_pred[2:, 1:-1] + h_pred[:-2, 1:-1] +
                   h_pred[1:-1, 2:] + h_pred[1:-1, :-2] -
                   4.0 * h_pred[1:-1, 1:-1])
        reg_lap = torch.mean(lap ** 2)

        reg  = lambda_tv * tv + lambda_lap * reg_lap
        loss = mse + reg

        loss.backward()
        opt.step()

        log_loss.append(float(loss))
        log_mse.append(float(mse))
        log_reg.append(float(reg))

        if (i + 1) % LOG_EVERY == 0:
            h_max_nm = float(h_pred.abs().max()) * 1e9
            print(f"  iter {i+1:4d}  loss={float(loss):.3e}  "
                  f"mse={float(mse):.3e}  reg={float(reg):.3e}  "
                  f"h_max={h_max_nm:.2f} nm", flush=True)

    with torch.no_grad():
        I_final = sensor(h_pred).cpu().numpy()

    results.append(dict(
        name       = name,
        color      = cfg["color"],
        h_pred_nm  = h_pred.detach().cpu().numpy() * 1e9,
        I_final    = I_final,
        log_loss   = log_loss,
        log_mse    = log_mse,
        log_reg    = log_reg,
    ))

# -----------------------------------------------------------------------
# Common data
# -----------------------------------------------------------------------
mem_coords  = (np.arange(mem_res)  - (mem_res  - 1) * 0.5) * mem_pitch
cmos_coords = (np.arange(cmos_res) - (cmos_res - 1) * 0.5) * mem_pitch
m_um  = mem_coords  * 1e6
c_um  = cmos_coords * 1e6

h_true_nm = h_true.detach().cpu().numpy() * 1e9
I_tgt_np  = I_target.cpu().numpy()

# Centre crop for CMOS display
zoom_um = 600
mask = np.abs(c_um) <= zoom_um
idx  = np.where(mask)[0]
s_c, e_c = idx[0], idx[-1] + 1
ext_cz = [c_um[s_c], c_um[e_c - 1], c_um[s_c], c_um[e_c - 1]]

# -----------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------
plt.rcParams.update(STYLE)
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle(
    f"Regularization comparison  ({N_ITER} iterations, Adam lr=5e-10)\n"
    f"h_true: Gaussian  amp=200 nm  σ=150 μm",
    fontsize=11, color="#e6edf3",
)
fig.patch.set_facecolor("#0d1117")

def _style_ax(ax):
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")
    ax.grid(color="#30363d", linewidth=0.4, linestyle=":")

def _cb(fig, ax, im, label):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(label, color="#8b949e", fontsize=7)
    cb.ax.yaxis.set_tick_params(color="#8b949e", labelsize=6)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="#8b949e")

# ---- Row 0: Loss curves ------------------------------------------------
for col, r in enumerate(results):
    ax = axes[0, col]
    iters = np.arange(1, N_ITER + 1)
    ax.plot(iters, r["log_mse"],  color="#f78166", lw=1.2, label="MSE")
    ax.plot(iters, r["log_reg"],  color="#e3b341", lw=1.0, linestyle="--", label="Reg")
    ax.plot(iters, r["log_loss"], color="#58a6ff", lw=1.5, label="Total")
    ax.set_title(r["name"], fontsize=9, color="#e6edf3")
    ax.set_xlabel("iteration", fontsize=8, color="#8b949e")
    ax.set_ylabel("loss", fontsize=8, color="#8b949e")
    ax.set_yscale("log")
    ax.legend(fontsize=7, labelcolor="#e6edf3", framealpha=0.3)
    _style_ax(ax)

    # Print final values
    final_mse = r["log_mse"][-1]
    final_reg = r["log_reg"][-1]
    print(f"\n{r['name']}:  final MSE={final_mse:.3e}  reg={final_reg:.3e}  "
          f"total={r['log_loss'][-1]:.3e}")

# ---- Row 1: h cross-section at membrane centre -------------------------
mid = mem_res // 2
for col, r in enumerate(results):
    ax = axes[1, col]
    ax.plot(m_um, h_true_nm[mid, :], color="#8b949e", lw=1.5,
            linestyle=":", label="h_true")
    ax.plot(m_um, r["h_pred_nm"][mid, :], color=r["color"], lw=1.5,
            label="h_pred")
    ax.set_title(f"{r['name']}  — h cross-section", fontsize=9, color="#e6edf3")
    ax.set_xlabel("x [μm]", fontsize=8, color="#8b949e")
    ax.set_ylabel("h [nm]",  fontsize=8, color="#8b949e")
    ax.legend(fontsize=7, labelcolor="#e6edf3", framealpha=0.3)
    _style_ax(ax)

# ---- Row 2: CMOS intensity (zoomed) ------------------------------------
I_max = float(I_tgt_np.max())
ext_m = [m_um.min(), m_um.max(), m_um.min(), m_um.max()]

for col, r in enumerate(results):
    ax  = axes[2, col]
    arr = r["I_final"][s_c:e_c, s_c:e_c]
    im  = ax.imshow(arr.T, extent=ext_cz, origin="lower", aspect="auto",
                    cmap="inferno", vmin=0, vmax=I_max)
    _cb(fig, ax, im, "I [a.u.]")
    ax.set_title(f"{r['name']}  — I_pred (zoom ±{zoom_um}μm)", fontsize=9, color="#e6edf3")
    ax.set_xlabel("y' [μm]", fontsize=8, color="#8b949e")
    ax.set_ylabel("x' [μm]", fontsize=8, color="#8b949e")
    _style_ax(ax)

plt.tight_layout()
out = Path(__file__).parent / "output" / "optimize_h_compare.png"
out.parent.mkdir(exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"\nSaved -> {out}")
