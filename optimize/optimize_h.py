"""
optimize_h.py
-------------
Inverse problem: recover membrane height h from CMOS intensity.

Parameterization
  h = h_var * H_SCALE   (h_var is the trainable tensor, O(1) magnitude)
This keeps gradient magnitudes in a normal range so Adam with lr ~ 1e-3
behaves like in standard ML training (instead of requiring lr ~ 1e-10 when h
is parameterized directly in meters).

Output:
  output/optimize_h/
    reconstruction.png           - qualitative 2x4 panel (h, I, profile, loss)
    metrics.txt                  - quantitative summary (PSNR, RMSE, loss)
"""

import sys, json, time
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from sensor_model  import HolographicSensor, gaussian_bump
from sensor_params import wavelength, mem_pitch, distance
from sensor_utils  import STYLE

# CPU-friendly resolution for quick demonstrations
mem_res  = 256
cmos_res = 512
grid_res = 768

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "output" / "optimize_h"
OUT.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
AMP      = 200e-9
SIGMA    = 150e-6
H_SCALE  = 500e-9        # typical peak scale; h_var then lives in ~[-1, 1]
LR       = 1e-3          # natural ML-scale learning rate
N_ITER   = 2000
LOG_EVERY = 200

# Regularization (scaled for h_var, dimensionless)
lambda_tv  = 0.0
lambda_lap = 0.0

# -----------------------------------------------------------------------
# Forward model + ground truth
# -----------------------------------------------------------------------
sensor = HolographicSensor(
    wavelength=wavelength, mem_res=mem_res, mem_pitch=mem_pitch,
    cmos_res=cmos_res, grid_res=grid_res, distance=distance, device=device,
).to(device)

h_true = gaussian_bump(N=mem_res, dx=mem_pitch, amplitude=AMP,
                       sigma=SIGMA, device=device)
with torch.no_grad():
    I_target = sensor(h_true)

print(f"I_target  min/max: {I_target.min():.4f} / {I_target.max():.4f}")

# -----------------------------------------------------------------------
# Optimization (h_var * H_SCALE parameterization)
# -----------------------------------------------------------------------
h_var = torch.zeros(mem_res, mem_res, dtype=torch.float32,
                    device=device, requires_grad=True)
opt   = torch.optim.Adam([h_var], lr=LR)

losses, losses_mse, losses_tv, losses_lap = [], [], [], []
print(f"\nOptimizing for {N_ITER} iters, lr={LR}, H_SCALE={H_SCALE*1e9:.0f}nm")

t0 = time.time()
for i in range(N_ITER):
    opt.zero_grad()
    h_pred = h_var * H_SCALE
    I_pred = sensor(h_pred)
    mse    = torch.mean((I_pred - I_target) ** 2)

    tv  = (torch.mean(torch.abs(h_var[1:, :] - h_var[:-1, :])) +
           torch.mean(torch.abs(h_var[:, 1:] - h_var[:, :-1])))
    lap = (h_var[2:, 1:-1] + h_var[:-2, 1:-1] +
           h_var[1:-1, 2:] + h_var[1:-1, :-2] -
           4.0 * h_var[1:-1, 1:-1])
    reg_lap = torch.mean(lap ** 2)

    loss = mse + lambda_tv * tv + lambda_lap * reg_lap
    loss.backward()
    opt.step()

    losses.append(float(loss.item()))
    losses_mse.append(float(mse.item()))
    losses_tv.append(float(tv.item()))
    losses_lap.append(float(reg_lap.item()))

    if (i + 1) % LOG_EVERY == 0:
        h_abs_nm = (h_var.detach() * H_SCALE).abs().max().item() * 1e9
        print(f"  iter {i+1:4d}  loss={loss.item():.4e}  mse={mse.item():.4e}  "
              f"h_pred_max={h_abs_nm:.2f} nm", flush=True)

dt = time.time() - t0
print(f"Done in {dt:.1f}s.")

with torch.no_grad():
    h_pred_final = h_var.detach() * H_SCALE
    I_pred_final = sensor(h_pred_final)

# -----------------------------------------------------------------------
# Quantitative metrics
# -----------------------------------------------------------------------
def rmse_nm(a, b): return float(torch.sqrt(torch.mean((a - b)**2))) * 1e9
def psnr_db(a, b):
    mse_v = float(torch.mean((a - b)**2))
    peak  = float(b.abs().max())
    if mse_v < 1e-30: return 100.0
    return 10.0 * np.log10(peak**2 / mse_v)

final_psnr  = psnr_db(h_pred_final, h_true)
final_rmse  = rmse_nm(h_pred_final, h_true)
final_loss  = losses[-1]
I_rel_err   = float(torch.mean(torch.abs(I_pred_final - I_target)) /
                    (I_target.mean() + 1e-12))
h_peak_pred = float(h_pred_final.abs().max()) * 1e9
h_peak_true = float(h_true.abs().max()) * 1e9

metrics_txt = (
    "optimize_h.py — Quantitative Results\n"
    "====================================\n"
    f"Config\n"
    f"  amplitude (GT)     : {AMP*1e9:.1f} nm\n"
    f"  sigma (GT)         : {SIGMA*1e6:.1f} um\n"
    f"  distance           : {distance*1e3:.1f} mm\n"
    f"  mem_res / pitch    : {mem_res} / {mem_pitch*1e6:.1f} um\n"
    f"  cmos_res           : {cmos_res}\n"
    f"  H_SCALE            : {H_SCALE*1e9:.1f} nm\n"
    f"  learning rate      : {LR:.0e}\n"
    f"  iterations         : {N_ITER}\n"
    f"  lambda_tv/lap      : {lambda_tv:.0e} / {lambda_lap:.0e}\n"
    f"  wall time          : {dt:.2f} s\n"
    f"\nMetrics\n"
    f"  final loss         : {final_loss:.6e}\n"
    f"  PSNR (h)           : {final_psnr:.2f} dB\n"
    f"  RMSE (h)           : {final_rmse:.3f} nm\n"
    f"  peak h_pred        : {h_peak_pred:.2f} nm\n"
    f"  peak h_true        : {h_peak_true:.2f} nm\n"
    f"  relative |dI|      : {I_rel_err:.4e}\n"
)
(OUT / "metrics.txt").write_text(metrics_txt, encoding="utf-8")
print("\n" + metrics_txt)

(OUT / "metrics.json").write_text(json.dumps(dict(
    amp_nm=AMP*1e9, sigma_um=SIGMA*1e6, distance_mm=distance*1e3,
    H_SCALE_nm=H_SCALE*1e9, lr=LR, n_iter=N_ITER,
    final_loss=final_loss, psnr_db=final_psnr, rmse_nm=final_rmse,
    h_peak_pred_nm=h_peak_pred, h_peak_true_nm=h_peak_true,
    wall_time_s=dt,
), indent=2))

# -----------------------------------------------------------------------
# Qualitative figure
# -----------------------------------------------------------------------
mem_coords  = (np.arange(mem_res)  - (mem_res  - 1) * 0.5) * mem_pitch
cmos_coords = (np.arange(cmos_res) - (cmos_res - 1) * 0.5) * mem_pitch
m_um = mem_coords  * 1e6
c_um = cmos_coords * 1e6
ext_m = [m_um.min(), m_um.max(), m_um.min(), m_um.max()]

h_true_nm = h_true.detach().cpu().numpy() * 1e9
h_pred_nm = h_pred_final.detach().cpu().numpy() * 1e9
I_true_np = I_target.cpu().numpy()
I_pred_np = I_pred_final.detach().cpu().numpy()
diff_h_nm = h_pred_nm - h_true_nm
diff_I    = I_pred_np - I_true_np

zoom_um = 500
mask = np.abs(c_um) <= zoom_um
idx  = np.where(mask)[0]
s, e = idx[0], idx[-1] + 1
ext_cz = [c_um[s], c_um[e-1], c_um[s], c_um[e-1]]

plt.rcParams.update(STYLE)
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
fig.suptitle(
    f"Inverse problem (scaled param, h = h_var * {H_SCALE*1e9:.0f}nm)\n"
    f"Adam lr={LR:.0e}, {N_ITER} iters  |  PSNR={final_psnr:.1f}dB  RMSE={final_rmse:.2f}nm",
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

h_max  = max(h_true_nm.max(), abs(h_pred_nm).max(), 1.0)
I_max  = max(I_true_np.max(), I_pred_np.max())
dh_abs = max(abs(diff_h_nm).max(), 1e-3)
dI_abs = max(abs(diff_I).max(), 1e-10)

_im(axes[0,0], h_true_nm, ext_m, "RdBu", -h_max, h_max, "h [nm]", "h_true")
_im(axes[0,1], h_pred_nm, ext_m, "RdBu", -h_max, h_max, "h [nm]", "h_pred")
_im(axes[0,2], diff_h_nm, ext_m, "RdBu", -dh_abs, dh_abs, "dh [nm]", "h_pred - h_true")

ax_l = axes[0,3]
ax_l.set_facecolor("#0d1117")
ax_l.plot(losses,     color="#58a6ff", lw=1.2, label="total")
ax_l.plot(losses_mse, color="#f78166", lw=1.0, ls="--", label="MSE")
ax_l.legend(fontsize=7, labelcolor="#e6edf3", framealpha=0.3)
ax_l.set_xlabel("iteration", color="#8b949e", fontsize=8)
ax_l.set_ylabel("loss", color="#8b949e", fontsize=8)
ax_l.set_title("Loss curve", fontsize=9, color="#e6edf3")
ax_l.set_yscale("log")
ax_l.tick_params(colors="#8b949e", labelsize=7)
ax_l.grid(color="#30363d", lw=0.4, ls=":")
for sp in ax_l.spines.values(): sp.set_edgecolor("#30363d")

_im(axes[1,0], I_true_np[s:e,s:e], ext_cz, "inferno", 0, I_max, "I [a.u.]", "I_target (zoom)")
_im(axes[1,1], I_pred_np[s:e,s:e], ext_cz, "inferno", 0, I_max, "I [a.u.]", "I_pred (zoom)")
_im(axes[1,2], diff_I[s:e,s:e],    ext_cz, "RdBu", -dI_abs, dI_abs, "dI [a.u.]", "I_pred - I_target")

ax_p = axes[1,3]
ax_p.set_facecolor("#0d1117")
ax_p.plot(m_um, h_true_nm[mem_res//2, :], color="#58a6ff", lw=1.5, label="h_true")
ax_p.plot(m_um, h_pred_nm[mem_res//2, :], color="#f78166", lw=1.5, ls="--", label="h_pred")
ax_p.set_xlabel("x [um]", color="#8b949e", fontsize=8)
ax_p.set_ylabel("h [nm]", color="#8b949e", fontsize=8)
ax_p.set_title("Centre cross-section", fontsize=9, color="#e6edf3")
ax_p.tick_params(colors="#8b949e", labelsize=7)
ax_p.legend(fontsize=8, labelcolor="#e6edf3", framealpha=0.3)
ax_p.grid(color="#30363d", lw=0.4, ls=":")
for sp in ax_p.spines.values(): sp.set_edgecolor("#30363d")

plt.tight_layout()
fig_path = OUT / "reconstruction.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"Saved -> {fig_path}")
