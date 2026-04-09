"""
check_farfield_gaussian.py
--------------------------
Small Gaussian bump far-field check.

For small h:  U0 ≈ 1 + i*(4pi/lam)*h
-> delta_I in far field ∝ |FT(h)|^2  (should be Gaussian for Gaussian h)

3 rows x N_dist cols:
  row 0: h map
  row 1: delta_I  (I_def - I_ref)
  row 2: delta_I centre cross-section vs Gaussian fit
"""

import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from sensor_model  import HolographicSensor, gaussian_bump
from sensor_params import wavelength, mem_res, mem_pitch, cmos_res, grid_res
from sensor_utils  import STYLE

os.makedirs("output", exist_ok=True)

device = "cpu"
lam    = wavelength

# small amplitude: linear regime (h << lambda)
amp   = lam / 50    # ~12.7 nm, very small
sigma = 50e-6       # 50 um sigma (tight Gaussian)

distances = [5e-3, 20e-3, 50e-3, 100e-3, 200e-3]
labels    = ["5 mm", "20 mm", "50 mm", "100 mm", "200 mm"]

cmos_coords = (np.arange(cmos_res) - (cmos_res - 1) * 0.5) * mem_pitch
mem_coords  = (np.arange(mem_res)  - (mem_res  - 1) * 0.5) * mem_pitch
c_um = cmos_coords * 1e6
m_um = mem_coords  * 1e6
ext_m = [m_um.min(), m_um.max(), m_um.min(), m_um.max()]

h_def  = gaussian_bump(N=mem_res, dx=mem_pitch, amplitude=amp, sigma=sigma, device=device)
h_zero = torch.zeros_like(h_def)
h_nm   = h_def.cpu().numpy() * 1e9

# theoretical FT: FT of Gaussian(sigma) = Gaussian(1/(2pi*sigma)) in freq domain
# in CMOS space (distance d): sigma_cmos = lam*d / (2*pi*sigma)
def gaussian_fit(x_um, sigma_um):
    return np.exp(-x_um**2 / (2 * sigma_um**2))

print("Computing ...", flush=True)
results = []
for d, label in zip(distances, labels):
    sensor = HolographicSensor(
        wavelength=lam, mem_res=mem_res, mem_pitch=mem_pitch,
        cmos_res=cmos_res, grid_res=grid_res, distance=d, device=device,
    ).to(device)

    with torch.no_grad():
        I_def  = sensor(h_def).cpu().numpy()
        I_ref  = sensor(h_zero).cpu().numpy()

    dI = I_def - I_ref

    # theoretical sigma in CMOS plane
    sigma_cmos_um = (lam * d) / (2 * np.pi * sigma) * 1e6

    results.append({"dI": dI, "sigma_cmos_um": sigma_cmos_um, "d": d})
    print(f"  d={label}  dI_max={abs(dI).max():.4e}  sigma_cmos_theory={sigma_cmos_um:.1f}um", flush=True)

dI_abs = max(abs(r["dI"]).max() for r in results)

# zoom
zoom_um = 800
mask = np.abs(c_um) <= zoom_um
idx  = np.where(mask)[0]
s, e = idx[0], idx[-1] + 1
ext_cz = [c_um[s], c_um[e-1], c_um[s], c_um[e-1]]

plt.rcParams.update(STYLE)
fig, axes = plt.subplots(3, len(distances), figsize=(22, 12))
fig.suptitle(
    f"Far-field Gaussian test  (h = Gaussian, amp={amp*1e9:.1f} nm, sigma={sigma*1e6:.0f} um)\n"
    f"Prediction: delta_I in far field -> Gaussian  (FT of Gaussian = Gaussian)",
    fontsize=11, color="#e6edf3",
)
fig.patch.set_facecolor("#0d1117")

def _ax_style(ax):
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e", labelsize=6)
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

def _cb(fig, ax, im, label):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(label, color="#8b949e", fontsize=7)
    cb.ax.yaxis.set_tick_params(labelsize=6, color="#8b949e")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="#8b949e")

# zoom h to ±5*sigma
zoom_h_um = sigma * 1e6 * 5
mask_h = np.abs(m_um) <= zoom_h_um
idx_h  = np.where(mask_h)[0]
sh, eh = idx_h[0], idx_h[-1] + 1
ext_hz = [m_um[sh], m_um[eh-1], m_um[sh], m_um[eh-1]]

for col, (r, label) in enumerate(zip(results, labels)):
    # row 0: h map
    ax = axes[0, col]
    im = ax.imshow(h_nm[sh:eh, sh:eh].T, extent=ext_hz, origin="lower",
                   aspect="equal", cmap="viridis", vmin=0, vmax=float(h_nm.max()))
    _cb(fig, ax, im, "h [nm]")
    ax.set_title(f"d = {label}", fontsize=10, color="#e6edf3")
    ax.set_xlabel("y [um]", color="#8b949e", fontsize=7)
    ax.set_ylabel("x [um]", color="#8b949e", fontsize=7)
    _ax_style(ax)

    # row 1: delta_I (zoomed)
    ax = axes[1, col]
    dI_z = r["dI"][s:e, s:e]
    norm = TwoSlopeNorm(vmin=-dI_abs, vcenter=0, vmax=dI_abs)
    im = ax.imshow(dI_z.T, extent=ext_cz, origin="lower",
                   aspect="equal", cmap="RdBu", norm=norm)
    _cb(fig, ax, im, "dI [a.u.]")
    ax.set_title("delta_I  (I_def - I_ref)", fontsize=8, color="#e6edf3")
    ax.set_xlabel("y' [um]", color="#8b949e", fontsize=7)
    ax.set_ylabel("x' [um]", color="#8b949e", fontsize=7)
    _ax_style(ax)

    # row 2: cross-section + Gaussian fit
    ax = axes[2, col]
    dI_row  = r["dI"][cmos_res//2, s:e]
    c_um_z  = c_um[s:e]

    # normalize to peak for shape comparison
    peak = dI_row.max()
    if abs(peak) > 1e-10:
        dI_norm = dI_row / peak
    else:
        dI_norm = dI_row

    sigma_cmos = r["sigma_cmos_um"]
    theory     = gaussian_fit(c_um_z, sigma_cmos)

    ax.plot(c_um_z, dI_norm, color="#58a6ff", lw=1.2, label="delta_I (norm)")
    ax.plot(c_um_z, theory,  color="#f78166", lw=1.2, linestyle="--",
            label=f"Gaussian fit\nsigma={sigma_cmos:.0f}um")
    ax.set_xlabel("y' [um]", color="#8b949e", fontsize=7)
    ax.set_ylabel("delta_I (norm)", color="#8b949e", fontsize=7)
    ax.set_title("Cross-section vs theory", fontsize=8, color="#e6edf3")
    ax.legend(fontsize=6, labelcolor="#e6edf3", framealpha=0.3)
    ax.grid(color="#30363d", linewidth=0.4, linestyle=":")
    _ax_style(ax)

plt.tight_layout()
out = "output/check_farfield_gaussian.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"Saved -> {out}")
