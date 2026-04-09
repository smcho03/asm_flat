"""
check_single_pixel.py
---------------------
Center pixel only deformed at 0, lambda/4, lambda/2, 3*lambda/4, lambda.
3 rows x 5 cols:
  row 0: h map
  row 1: CMOS intensity
  row 2: CMOS phase (unwrapped)
"""

import os
import numpy as np
import torch, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.restoration import unwrap_phase
sys.path.insert(0, str(Path(__file__).parent))
from sensor_model import HolographicSensor
from sensor_params import wavelength, mem_res, mem_pitch, cmos_res, grid_res, distance
from sensor_utils import STYLE

os.makedirs("output", exist_ok=True)

device = "cpu"
sensor = HolographicSensor(
    wavelength=wavelength, mem_res=mem_res, mem_pitch=mem_pitch,
    cmos_res=cmos_res, grid_res=grid_res, distance=distance, device=device,
).to(device)

lam = wavelength
amplitudes = [0, lam/4, lam/2, lam*3/4, lam]
labels     = ["h = 0", "h = λ/4", "h = λ/2", "h = 3λ/4", "h = λ"]

mem_coords  = (np.arange(mem_res)  - (mem_res  - 1) * 0.5) * mem_pitch
cmos_coords = (np.arange(cmos_res) - (cmos_res - 1) * 0.5) * mem_pitch
m_um = mem_coords  * 1e6
c_um = cmos_coords * 1e6
ext_m = [m_um.min(), m_um.max(), m_um.min(), m_um.max()]
ext_c = [c_um.min(), c_um.max(), c_um.min(), c_um.max()]

results = []
for amp in amplitudes:
    h = torch.zeros(mem_res, mem_res, dtype=torch.float32, device=device)
    c = (mem_res - 1) // 2
    h[c, c] = amp

    with torch.no_grad():
        I  = sensor(h)
        Ud = sensor.propagated_field(h)
        Uc = sensor.crop(Ud)

    results.append({
        "h_nm"  : h.cpu().numpy() * 1e9,
        "I"     : I.cpu().numpy(),
        "phi"   : unwrap_phase(np.angle(Uc.cpu().numpy()).astype(np.float64)).astype(np.float32),
    })
    print(f"  done: {amp*1e9:.1f} nm", flush=True)

# --- 공통 스케일 ---
I_max   = max(r["I"].max() for r in results)
phi_max = max(abs(r["phi"].max()) for r in results)
phi_min = min(r["phi"].min() for r in results)

plt.rcParams.update(STYLE)
fig, axes = plt.subplots(3, 5, figsize=(22, 12))
fig.suptitle(
    f"Single center-pixel deformation sweep\n"
    f"(mem pitch = {mem_pitch*1e6:.0f} um,  lambda = {lam*1e9:.1f} nm)",
    fontsize=12, color="#e6edf3",
)
fig.patch.set_facecolor("#0d1117")

def _im(ax, data, ext, cmap, vmin, vmax, label, title):
    im = ax.imshow(data.T, extent=ext, origin="lower", aspect="auto",
                   cmap=cmap, vmin=vmin, vmax=vmax)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(label, color="#8b949e", fontsize=7)
    cb.ax.yaxis.set_tick_params(labelsize=6, color="#8b949e")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="#8b949e")
    ax.set_title(title, fontsize=9)
    ax.set_facecolor("#0d1117")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    ax.tick_params(colors="#8b949e", labelsize=6)

for col, (r, label) in enumerate(zip(results, labels)):
    # row 0: h map — zoom to center 5x5 pixels to see single pixel
    zoom = 5
    c = mem_res // 2
    h_zoom = r["h_nm"][c-zoom:c+zoom+1, c-zoom:c+zoom+1]
    ext_zoom = np.array([-zoom, zoom, -zoom, zoom], dtype=float) * mem_pitch * 1e6
    _im(axes[0, col], h_zoom, ext_zoom, "viridis",
        0, max(float(h_zoom.max()), 1e-3), "h [nm]",
        f"{label}\n({amplitudes[col]*1e9:.1f} nm)")
    axes[0, col].set_xlabel("y [um]")
    axes[0, col].set_ylabel("x [um]")

    # row 1: CMOS intensity — zoom to centre 200x200 um
    zoom_um = 200
    mask = np.abs(c_um) <= zoom_um
    idx  = np.where(mask)[0]
    s, e = idx[0], idx[-1] + 1
    ext_cz = [c_um[s], c_um[e-1], c_um[s], c_um[e-1]]
    _im(axes[1, col], r["I"][s:e, s:e], ext_cz, "inferno",
        0, I_max if I_max > 0 else 1, "I [a.u.]", "CMOS Intensity (zoom)")
    axes[1, col].set_xlabel("y' [um]")
    axes[1, col].set_ylabel("x' [um]")

    # row 2: CMOS phase — same zoom
    _im(axes[2, col], r["phi"][s:e, s:e], ext_cz, "viridis",
        phi_min, phi_max, "phi [rad]", "CMOS Phase (zoom, unwrapped)")
    axes[2, col].set_xlabel("y' [um]")
    axes[2, col].set_ylabel("x' [um]")

plt.tight_layout()
out = "output/check_single_pixel.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"Saved -> {out}")
