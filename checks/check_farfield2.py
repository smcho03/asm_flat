"""
check_farfield2.py
------------------
Far-field check: single center pixel deformation across distances.
3 rows x N_dist cols:
  row 0: h map
  row 1: CMOS intensity I
  row 2: I - I_ref  (I_ref = h=0 flat mirror)
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
from sensor_model  import HolographicSensor
from sensor_params import wavelength, mem_res, mem_pitch, cmos_res, grid_res
from sensor_utils  import STYLE

os.makedirs("output", exist_ok=True)

device    = "cpu"
lam       = wavelength
amp       = lam / 4
distances = [5e-3, 20e-3, 50e-3, 100e-3, 200e-3]
labels    = ["5 mm", "20 mm", "50 mm", "100 mm", "200 mm"]

cmos_coords = (np.arange(cmos_res) - (cmos_res - 1) * 0.5) * mem_pitch
mem_coords  = (np.arange(mem_res)  - (mem_res  - 1) * 0.5) * mem_pitch
c_um = cmos_coords * 1e6
m_um = mem_coords  * 1e6
ext_c = [c_um.min(), c_um.max(), c_um.min(), c_um.max()]
ext_m = [m_um.min(), m_um.max(), m_um.min(), m_um.max()]

# single pixel h
h_def = torch.zeros(mem_res, mem_res, dtype=torch.float32, device=device)
c = (mem_res - 1) // 2
h_def[c, c] = float(amp)
h_zero = torch.zeros_like(h_def)
h_nm = h_def.cpu().numpy() * 1e9

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
    results.append({"I": I_def, "I_ref": I_ref, "dI": dI})
    print(f"  d={label}  I max={I_def.max():.4f}  dI max={abs(dI).max():.4f}", flush=True)

I_max  = max(r["I"].max() for r in results)
dI_abs = max(abs(r["dI"]).max() for r in results)

# zoom to centre
zoom_um = 600
mask = np.abs(c_um) <= zoom_um
idx  = np.where(mask)[0]
s, e = idx[0], idx[-1] + 1
ext_cz = [c_um[s], c_um[e-1], c_um[s], c_um[e-1]]

# h zoom
zoom_h = 5  # ±5 pixels
hs = c - zoom_h; he = c + zoom_h + 1
ext_hz = [m_um[hs], m_um[he-1], m_um[hs], m_um[he-1]]

plt.rcParams.update(STYLE)
fig, axes = plt.subplots(3, len(distances), figsize=(22, 12))
fig.suptitle(
    f"Far-field evolution: single center pixel  (h = lambda/4 = {amp*1e9:.1f} nm)\n"
    f"Row 0: h map   |   Row 1: CMOS Intensity   |   Row 2: I - I_ref",
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

for col, (r, label) in enumerate(zip(results, labels)):
    # row 0: h map (zoomed to single pixel)
    ax = axes[0, col]
    im = ax.imshow(h_nm[hs:he, hs:he].T, extent=ext_hz, origin="lower",
                   aspect="equal", cmap="viridis", vmin=0, vmax=max(float(h_nm.max()), 1e-3))
    _cb(fig, ax, im, "h [nm]")
    ax.set_title(f"d = {label}", fontsize=10, color="#e6edf3")
    ax.set_xlabel("y [um]", color="#8b949e", fontsize=7)
    ax.set_ylabel("x [um]", color="#8b949e", fontsize=7)
    _ax_style(ax)

    # row 1: CMOS intensity (zoomed)
    ax = axes[1, col]
    im = ax.imshow(r["I"][s:e, s:e].T, extent=ext_cz, origin="lower",
                   aspect="equal", cmap="inferno", vmin=0, vmax=I_max)
    _cb(fig, ax, im, "I [a.u.]")
    ax.set_title(f"CMOS Intensity", fontsize=8, color="#e6edf3")
    ax.set_xlabel("y' [um]", color="#8b949e", fontsize=7)
    ax.set_ylabel("x' [um]", color="#8b949e", fontsize=7)
    _ax_style(ax)

    # row 2: I - I_ref (zoomed)
    ax = axes[2, col]
    dI_z = r["dI"][s:e, s:e]
    norm = TwoSlopeNorm(vmin=-dI_abs, vcenter=0, vmax=dI_abs)
    im = ax.imshow(dI_z.T, extent=ext_cz, origin="lower",
                   aspect="equal", cmap="RdBu", norm=norm)
    _cb(fig, ax, im, "dI [a.u.]")
    ax.set_title(f"I - I_ref", fontsize=8, color="#e6edf3")
    ax.set_xlabel("y' [um]", color="#8b949e", fontsize=7)
    ax.set_ylabel("x' [um]", color="#8b949e", fontsize=7)
    _ax_style(ax)

plt.tight_layout()
out = "output/check_farfield2.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"Saved -> {out}")
