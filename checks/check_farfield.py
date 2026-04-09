"""
check_farfield.py
-----------------
Single center pixel: CMOS pattern as distance increases toward far field.
Distances: 5mm, 20mm, 50mm, 100mm, 200mm, 500mm

Far-field (Fraunhofer) condition: d >> a^2/lambda
  a = pixel size = 10um, lambda = 632.8nm
  -> d >> 10e-6^2 / 632.8e-9 = 0.158 mm   (already satisfied at 5mm)

For a POINT source (single pixel), Fourier transform = uniform plane wave
-> CMOS should become uniformly bright as d increases
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from sensor_model  import HolographicSensor
from sensor_params import wavelength, mem_res, mem_pitch, cmos_res, grid_res
from sensor_utils  import STYLE

os.makedirs("output", exist_ok=True)

device    = "cpu"
lam       = wavelength
amp       = lam / 4
distances = [5e-3, 20e-3, 50e-3, 100e-3, 200e-3, 500e-3]
labels    = ["5 mm", "20 mm", "50 mm", "100 mm", "200 mm", "500 mm"]

cmos_coords = (np.arange(cmos_res) - (cmos_res - 1) * 0.5) * mem_pitch
c_um = cmos_coords * 1e6

# zoom: 전체 CMOS 보기
ext_c = [c_um.min(), c_um.max(), c_um.min(), c_um.max()]

frames = []
center_vals = []
std_vals    = []

print("Computing ...", flush=True)
for d in distances:
    sensor = HolographicSensor(
        wavelength=lam, mem_res=mem_res, mem_pitch=mem_pitch,
        cmos_res=cmos_res, grid_res=grid_res, distance=d, device=device,
    ).to(device)

    h = torch.zeros(mem_res, mem_res, dtype=torch.float32, device=device)
    c = (mem_res - 1) // 2
    h[c, c] = float(amp)

    with torch.no_grad():
        I = sensor(h).cpu().numpy()

    frames.append(I)
    center_vals.append(float(I[cmos_res//2, cmos_res//2]))
    std_vals.append(float(I.std()))
    print(f"  d={d*1e3:.0f}mm  I_center={center_vals[-1]:.4f}  I_std={std_vals[-1]:.4f}", flush=True)

I_max = max(f.max() for f in frames)

plt.rcParams.update(STYLE)
fig, axes = plt.subplots(2, len(distances), figsize=(22, 8))
fig.suptitle(
    f"Single center pixel  ->  far-field evolution\n"
    f"h = lambda/4 = {amp*1e9:.1f} nm,  lambda = {lam*1e9:.1f} nm\n"
    f"Far field: pattern -> Fourier transform (uniform for point source)",
    fontsize=10, color="#e6edf3",
)
fig.patch.set_facecolor("#0d1117")

for col, (I, label) in enumerate(zip(frames, labels)):
    # row 0: full CMOS image (same colorscale)
    ax = axes[0, col]
    im = ax.imshow(I.T, extent=ext_c, origin="lower", aspect="equal",
                   cmap="inferno", vmin=0, vmax=I_max)
    ax.set_title(f"d = {label}", fontsize=9, color="#e6edf3")
    ax.set_xlabel("y' [um]", color="#8b949e", fontsize=7)
    ax.set_ylabel("x' [um]", color="#8b949e", fontsize=7)
    ax.tick_params(colors="#8b949e", labelsize=6)
    ax.set_facecolor("#0d1117")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

    # row 1: centre cross-section
    ax2 = axes[1, col]
    ax2.plot(c_um, I[cmos_res//2, :], color="#58a6ff", lw=1.2)
    ax2.axhline(I.mean(), color="#f78166", lw=0.8, linestyle="--",
                label=f"mean={I.mean():.3f}")
    ax2.set_xlabel("y' [um]", color="#8b949e", fontsize=7)
    ax2.set_ylabel("I [a.u.]", color="#8b949e", fontsize=7)
    ax2.set_title(f"std={std_vals[col]:.4f}", fontsize=8, color="#e6edf3")
    ax2.tick_params(colors="#8b949e", labelsize=6)
    ax2.set_facecolor("#0d1117")
    ax2.legend(fontsize=6, labelcolor="#e6edf3", framealpha=0.3)
    ax2.grid(color="#30363d", linewidth=0.4, linestyle=":")
    for sp in ax2.spines.values(): sp.set_edgecolor("#30363d")

# colorbar
fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.90, 0.35, 0.012, 0.5])
sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(vmin=0, vmax=I_max))
cb = fig.colorbar(sm, cax=cbar_ax)
cb.set_label("I [a.u.]", color="#8b949e", fontsize=8)
cb.ax.yaxis.set_tick_params(labelsize=6, color="#8b949e")
plt.setp(cb.ax.yaxis.get_ticklabels(), color="#8b949e")

plt.tight_layout()
out = "output/check_farfield.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"Saved -> {out}")
