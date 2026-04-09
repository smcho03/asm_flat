"""
check_center_brightness_img.py
-------------------------------
CMOS intensity images for single center pixel at 0, lambda/4, lambda/2, 3*lambda/4, lambda
1 row x 5 cols, same colorscale, equal aspect
"""

import os
import numpy as np
import torch, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from sensor_model import HolographicSensor
from sensor_params import wavelength, mem_res, mem_pitch, cmos_res, grid_res, distance
from sensor_utils import STYLE

os.makedirs("output", exist_ok=True)

device = "cpu"
sensor = HolographicSensor(
    wavelength=wavelength, mem_res=mem_res, mem_pitch=mem_pitch,
    cmos_res=cmos_res, grid_res=grid_res, distance=distance, device=device,
).to(device)

lam        = wavelength
amplitudes = [0, lam/4, lam/2, lam*3/4, lam]
labels     = ["h = 0", "h = λ/4\n(158 nm)", "h = λ/2\n(316 nm)", "h = 3λ/4\n(475 nm)", "h = λ\n(633 nm)"]

cmos_coords = (np.arange(cmos_res) - (cmos_res - 1) * 0.5) * mem_pitch
c_um = cmos_coords * 1e6

# zoom 범위: 링이 잘 보이는 크기로
zoom_um = 300
mask = np.abs(c_um) <= zoom_um
idx  = np.where(mask)[0]
s, e = idx[0], idx[-1] + 1
ext_z = [c_um[s], c_um[e-1], c_um[s], c_um[e-1]]

frames = []
for amp in amplitudes:
    h = torch.zeros(mem_res, mem_res, dtype=torch.float32, device=device)
    c = (mem_res - 1) // 2
    h[c, c] = float(amp)
    with torch.no_grad():
        I = sensor(h)
    frames.append(I.cpu().numpy()[s:e, s:e])

I_max = max(f.max() for f in frames)

plt.rcParams.update(STYLE)
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
fig.suptitle(
    f"CMOS intensity — single center pixel deformation\n"
    f"lambda = {lam*1e9:.1f} nm,  d = {distance*1e3:.0f} mm,  zoom = ±{zoom_um} um",
    fontsize=11, color="#e6edf3",
)
fig.patch.set_facecolor("#0d1117")

for col, (I_z, label) in enumerate(zip(frames, labels)):
    ax = axes[col]
    im = ax.imshow(I_z.T, extent=ext_z, origin="lower", aspect="equal",
                   cmap="inferno", vmin=0, vmax=I_max)
    ax.set_title(label, fontsize=10, color="#e6edf3")
    ax.set_xlabel("y' [um]", color="#8b949e", fontsize=8)
    ax.set_ylabel("x' [um]", color="#8b949e", fontsize=8)
    ax.tick_params(colors="#8b949e", labelsize=7)
    ax.set_facecolor("#0d1117")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

    # 중앙 픽셀 밝기 표시
    cx = cmos_res // 2
    center_val = frames[col][cx - s, cx - s]
    ax.text(0, zoom_um * 0.88,
            f"center I = {center_val:.4f}",
            ha="center", fontsize=7, color="#8b949e")

# 공통 colorbar
fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
sm = plt.cm.ScalarMappable(cmap="inferno",
                            norm=plt.Normalize(vmin=0, vmax=I_max))
cb = fig.colorbar(sm, cax=cbar_ax)
cb.set_label("I [a.u.]", color="#8b949e", fontsize=8)
cb.ax.yaxis.set_tick_params(labelsize=7, color="#8b949e")
plt.setp(cb.ax.yaxis.get_ticklabels(), color="#8b949e")

out = "output/check_center_brightness_img.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"Saved -> {out}")
