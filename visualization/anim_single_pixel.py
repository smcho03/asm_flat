"""
anim_single_pixel.py
--------------------
Animation: center pixel deformation sweep 0 -> lambda (100 frames, 20s)
Output: output/anim_single_pixel.gif
"""

import os
import numpy as np
import torch, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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

lam      = wavelength
N_frames = 100
interval_ms = 20000 // N_frames   # 20초 / 100프레임 = 200ms

cmos_coords = (np.arange(cmos_res) - (cmos_res - 1) * 0.5) * mem_pitch
c_um = cmos_coords * 1e6

# zoom 범위
zoom_um = 200
mask = np.abs(c_um) <= zoom_um
idx  = np.where(mask)[0]
s, e = idx[0], idx[-1] + 1
ext_cz = [c_um[s], c_um[e-1], c_um[s], c_um[e-1]]

# 모든 프레임 미리 계산
print("Pre-computing all frames ...", flush=True)
frames_I   = []
amplitudes = np.linspace(0, lam, N_frames)

for i, amp in enumerate(amplitudes):
    h = torch.zeros(mem_res, mem_res, dtype=torch.float32, device=device)
    c = (mem_res - 1) // 2
    h[c, c] = float(amp)

    with torch.no_grad():
        I = sensor(h)

    frames_I.append(I.cpu().numpy()[s:e, s:e])

    if (i+1) % 10 == 0:
        print(f"  {i+1}/{N_frames}  ({amp*1e9:.1f} nm)", flush=True)

I_max = max(f.max() for f in frames_I)

print("Building animation ...", flush=True)

plt.rcParams.update(STYLE)
fig, ax = plt.subplots(1, 1, figsize=(6, 5))
fig.patch.set_facecolor("#0d1117")
title_obj = fig.suptitle("", fontsize=11, color="#e6edf3")

im0 = ax.imshow(frames_I[0].T, extent=ext_cz, origin="lower", aspect="equal",
                cmap="inferno", vmin=0, vmax=I_max)
cb0 = fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)
cb0.set_label("I [a.u.]", color="#8b949e", fontsize=7)
cb0.ax.yaxis.set_tick_params(labelsize=6, color="#8b949e")
plt.setp(cb0.ax.yaxis.get_ticklabels(), color="#8b949e")

ax.set_title("CMOS Intensity (zoom)", fontsize=9, color="#e6edf3")
ax.set_xlabel("y' [um]", color="#8b949e", fontsize=8)
ax.set_ylabel("x' [um]", color="#8b949e", fontsize=8)
ax.tick_params(colors="#8b949e", labelsize=6)
ax.set_facecolor("#0d1117")
for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

plt.tight_layout()

def update(i):
    im0.set_data(frames_I[i].T)
    title_obj.set_text(
        f"Single center-pixel sweep:  h = {amplitudes[i]*1e9:.1f} nm"
        f"  ({amplitudes[i]/lam*100:.1f}% of lambda)"
    )
    return im0, title_obj

ani = animation.FuncAnimation(
    fig, update, frames=N_frames, interval=interval_ms, blit=False,
)

out = "output/anim_single_pixel.gif"
ani.save(out, writer="pillow", fps=N_frames//20,
         savefig_kwargs=dict(facecolor="#0d1117"))
plt.close()
print(f"Saved -> {out}")
