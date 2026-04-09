"""
check_center_brightness.py
--------------------------
Compare center pixel brightness on CMOS for single pixel deformations:
0, lambda/4, lambda/2, 3*lambda/4, lambda
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
labels     = ["0", "λ/4", "λ/2", "3λ/4", "λ"]

center_brightness = []
for amp in amplitudes:
    h = torch.zeros(mem_res, mem_res, dtype=torch.float32, device=device)
    c = (mem_res - 1) // 2
    h[c, c] = float(amp)
    with torch.no_grad():
        I = sensor(h)
    cx = cmos_res // 2
    center_brightness.append(float(I[cx, cx].item()))

print("Center pixel brightness:")
for lbl, b in zip(labels, center_brightness):
    print(f"  h={lbl:6s}  I={b:.6f}")

# --- plot ---
plt.rcParams.update(STYLE)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    f"Center pixel brightness vs deformation  (single center pixel)\n"
    f"lambda = {lam*1e9:.1f} nm,  d = {distance*1e3:.0f} mm",
    fontsize=11, color="#e6edf3",
)
fig.patch.set_facecolor("#0d1117")

h_nm = [a * 1e9 for a in amplitudes]

# left: bar chart
ax = axes[0]
ax.set_facecolor("#0d1117")
bars = ax.bar(labels, center_brightness, color="#58a6ff", width=0.5)
for bar, val in zip(bars, center_brightness):
    ax.text(bar.get_x() + bar.get_width()/2, val + max(center_brightness)*0.01,
            f"{val:.4f}", ha="center", va="bottom", fontsize=8, color="#e6edf3")
ax.set_xlabel("deformation h", color="#8b949e", fontsize=9)
ax.set_ylabel("CMOS center pixel intensity [a.u.]", color="#8b949e", fontsize=9)
ax.set_title("Bar chart", fontsize=9, color="#e6edf3")
ax.tick_params(colors="#8b949e", labelsize=8)
for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
ax.grid(axis="y", color="#30363d", linewidth=0.5, linestyle=":")

# right: line plot with continuous sweep
N_cont = 200
amps_cont = np.linspace(0, lam, N_cont)
bright_cont = []
for amp in amps_cont:
    h = torch.zeros(mem_res, mem_res, dtype=torch.float32, device=device)
    c = (mem_res - 1) // 2
    h[c, c] = float(amp)
    with torch.no_grad():
        I = sensor(h)
    cx = cmos_res // 2
    bright_cont.append(float(I[cx, cx].item()))

ax2 = axes[1]
ax2.set_facecolor("#0d1117")
ax2.plot(amps_cont * 1e9, bright_cont, color="#58a6ff", lw=1.5)
for amp, b, lbl in zip(amplitudes, center_brightness, labels):
    ax2.plot(amp * 1e9, b, "o", color="#f78166", ms=8, zorder=5)
    ax2.text(amp * 1e9, b + max(bright_cont)*0.02, lbl,
             ha="center", fontsize=8, color="#f78166")
ax2.set_xlabel("deformation h [nm]", color="#8b949e", fontsize=9)
ax2.set_ylabel("CMOS center pixel intensity [a.u.]", color="#8b949e", fontsize=9)
ax2.set_title("Continuous sweep 0 -> lambda", fontsize=9, color="#e6edf3")
ax2.tick_params(colors="#8b949e", labelsize=8)
for sp in ax2.spines.values(): sp.set_edgecolor("#30363d")
ax2.grid(color="#30363d", linewidth=0.5, linestyle=":")

# lambda/2 주기 표시
for k in [0.5, 1.0]:
    ax2.axvline(lam * k * 1e9, color="#8b949e", lw=0.8, linestyle="--", alpha=0.5)
    ax2.text(lam * k * 1e9, ax2.get_ylim()[0] if ax2.get_ylim()[0] != 0 else 0,
             f"{k}λ", color="#8b949e", fontsize=7, ha="center")

plt.tight_layout()
out = "output/check_center_brightness.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"Saved -> {out}")
