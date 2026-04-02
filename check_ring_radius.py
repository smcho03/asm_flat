"""
check_ring_radius.py
--------------------
Sanity check: Fresnel ring radius scales as sqrt(lambda * d).

For a single center-pixel perturbation, the first bright ring radius
should follow:  r_1 = sqrt(lambda * d)

Distances tested: 1mm, 2mm, 3mm, 5mm, 8mm, 12mm

Output:
  output/check_ring_radius.png
    top row   : CMOS intensity (zoom) for each distance
    bottom row: measured r_1 vs sqrt(lambda*d) -> should be linear
"""

import os
import numpy as np
import torch, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from sensor_model import HolographicSensor
from sensor_params import wavelength, mem_res, mem_pitch, cmos_res, grid_res
from sensor_utils import STYLE

os.makedirs("output", exist_ok=True)

device   = "cpu"
lam      = wavelength
distances = [1e-3, 2e-3, 3e-3, 5e-3, 8e-3, 12e-3]  # [m]

cmos_coords = (np.arange(cmos_res) - (cmos_res - 1) * 0.5) * mem_pitch
c_um = cmos_coords * 1e6

# single center pixel, amplitude = lambda/4 (pi/2 phase -> good contrast)
amp = lam / 4

print("Computing CMOS patterns ...", flush=True)
results = []
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

    # measure first ring radius from radial profile
    # use centre row of I
    centre_row = I[cmos_res // 2, :]
    centre_val = float(centre_row[cmos_res // 2])

    # find first local max away from centre (skip central 5px)
    skip = 5
    half = cmos_res // 2
    row_half = centre_row[half + skip:]   # right half, skipping centre
    # first local maximum
    r_px = None
    for k in range(1, len(row_half) - 1):
        if row_half[k] > row_half[k-1] and row_half[k] > row_half[k+1]:
            r_px = k + skip   # pixel offset from centre
            break

    r_um = float(r_px * mem_pitch * 1e6) if r_px is not None else float("nan")
    r_theory_um = np.sqrt(lam * d) * 1e6

    results.append({
        "d"          : d,
        "I"          : I,
        "r_meas_um"  : r_um,
        "r_theory_um": r_theory_um,
    })
    print(f"  d={d*1e3:.0f}mm  r_meas={r_um:.1f}um  r_theory={r_theory_um:.1f}um", flush=True)

# ---- figure ----
plt.rcParams.update(STYLE)
fig = plt.figure(figsize=(20, 10))
fig.suptitle(
    f"Fresnel ring radius sanity check  (single center pixel, h=lambda/4)\n"
    f"expected: r_1 = sqrt(lambda * d)",
    fontsize=12, color="#e6edf3",
)
fig.patch.set_facecolor("#0d1117")

I_max = max(r["I"].max() for r in results)

# 모든 패널 동일 zoom: 가장 먼 거리의 이론 반경 기준
zoom_um = max(results[-1]["r_theory_um"] * 5, 200)
mask = np.abs(c_um) <= zoom_um
idx  = np.where(mask)[0]
sz, ez = idx[0], idx[-1] + 1
ext_z = [c_um[sz], c_um[ez-1], c_um[sz], c_um[ez-1]]

for col, r in enumerate(results):
    ax = fig.add_subplot(2, len(distances), col + 1)
    I_z = r["I"][sz:ez, sz:ez]

    im = ax.imshow(I_z.T, extent=ext_z, origin="lower", aspect="equal",
                   cmap="inferno", vmin=0, vmax=I_max)
    ax.set_title(f"d = {r['d']*1e3:.0f} mm", fontsize=9, color="#e6edf3")
    ax.set_xlabel("y' [um]", color="#8b949e", fontsize=7)
    ax.set_ylabel("x' [um]", color="#8b949e", fontsize=7)
    ax.tick_params(colors="#8b949e", labelsize=6)
    ax.set_facecolor("#0d1117")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

    # mark measured and theoretical ring radius
    theta = np.linspace(0, 2*np.pi, 300)
    if not np.isnan(r["r_meas_um"]):
        ax.plot(r["r_meas_um"] * np.cos(theta),
                r["r_meas_um"] * np.sin(theta),
                "c--", lw=1.2, label=f"meas {r['r_meas_um']:.0f}um")
    ax.plot(r["r_theory_um"] * np.cos(theta),
            r["r_theory_um"] * np.sin(theta),
            "r:",  lw=1.2, label=f"theory {r['r_theory_um']:.0f}um")
    ax.legend(fontsize=6, loc="upper right", labelcolor="#e6edf3", framealpha=0.3)

# bottom: r vs sqrt(lambda*d)
ax_bot = fig.add_subplot(2, 1, 2)
ax_bot.set_facecolor("#0d1117")
for sp in ax_bot.spines.values(): sp.set_edgecolor("#30363d")
ax_bot.tick_params(colors="#8b949e", labelsize=8)
ax_bot.set_xlabel("sqrt(lambda * d)  [um]", color="#8b949e", fontsize=9)
ax_bot.set_ylabel("ring radius  [um]",      color="#8b949e", fontsize=9)
ax_bot.set_title("Measured r_1 vs theory  (slope should = 1)", fontsize=10, color="#e6edf3")
ax_bot.grid(color="#30363d", linewidth=0.5, linestyle=":")

d_arr      = np.array([r["d"] for r in results])
r_meas     = np.array([r["r_meas_um"]   for r in results])
r_theory   = np.array([r["r_theory_um"] for r in results])
x_theory   = np.sqrt(lam * d_arr) * 1e6

ax_bot.plot(x_theory, r_theory, "w--", lw=1.5, label="theory: r = sqrt(lambda*d)")
ax_bot.plot(x_theory, r_meas,   "o",   color="#58a6ff", ms=8, label="measured r_1")

# linear fit
valid = ~np.isnan(r_meas)
if valid.sum() >= 2:
    slope = np.polyfit(x_theory[valid], r_meas[valid], 1)
    x_fit = np.linspace(x_theory.min(), x_theory.max(), 100)
    ax_bot.plot(x_fit, np.polyval(slope, x_fit), "-",
                color="#f78166", lw=1.5, label=f"linear fit  slope={slope[0]:.3f}")

ax_bot.legend(fontsize=8, labelcolor="#e6edf3", framealpha=0.3)

plt.tight_layout()
out = "output/check_ring_radius.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"Saved -> {out}")
