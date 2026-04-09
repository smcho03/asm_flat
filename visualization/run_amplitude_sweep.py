"""
run_amplitude_sweep.py
----------------------
Gaussian bump at 5 amplitudes: 0, lambda/4, lambda/2, 3*lambda/4, lambda

Output: output/amplitude_sweep.png
  3 rows x 5 cols
    rows : h(x,y)  |  CMOS Intensity  |  CMOS Phase (unwrapped)
    cols : 0  |  lambda/4  |  lambda/2  |  3*lambda/4  |  lambda
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.restoration import unwrap_phase

sys.path.insert(0, str(Path(__file__).parent.parent))
from sensor_model  import HolographicSensor, gaussian_bump
from sensor_params import wavelength, mem_res, mem_pitch, cmos_res, grid_res, distance
from sensor_utils  import STYLE

os.makedirs("output", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

sensor = HolographicSensor(
    wavelength=wavelength, mem_res=mem_res, mem_pitch=mem_pitch,
    cmos_res=cmos_res, grid_res=grid_res, distance=distance, device=device,
).to(device)

lam = wavelength
amplitudes = [0, lam/4, lam/2, lam*3/4, lam]
labels     = ["h = 0", "h = λ/4", "h = λ/2", "h = 3λ/4", "h = λ"]

# coordinate axes
mem_coords  = (np.arange(mem_res)  - (mem_res  - 1) * 0.5) * mem_pitch
cmos_coords = (np.arange(cmos_res) - (cmos_res - 1) * 0.5) * mem_pitch

plt.rcParams.update(STYLE)
fig, axes = plt.subplots(3, 5, figsize=(22, 12))
fig.suptitle(
    f"Amplitude Sweep: 0 → λ  (λ={lam*1e9:.1f} nm, sigma=100 μm)",
    fontsize=12, color="#e6edf3",
)
fig.patch.set_facecolor("#0d1117")

for col, (amp, label) in enumerate(zip(amplitudes, labels)):
    print(f"  computing {label} ({amp*1e9:.1f} nm) ...", flush=True)

    h = gaussian_bump(N=mem_res, dx=mem_pitch, amplitude=amp, sigma=100e-6, device=device)

    with torch.no_grad():
        I_cmos = sensor(h)
        Ud     = sensor.propagated_field(h)
        Ud_c   = sensor.crop(Ud)

    h_nm    = h.cpu().numpy() * 1e9
    I_np    = I_cmos.cpu().numpy()
    phi_raw = np.angle(Ud_c.cpu().numpy()).astype(np.float64)
    phi_uw  = unwrap_phase(phi_raw).astype(np.float32)

    c_um   = cmos_coords * 1e6
    m_um   = mem_coords  * 1e6
    ext_c  = [c_um.min(), c_um.max(), c_um.min(), c_um.max()]
    ext_m  = [m_um.min(), m_um.max(), m_um.min(), m_um.max()]

    # --- row 0: h map ---
    ax = axes[0, col]
    vmax_h = max(float(h_nm.max()), 1.0)
    im0 = ax.imshow(h_nm.T, extent=ext_m, origin="lower", aspect="auto",
                    cmap="viridis", vmin=0, vmax=vmax_h)
    fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04).set_label("h [nm]", color="#8b949e", fontsize=7)
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("y [μm]"); ax.set_ylabel("x [μm]")

    # --- row 1: CMOS intensity ---
    ax = axes[1, col]
    im1 = ax.imshow(I_np.T, extent=ext_c, origin="lower", aspect="auto",
                    cmap="inferno", vmin=0)
    fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04).set_label("I [a.u.]", color="#8b949e", fontsize=7)
    ax.set_title(f"CMOS Intensity  ({amp*1e9:.0f} nm)", fontsize=8)
    ax.set_xlabel("y' [μm]"); ax.set_ylabel("x' [μm]")

    # --- row 2: CMOS phase (unwrapped) ---
    ax = axes[2, col]
    span   = float(phi_uw.max() - phi_uw.min())
    if span <= 2 * np.pi + 1e-3:
        cmap_p, vmin_p, vmax_p = "hsv", -np.pi, np.pi
    else:
        cmap_p = "viridis"
        vmin_p, vmax_p = float(phi_uw.min()), float(phi_uw.max())
    im2 = ax.imshow(phi_uw.T, extent=ext_c, origin="lower", aspect="auto",
                    cmap=cmap_p, vmin=vmin_p, vmax=vmax_p)
    fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04).set_label("φ [rad]", color="#8b949e", fontsize=7)
    ax.set_title("CMOS Phase (unwrapped)", fontsize=8)
    ax.set_xlabel("y' [μm]"); ax.set_ylabel("x' [μm]")

# style all axes
for ax in axes.flat:
    ax.tick_params(colors="#8b949e", labelsize=6)
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")
    ax.set_facecolor("#0d1117")

plt.tight_layout()
out = "output/amplitude_sweep.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"\nSaved -> {out}")
