"""
aliasing_analysis.py
--------------------
Identify and visualize all aliasing sources in the holographic simulation.

Sources considered
  (1) Membrane sampling (Nyquist on the height map itself)
  (2) Phase-wrap Nyquist (|dh/pixel| < lam/4 for reflection)
  (3) ASM propagation aliasing (Matsushima & Shimobaba 2009)
  (4) FFT circular wrap (finite grid_res, zero padding)
  (5) Amplitude phase-wrap regime on intensity

Output:
  output/aliasing_analysis/
    aliasing_summary.png
    aliasing_metrics.txt
    aliasing_metrics.json
"""

import sys, json
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from sensor_model  import HolographicSensor, gaussian_bump
from sensor_params import wavelength, mem_pitch
from sensor_utils  import STYLE

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "output" / "aliasing_analysis"
OUT.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

lam = wavelength
dx  = mem_pitch

# Analytic limits
f_nyq            = 1.0 / (2.0 * dx)
dh_wrap_nm       = lam * 0.25 * 1e9
max_slope_ratio  = lam / (8.0 * np.pi * dx)

def asm_bandlimit(L, d):
    sin_max = (L / 2.0) / np.sqrt(d**2 + (L / 2.0)**2)
    return sin_max / lam

# Numerical: grid_res sweep for FFT wrap-around
MEM   = 256
CMOS  = 512
DIST  = 5e-3
AMP   = 300e-9
SIGMA = 150e-6

grid_variants = [CMOS, int(CMOS*1.2), int(CMOS*1.5), int(CMOS*2.0), int(CMOS*3.0)]
intensity_list = []
for gr in grid_variants:
    sensor = HolographicSensor(
        wavelength=lam, mem_res=MEM, mem_pitch=dx,
        cmos_res=CMOS, grid_res=gr, distance=DIST, device=device,
    ).to(device)
    h = gaussian_bump(N=MEM, dx=dx, amplitude=AMP, sigma=SIGMA, device=device)
    with torch.no_grad():
        I = sensor(h).cpu().numpy()
    intensity_list.append(I)
    print(f"  grid_res={gr}  I sum={I.sum():.3e}")

I_ref = intensity_list[-1]
grid_rms = [float(np.sqrt(np.mean((I - I_ref)**2))) for I in intensity_list]

# Numerical: amplitude / phase-wrap
sensor_ref = HolographicSensor(
    wavelength=lam, mem_res=MEM, mem_pitch=dx,
    cmos_res=CMOS, grid_res=CMOS*2, distance=DIST, device=device,
).to(device)

amps_nm = np.array([10, 30, 79, 158, 237, 316, 500, 1000])
amps_m  = amps_nm * 1e-9
I_std_ratio = []
for A in amps_m:
    h_full = gaussian_bump(N=MEM, dx=dx, amplitude=float(A),     sigma=SIGMA, device=device)
    h_half = gaussian_bump(N=MEM, dx=dx, amplitude=float(A*0.5), sigma=SIGMA, device=device)
    with torch.no_grad():
        I_full = sensor_ref(h_full).cpu().numpy()
        I_half = sensor_ref(h_half).cpu().numpy()
    s_full = np.std(I_full - 1.0)  # fluctuation about flat baseline
    s_half = np.std(I_half - 1.0)
    I_std_ratio.append(float(s_full / max(s_half, 1e-30)))

# -----------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------
plt.rcParams.update(STYLE)
fig, axes = plt.subplots(2, 3, figsize=(20, 11))
fig.suptitle("Aliasing sources in the holographic simulation",
             fontsize=13, color="#e6edf3")
fig.patch.set_facecolor("#0d1117")

def _style(ax):
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e", labelsize=7)
    ax.grid(color="#30363d", lw=0.4, ls=":")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

# (1) Membrane sampling
ax = axes[0,0]
period_um = 2 * dx * 1e6
ax.axvline(1.0, color="#f85149", ls="--", lw=1.0,
           label=f"Nyquist period = 2 dx = {period_um:.0f} um")
ax.fill_betweenx([0, 1], 0, 1, color="#f85149", alpha=0.15, label="aliased")
ax.fill_betweenx([0, 1], 1, 5, color="#3fb950", alpha=0.10, label="resolved")
ax.set_xlabel("feature period / (2 dx)", fontsize=9, color="#8b949e")
ax.set_ylabel("—", fontsize=9, color="#8b949e")
ax.set_title(f"(1) Membrane Nyquist  (dx={dx*1e6:.0f} um)",
             fontsize=10, color="#e6edf3")
ax.set_xlim(0, 5); ax.set_ylim(0, 1)
ax.legend(fontsize=7, labelcolor="#e6edf3", framealpha=0.3)
_style(ax)

# (2) Phase-wrap
ax = axes[0,1]
ax.axhline(dh_wrap_nm, color="#f85149", ls="--", lw=1.0,
           label=f"|dh/pixel| < lam/4 = {dh_wrap_nm:.1f} nm")
for A_nm in [50, 158, 400]:
    hprof = A_nm * np.exp(-((np.arange(-50,50))**2)/(2*(15)**2))
    ax.plot(np.abs(np.diff(hprof)), lw=1.0, label=f"Gaussian A={A_nm} nm")
ax.set_xlabel("pixel index along profile", fontsize=9, color="#8b949e")
ax.set_ylabel("|dh| per pixel [nm]", fontsize=9, color="#8b949e")
ax.set_title("(2) Phase-wrap Nyquist (reflection)", fontsize=10, color="#e6edf3")
ax.set_ylim(0, dh_wrap_nm*3)
ax.legend(fontsize=7, labelcolor="#e6edf3", framealpha=0.3)
_style(ax)

# (3) ASM band-limit
ax = axes[0,2]
ds_mm = np.linspace(0.5, 30, 200)
Ls = [512*dx, 1024*dx, 1536*dx, 2048*dx]
for L in Ls:
    fmax = asm_bandlimit(L, ds_mm*1e-3) / 1e3
    ax.plot(ds_mm, fmax, lw=1.2, label=f"L = {L*1e3:.2f} mm")
ax.axhline(f_nyq/1e3, color="#f85149", ls="--", lw=1.0,
           label=f"grid Nyquist = {f_nyq/1e3:.0f} cyc/mm")
ax.set_xlabel("distance d [mm]", fontsize=9, color="#8b949e")
ax.set_ylabel("ASM band-limit [cyc/mm]", fontsize=9, color="#8b949e")
ax.set_title("(3) ASM propagation aliasing limit", fontsize=10, color="#e6edf3")
ax.legend(fontsize=7, labelcolor="#e6edf3", framealpha=0.3)
_style(ax)

# (4) FFT wrap-around
ax = axes[1,0]
ax.plot(grid_variants, grid_rms, "o-", color="#58a6ff", lw=1.5, markersize=6)
ax.set_xlabel("grid_res (zero-padded)", fontsize=9, color="#8b949e")
ax.set_ylabel("I RMS vs largest grid", fontsize=9, color="#8b949e")
ax.set_title(f"(4) FFT wrap-around residual  (cmos={CMOS})",
             fontsize=10, color="#e6edf3")
ax.set_yscale("log")
_style(ax)

# (5) Amplitude phase-wrap on intensity
ax = axes[1,1]
ax.plot(amps_nm, I_std_ratio, "o-", color="#f78166", lw=1.5, markersize=6)
ax.axvline(lam*0.25*1e9, color="#f85149", ls="--", lw=0.8,
           label=f"lam/4 = {lam*0.25*1e9:.1f} nm")
ax.axhline(4.0, color="#8b949e", ls=":", lw=0.6, label="ideal quadratic = 4")
ax.set_xlabel("peak amplitude A [nm]", fontsize=9, color="#8b949e")
ax.set_ylabel("std(I) / std(I at A/2)", fontsize=9, color="#8b949e")
ax.set_title("(5) Phase-wrap signature in intensity", fontsize=10, color="#e6edf3")
ax.set_xscale("log")
ax.legend(fontsize=7, labelcolor="#e6edf3", framealpha=0.3)
_style(ax)

# Summary
ax = axes[1,2]
ax.axis("off")
summary = (
    f"Summary of aliasing limits\n"
    f"===========================\n"
    f"Wavelength          lam  = {lam*1e9:.2f} nm\n"
    f"Membrane pitch      dx   = {dx*1e6:.2f} um\n"
    f"Nyquist (spatial)   1/2dx= {f_nyq/1e3:.1f} cyc/mm\n"
    f"Min resolvable feat 2 dx = {2*dx*1e6:.1f} um\n"
    f"\n"
    f"Phase-wrap limit (reflection)\n"
    f"  |dh/pixel| < lam/4 = {lam*0.25*1e9:.2f} nm\n"
    f"  (A/T < {max_slope_ratio:.2e})\n"
    f"\n"
    f"ASM band-limit (d=5 mm, L=15.36 mm)\n"
    f"  fmax = {asm_bandlimit(1536*dx, 5e-3)/1e3:.1f} cyc/mm\n"
    f"\n"
    f"FFT wrap-around residual vs x3 grid\n"
)
for gr, rr in zip(grid_variants, grid_rms):
    summary += f"  grid={gr:4d}  RMS={rr:.3e}\n"
ax.text(0.02, 0.98, summary, transform=ax.transAxes, va="top", ha="left",
        fontsize=8, color="#e6edf3", family="monospace")

plt.tight_layout()
plt.savefig(OUT / "aliasing_summary.png", dpi=150, bbox_inches="tight",
            facecolor="#0d1117")
plt.close()

(OUT / "aliasing_metrics.txt").write_text(summary, encoding="utf-8")
(OUT / "aliasing_metrics.json").write_text(json.dumps(dict(
    wavelength_nm=lam*1e9, dx_um=dx*1e6,
    nyquist_cyc_per_mm=f_nyq/1e3, min_resolv_um=2*dx*1e6,
    phase_wrap_dh_max_nm=lam*0.25*1e9,
    grid_wrap_rms={str(gr): rr for gr, rr in zip(grid_variants, grid_rms)},
    amp_vs_std_ratio={f"A={a}nm": r for a, r in zip(amps_nm.tolist(), I_std_ratio)},
), indent=2))

print(f"Saved -> {OUT/'aliasing_summary.png'}")
print(summary)
