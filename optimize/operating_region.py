"""
operating_region.py
-------------------
2D sweep over (amplitude A, lateral sigma) of a Gaussian bump.
For each (A, sigma) we run the full forward + Adam reconstruction
and record PSNR. The result is a heatmap that shows where the
sensor is *actually* usable, with the theoretical phase-wrap
boundary overlaid.

Theory recap
------------
Reflection phase: phi = -(4 pi / lam) h
Per-pixel phase change must stay below pi (Nyquist):
   |dh/dx| * dx < lam/4
For a Gaussian h(r) = A exp(-r^2/(2 sigma^2)),
   max |dh/dx| = A / (sigma * sqrt(e))
So the per-pixel limit is
   A_max(sigma) = (lam/4) * sigma * sqrt(e) / dx

That curve is overlaid on the heatmap. Above it, multi-cycle
phase wrap occurs -> reconstruction is locally non-unique.

Output:
  output/operating_region/
    operating_region.png
    operating_region.txt
    operating_region.json
"""

import sys, time, json, functools
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

print = functools.partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).parent.parent))
from sensor_model  import HolographicSensor, gaussian_bump
from sensor_params import wavelength, mem_pitch
from sensor_utils  import STYLE

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "output" / "operating_region"
OUT.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

# Smaller config for the 2D sweep (~30 cells)
# Uses the fixed algorithm: h = raw^2 * H_SCALE (positivity) + Adam-3k + cosine decay.
# Matches reconstruction_squared_polish / random_pattern_reconstruction defaults.
MEM   = 128
CMOS  = 256
GRID  = 384
DIST  = 5e-3
LR    = 3e-3
N_ITER = 1500

lam = wavelength
dx  = mem_pitch

# Sweep grid (log-spaced)
A_nm     = np.array([10, 30, 100, 300, 1000, 3000, 10000])
sigma_um = np.array([50, 150, 500, 1500, 5000])

sensor = HolographicSensor(
    wavelength=lam, mem_res=MEM, mem_pitch=dx,
    cmos_res=CMOS, grid_res=GRID, distance=DIST, device=device,
).to(device)

def make_bump(A_m, sig_m):
    coords = (torch.arange(MEM, dtype=torch.float32, device=device) - (MEM-1)*0.5) * dx
    X, Y = torch.meshgrid(coords, coords, indexing="ij")
    return A_m * torch.exp(-(X**2 + Y**2) / (2 * sig_m**2))

def reconstruct(h_true, h_scale):
    # Squared reparam (h >= 0) + Adam + cosine decay — fixed algorithm.
    # raw=0.2 is appropriate for a centred Gaussian bump (mean << peak).
    with torch.no_grad():
        I_tgt = sensor(h_true)
    raw = torch.full((MEM, MEM), 0.2, dtype=torch.float32, device=device,
                     requires_grad=True)
    opt = torch.optim.Adam([raw], lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_ITER)
    for _ in range(N_ITER):
        opt.zero_grad()
        h = (raw * raw) * h_scale
        loss = torch.mean((sensor(h) - I_tgt)**2)
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        h_final = (raw * raw) * h_scale
    return h_final

def psnr_db(a, b):
    mse = float(torch.mean((a - b)**2))
    peak = float(b.abs().max())
    if mse < 1e-30: return 100.0
    if peak < 1e-30: return 0.0
    return 10.0 * np.log10(peak**2 / mse)

# Pre-compute theoretical wrap boundary
def A_max_no_wrap(sig_m):
    return (lam / 4.0) * sig_m * np.sqrt(np.e) / dx

PSNR  = np.full((len(A_nm), len(sigma_um)), np.nan)
GRAD  = np.full_like(PSNR, np.nan)  # max |dh/dx| in lam/4 units (per pixel)
print(f"\n{'A[nm]':>8} {'sig[um]':>8} {'grad/(lam/4)':>14} {'PSNR':>7} {'time[s]':>7}")
for i, A in enumerate(A_nm):
    for j, sg in enumerate(sigma_um):
        A_m  = float(A)  * 1e-9
        sg_m = float(sg) * 1e-6
        # auto h_scale: 2x amplitude (or floor)
        h_scale = max(2.0 * A_m, 100e-9)
        max_grad_per_pix = (A_m / (sg_m * np.sqrt(np.e))) * dx
        ratio = max_grad_per_pix / (lam / 4.0)
        GRAD[i, j] = ratio
        t0 = time.time()
        h_true = make_bump(A_m, sg_m)
        h_rec  = reconstruct(h_true, h_scale)
        p = psnr_db(h_rec, h_true)
        PSNR[i, j] = p
        dt = time.time() - t0
        print(f"{A:>8.0f} {sg:>8.0f} {ratio:>14.3f} {p:>7.2f} {dt:>7.1f}")

# -----------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------
plt.rcParams.update(STYLE)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Operating region — (amplitude, lateral sigma) sweep",
             fontsize=13, color="#e6edf3")
fig.patch.set_facecolor("#0d1117")

def _style(ax):
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e", labelsize=8)
    ax.grid(color="#30363d", lw=0.4, ls=":")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

# (A) PSNR heatmap
ax = axes[0]
im = ax.imshow(PSNR, origin="lower", aspect="auto", cmap="viridis",
               vmin=0, vmax=55,
               extent=[np.log10(sigma_um[0]), np.log10(sigma_um[-1]),
                       np.log10(A_nm[0]),    np.log10(A_nm[-1])])
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("PSNR [dB]", color="#8b949e", fontsize=8)
plt.setp(cb.ax.yaxis.get_ticklabels(), color="#8b949e")
# overlay theoretical phase-wrap boundary  A_max(sigma)
sig_dense_um = np.logspace(np.log10(sigma_um[0]), np.log10(sigma_um[-1]), 200)
A_lim_nm = np.array([A_max_no_wrap(s*1e-6) for s in sig_dense_um]) * 1e9
A_lim_nm = np.clip(A_lim_nm, A_nm[0], A_nm[-1])
ax.plot(np.log10(sig_dense_um), np.log10(A_lim_nm),
        color="#f85149", lw=1.6, ls="--",
        label="phase-wrap limit  A = (lam/4) sigma sqrt(e)/dx")
ax.set_xlabel("log10 sigma [um]", fontsize=9, color="#8b949e")
ax.set_ylabel("log10 amplitude [nm]", fontsize=9, color="#8b949e")
ax.set_title("(A) Reconstruction PSNR map\nred dashed = theoretical wrap onset",
             fontsize=10, color="#e6edf3")
ax.legend(fontsize=8, labelcolor="#e6edf3", framealpha=0.3)
# annotate cell values
for i, A in enumerate(A_nm):
    for j, sg in enumerate(sigma_um):
        ax.text(np.log10(sg), np.log10(A), f"{PSNR[i,j]:.0f}",
                ha="center", va="center", fontsize=7,
                color=("white" if PSNR[i,j] < 30 else "black"))
_style(ax)

# (B) Per-pixel gradient ratio (theoretical)
ax = axes[1]
im = ax.imshow(np.log10(GRAD), origin="lower", aspect="auto", cmap="RdYlGn_r",
               vmin=-3, vmax=2,
               extent=[np.log10(sigma_um[0]), np.log10(sigma_um[-1]),
                       np.log10(A_nm[0]),    np.log10(A_nm[-1])])
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("log10 (max dh/pixel) / (lam/4)", color="#8b949e", fontsize=8)
plt.setp(cb.ax.yaxis.get_ticklabels(), color="#8b949e")
ax.contour(np.log10(sigma_um), np.log10(A_nm), GRAD, levels=[1.0],
           colors="#0d1117", linewidths=1.5)
ax.set_xlabel("log10 sigma [um]", fontsize=9, color="#8b949e")
ax.set_ylabel("log10 amplitude [nm]", fontsize=9, color="#8b949e")
ax.set_title("(B) Per-pixel phase wrap (red=many cycles)\nblack = wrap-onset contour",
             fontsize=10, color="#e6edf3")
for i, A in enumerate(A_nm):
    for j, sg in enumerate(sigma_um):
        ax.text(np.log10(sg), np.log10(A), f"{GRAD[i,j]:.2g}",
                ha="center", va="center", fontsize=7, color="black")
_style(ax)

plt.tight_layout()
plt.savefig(OUT / "operating_region.png", dpi=150, bbox_inches="tight",
            facecolor="#0d1117")
plt.close()

# -----------------------------------------------------------------------
# Text + JSON
# -----------------------------------------------------------------------
lines = [
    "operating_region.py — (A, sigma) sweep",
    "=======================================",
    f"  wavelength    = {lam*1e9:.1f} nm",
    f"  membrane pitch= {dx*1e6:.1f} um",
    f"  distance      = {DIST*1e3:.1f} mm",
    f"  lr={LR}, n_iter={N_ITER}",
    "",
    "Per-pixel phase-wrap onset:",
    "  A_max(sigma) = (lam/4) * sigma * sqrt(e) / dx",
    f"             = {(lam/4)*np.sqrt(np.e)/dx*1e3:.4f} nm per (1 um sigma)",
    "",
    "Result table (PSNR [dB]; row=A[nm], col=sigma[um])",
    "  " + " ".join(f"{int(s):>8}" for s in sigma_um),
]
for i, A in enumerate(A_nm):
    lines.append(f"A={int(A):>5}  " + " ".join(f"{PSNR[i,j]:>7.2f}" for j in range(len(sigma_um))))
lines.append("")
lines.append("Per-pixel gradient ratio  (max |dh/pixel|) / (lam/4):")
lines.append("  " + " ".join(f"{int(s):>8}" for s in sigma_um))
for i, A in enumerate(A_nm):
    lines.append(f"A={int(A):>5}  " + " ".join(f"{GRAD[i,j]:>8.2g}" for j in range(len(sigma_um))))
lines.append("")
lines.append("Reading the maps")
lines.append("  * Cells with ratio < 1 (green) are pre-wrap; reconstruction")
lines.append("    should be high quality.")
lines.append("  * Cells with ratio >> 1 (red) suffer multi-cycle wrap;")
lines.append("    Adam reconstruction will mis-estimate amplitude.")
lines.append("  * For mm-scale deformation (A >= 1000 nm), only sigma >= 1.5 mm")
lines.append("    keeps the per-pixel slope below the wrap limit.")
txt = "\n".join(lines)
(OUT / "operating_region.txt").write_text(txt, encoding="utf-8")
print("\n" + txt)

(OUT / "operating_region.json").write_text(json.dumps(dict(
    A_nm=A_nm.tolist(),
    sigma_um=sigma_um.tolist(),
    psnr_db=PSNR.tolist(),
    grad_ratio=GRAD.tolist(),
    config=dict(wavelength_nm=lam*1e9, mem_pitch_um=dx*1e6, mem=MEM,
                cmos=CMOS, grid=GRID, distance_mm=DIST*1e3,
                lr=LR, n_iter=N_ITER),
), indent=2))

print(f"\nSaved -> {OUT/'operating_region.png'}")
