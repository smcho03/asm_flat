"""
Case-wise Visualization for Holographic Tactile Sensor
Dark-background style with 2-D phase unwrapping.

3 x 3 layout per case
  Cols: h(x,y)  |  CMOS Intensity  |  CMOS Phase (unwrapped)
  Rows: Flat (ref)  |  Deformed  |  Difference (def - ref)

Cases
-----
  0. Flat            - zero deformation (baseline sanity check)
  1. Single pixel    - one pixel at centre (reveals PSF / Fresnel rings)
  2. Single bump     - one Gaussian bump (sigma=100 um, h=500 nm)
  3. Multi bump      - five bumps at various positions/sizes
  4. Random          - band-limited random membrane roughness
"""

from __future__ import annotations

import os
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, TwoSlopeNorm, Normalize
from skimage.restoration import unwrap_phase as skimage_unwrap

from holographic_tactile_sensor import HolographicSensor, gaussian_bump

# ── Global style ──────────────────────────────────────────────────────
plt.style.use("dark_background")
_DARK   = "#0a0a0a"
_GRID   = "#2a2a2a"
_WHITE  = "#e8e8e8"


# ======================================================================
#  Deformation generators
# ======================================================================

def make_h_flat(N: int, dx: float, device: str) -> torch.Tensor:
    return torch.zeros(N, N, dtype=torch.float32, device=device)


def make_h_single_pixel(
    N: int, dx: float, device: str,
    amplitude: float = 158e-9,      # lambda/4 -> pi phase shift
) -> torch.Tensor:
    """Single-pixel phase perturbation at membrane centre."""
    h = torch.zeros(N, N, dtype=torch.float32, device=device)
    c = (N - 1) // 2
    h[c, c] = amplitude
    return h


def make_h_single_bump(N: int, dx: float, device: str) -> torch.Tensor:
    return gaussian_bump(N=N, dx=dx, amplitude=500e-9, sigma=100e-6, device=device)


def make_h_multi_bump(N: int, dx: float, device: str) -> torch.Tensor:
    specs = [
        dict(amplitude=500e-9, sigma= 80e-6, center=(     0.0,      0.0)),
        dict(amplitude=400e-9, sigma=120e-6, center=( 800e-6,   300e-6)),
        dict(amplitude=300e-9, sigma= 60e-6, center=(-600e-6,  -700e-6)),
        dict(amplitude=450e-9, sigma= 90e-6, center=( 200e-6,  -900e-6)),
        dict(amplitude=350e-9, sigma=150e-6, center=(-900e-6,   600e-6)),
    ]
    h = torch.zeros(N, N, dtype=torch.float32, device=device)
    for s in specs:
        h = h + gaussian_bump(N=N, dx=dx, device=device, **s)
    return h


def make_h_random(
    N: int, dx: float, device: str,
    seed: int = 42, amplitude: float = 400e-9, sigma_filter: float = 200e-6,
) -> torch.Tensor:
    torch.manual_seed(seed)
    noise  = torch.randn(N, N, dtype=torch.float32, device=device)
    f      = torch.fft.fftfreq(N, d=dx)
    Fx, Fy = torch.meshgrid(f, f, indexing="ij")
    lpf    = torch.exp(-2.0 * (np.pi * sigma_filter)**2 * (Fx**2 + Fy**2))
    nc     = torch.view_as_complex(torch.stack([noise, torch.zeros_like(noise)], -1))
    fil    = torch.fft.ifft2(torch.fft.fft2(nc) * lpf).real
    return (fil / fil.abs().max().clamp(min=1e-30)) * amplitude


# ======================================================================
#  Physics computation
# ======================================================================

def _unwrap(phase_np: np.ndarray) -> np.ndarray:
    """2-D phase unwrapping via skimage Goldstein algorithm."""
    return skimage_unwrap(phase_np.astype(np.float64)).astype(np.float32)


def compute_case(
    sensor: HolographicSensor,
    h_def : torch.Tensor,
    device: str,
) -> dict:
    """
    Returns dict with keys 'h', 'I', 'phase', each a 3-tuple:
        (flat_array, deformed_array, difference_array)

    Heights are in nm.
    Phases are UNWRAPPED (continuous, not wrapped to [-pi, pi]).
    Phase difference uses the wrapped complex-product method to avoid
    discontinuities, then is unwrapped for display.
    """
    N  = sensor.mem_res
    dx = sensor.mem_pitch

    h_flat = torch.zeros(N, N, dtype=torch.float32, device=device)

    with torch.no_grad():
        Ud_flat_full = sensor.propagated_field(h_flat)
        Ud_def_full  = sensor.propagated_field(h_def)

    Ud_flat = sensor.crop(Ud_flat_full)
    Ud_def  = sensor.crop(Ud_def_full)

    # ---- Intensities ----
    I_flat = torch.abs(Ud_flat).pow(2).cpu().numpy().astype(np.float32)
    I_def  = torch.abs(Ud_def ).pow(2).cpu().numpy().astype(np.float32)
    I_diff = I_def - I_flat

    # ---- Phases (unwrapped) ----
    phase_flat_raw = torch.angle(Ud_flat).cpu().numpy().astype(np.float32)
    phase_def_raw  = torch.angle(Ud_def ).cpu().numpy().astype(np.float32)

    phase_flat = _unwrap(phase_flat_raw)
    phase_def  = _unwrap(phase_def_raw)

    # Phase difference: use complex product (respects wrapping), then unwrap
    phase_diff_raw = torch.angle(Ud_def * Ud_flat.conj()).cpu().numpy().astype(np.float32)
    phase_diff     = _unwrap(phase_diff_raw)

    # ---- Heights [nm] ----
    h_flat_nm = h_flat.cpu().numpy() * 1e9          # all zeros
    h_def_nm  = h_def .cpu().numpy() * 1e9
    h_diff_nm = h_def_nm - h_flat_nm                # = h_def_nm

    return dict(
        h     = (h_flat_nm,  h_def_nm,  h_diff_nm),
        I     = (I_flat,     I_def,     I_diff),
        phase = (phase_flat, phase_def, phase_diff),
    )


# ======================================================================
#  Zoom helpers
# ======================================================================

def _zoom_slice(M: int, half_px: int) -> tuple[int, int]:
    ctr = (M - 1) // 2
    s   = max(ctr - half_px, 0)
    e   = min(ctr + half_px + 1, M)
    return s, e


def _zoom_h_slice(N: int, half_px: int) -> tuple[int, int]:
    return _zoom_slice(N, half_px)


# ======================================================================
#  Plotting
# ======================================================================

def _sym_vmax(arr: np.ndarray, percentile: float = 99.5) -> float:
    v = float(np.percentile(np.abs(arr), percentile))
    return max(v, 1e-12)


def _add_colorbar(fig, im, ax, label: str = "") -> None:
    cb = fig.colorbar(im, ax=ax, pad=0.03, shrink=0.92, aspect=18)
    cb.ax.tick_params(labelsize=7, color=_WHITE, labelcolor=_WHITE)
    cb.set_label(label, fontsize=7, color=_WHITE)
    cb.outline.set_edgecolor(_GRID)


def plot_case(
    data       : dict,
    sensor     : HolographicSensor,
    case_name  : str,
    save_path  : str,
    zoom_cmos_um : float = 600.0,   # half-width of CMOS zoom [um]
    zoom_h_um    : float = 600.0,   # half-width of h zoom [um]  (0 = full)
) -> None:
    """
    Render and save a 3x3 dark-theme figure.
    """
    dx = sensor.mem_pitch
    N  = sensor.mem_res
    M  = sensor.cmos_res

    # ---- Zoom windows ----
    z_cmos_px = min(int(zoom_cmos_um / (dx * 1e6)), M // 2)
    z_h_px    = (int(zoom_h_um / (dx * 1e6)) if zoom_h_um > 0 else N // 2)
    z_h_px    = min(z_h_px, N // 2)

    sc_c, ec_c = _zoom_slice(M, z_cmos_px)
    sc_h, ec_h = _zoom_h_slice(N, z_h_px)

    # Physical axes [um]
    ax_full_h = (np.arange(N) - (N - 1) * 0.5) * dx * 1e6
    ax_full_c = (np.arange(M) - (M - 1) * 0.5) * dx * 1e6

    ax_h  = ax_full_h[sc_h:ec_h]
    ax_c  = ax_full_c[sc_c:ec_c]

    ext_h = [ax_h[0],  ax_h[-1],  ax_h[-1],  ax_h[0]]
    ext_c = [ax_c[0],  ax_c[-1],  ax_c[-1],  ax_c[0]]

    row_labels = ["h = 0  (reference)", "h  (deformed)", r"$\Delta$  (def $-$ ref)"]
    col_labels = [
        "h(x,y)   [nm]",
        "CMOS Intensity   I  [a.u.]",
        "CMOS Phase   $\\phi$  (unwrapped)  [rad]",
    ]

    # ---- Figure ----
    fig, axes = plt.subplots(
        3, 3, figsize=(16, 13),
        constrained_layout=True,
        facecolor=_DARK,
    )
    fig.suptitle(
        f"Holographic Tactile Sensor   |   {case_name}",
        fontsize=13, fontweight="bold", color=_WHITE,
    )

    for row_idx, row_label in enumerate(row_labels):
        for col_idx, col_label in enumerate(col_labels):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor(_DARK)

            flat_arr, def_arr, diff_arr = data[["h", "I", "phase"][col_idx]]

            # Select the right array and crop
            arr = [flat_arr, def_arr, diff_arr][row_idx]

            if col_idx == 0:   # height map — crop membrane
                arr_plot = arr[sc_h:ec_h, sc_h:ec_h]
                extent   = ext_h
                xlabel, ylabel = "y  [µm]", "x  [µm]"
            else:              # I or phase — crop CMOS
                arr_plot = arr[sc_c:ec_c, sc_c:ec_c]
                extent   = ext_c
                xlabel, ylabel = "y'  [µm]", "x'  [µm]"

            # ---- Colormap / normalization ----
            if row_idx < 2:
                # Absolute rows
                if col_idx == 0:          # h [nm]
                    cmap = "RdBu_r"
                    sym  = _sym_vmax(arr_plot)
                    norm = TwoSlopeNorm(vcenter=0, vmin=-sym, vmax=sym)
                    cb_label = "nm"
                elif col_idx == 1:        # intensity
                    cmap = "inferno"
                    vmin = float(np.percentile(arr_plot, 1))
                    vmax = float(np.percentile(arr_plot, 99))
                    norm = PowerNorm(gamma=0.45, vmin=max(vmin, 0), vmax=vmax)
                    cb_label = "a.u."
                else:                     # unwrapped phase
                    cmap = "hsv"
                    vmin = float(arr_plot.min())
                    vmax = float(arr_plot.max())
                    # centre the hsv range on the median for contrast
                    med  = float(np.median(arr_plot))
                    half = max(abs(vmax - med), abs(med - vmin))
                    norm = Normalize(vmin=med - half, vmax=med + half)
                    cb_label = "rad"
            else:
                # Difference row
                sym = _sym_vmax(arr_plot)
                if col_idx == 0:
                    cmap = "RdBu_r"
                    norm = TwoSlopeNorm(vcenter=0, vmin=-sym, vmax=sym)
                    cb_label = "nm"
                elif col_idx == 1:
                    cmap = "RdBu_r"
                    norm = TwoSlopeNorm(vcenter=0, vmin=-sym, vmax=sym)
                    cb_label = "a.u."
                else:
                    cmap = "RdBu_r"
                    norm = TwoSlopeNorm(vcenter=0, vmin=-sym, vmax=sym)
                    cb_label = "rad"

            im = ax.imshow(
                arr_plot,
                extent=extent, origin="upper",
                cmap=cmap, norm=norm,
                interpolation="nearest", aspect="equal",
            )

            # Axis labels & ticks
            if row_idx == 0:
                ax.set_title(col_label, fontsize=9, color=_WHITE, pad=5)
            if col_idx == 0:
                ax.set_ylabel(f"{row_label}\n{ylabel}", fontsize=8, color=_WHITE)
            else:
                ax.set_ylabel(ylabel, fontsize=8, color=_WHITE)
            ax.set_xlabel(xlabel, fontsize=8, color=_WHITE)
            ax.tick_params(colors=_WHITE, labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor(_GRID)

            _add_colorbar(fig, im, ax, label=cb_label)

    fig.savefig(save_path, dpi=130, facecolor=_DARK, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> '{save_path}'")


# ======================================================================
#  Main
# ======================================================================

# Per-case settings:  (display_name, h_generator, zoom_cmos_um, zoom_h_um)
# zoom_h_um=0 means full membrane view
CASES = [
    ("Flat (no deformation)",
     lambda N, dx, dev: make_h_flat(N, dx, dev),
     5120.0, 0),

    ("Single Pixel (centre)",
     lambda N, dx, dev: make_h_single_pixel(N, dx, dev, amplitude=158e-9),
     300.0, 100.0),

    ("Single Gaussian Bump  (sigma=100 um, h=500 nm)",
     lambda N, dx, dev: make_h_single_bump(N, dx, dev),
     600.0, 600.0),

    ("Multi Bump  (5 bumps)",
     lambda N, dx, dev: make_h_multi_bump(N, dx, dev),
     2000.0, 2000.0),

    ("Random Band-limited  (sigma_filter=200 um, max=400 nm)",
     lambda N, dx, dev: make_h_random(N, dx, dev),
     5120.0, 0),
]


def main() -> None:
    os.makedirs("output", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"PyTorch {torch.__version__}  |  device: {device}\n")

    sensor = HolographicSensor(
        wavelength=632.8e-9, mem_res=1260, mem_pitch=10e-6,
        cmos_res=1024, grid_res=1260, distance=5e-3, device=device,
    ).to(device)

    N, dx = sensor.mem_res, sensor.mem_pitch

    header = f"{'Case':<45s}  {'h_max [nm]':>12s}  {'elapsed':>8s}"
    print(header)
    print("-" * len(header))

    for name, h_gen, zoom_c, zoom_h in CASES:
        t0    = time.time()
        h_def = h_gen(N, dx, device)
        data  = compute_case(sensor, h_def, device)

        slug  = (name.lower()
                 .replace(" ", "_")
                 .replace("(", "").replace(")", "")
                 .replace("=", "").replace(",", "")
                 .replace("/", "_"))
        slug  = "_".join(slug.split())          # collapse spaces
        path  = f"output/case_{slug}.png"

        plot_case(data, sensor, name, path,
                  zoom_cmos_um=zoom_c, zoom_h_um=zoom_h)

        h_max = float(h_def.abs().max()) * 1e9
        print(f"  {name:<43s}  {h_max:>10.1f} nm  {time.time()-t0:>6.1f}s")

    print("\nDone.")


if __name__ == "__main__":
    main()
