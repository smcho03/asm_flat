"""
Sanity Checks & Height Sweep Visualisation
for the Holographic Tactile Sensor Forward Model

Sanity checks
-------------
  1. Flat membrane    : h = 0  ->  I = 1.0 everywhere  (max error < 1e-4)
  2. Energy conserv.  : sum|Ud|^2 == sum|U0|^2  (Parseval via FFT)
  3. Symmetry         : symmetric h  ->  symmetric I  (x- and y-flip)
  4. Phase periodicity: uniform h = lambda/2  ->  phase = 2pi  ->  I = 1.0
  5. Gradient flow    : loss.backward() produces finite, non-NaN gradients
  6. Off-centre shift : bump shifted +delta_x  ->  pattern peak shifts +delta_x

Height-sweep figures
--------------------
  A. Grid (4x5) of CMOS zoom images at heights 0 .. 5*lambda
  B. Modulation curve: I_peak vs h (shows lambda/2 periodicity)
  C. Waterfall: spatial cross-section vs height (2-D false-colour)
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

# Import sensor model from the main module
from holographic_tactile_sensor import HolographicSensor, gaussian_bump


# ======================================================================
#  Helpers
# ======================================================================

PASS = "PASS"
FAIL = "FAIL"

def _fmt(ok: bool, label: str, detail: str) -> str:
    tag = PASS if ok else FAIL
    return f"  [{tag}]  {label:<45s}  {detail}"


# ======================================================================
#  Sanity checks
# ======================================================================

def run_sanity_checks(sensor: HolographicSensor, device: str) -> bool:
    """
    Run all sanity checks.  Returns True if every test passes.
    """
    lam = sensor.lam
    N   = sensor.mem_res
    dx  = sensor.mem_pitch
    M   = sensor.cmos_res

    all_pass = True
    lines    = []

    print("\n" + "=" * 64)
    print("  SANITY CHECKS")
    print("=" * 64)

    # ------------------------------------------------------------------
    # Test 1 — Flat membrane -> uniform intensity 1.0
    # ------------------------------------------------------------------
    h_zero = torch.zeros(N, N, device=device)
    with torch.no_grad():
        I_zero = sensor(h_zero)

    err1 = float((I_zero - 1.0).abs().max())
    ok1  = err1 < 1e-4
    all_pass &= ok1
    print(_fmt(ok1, "Flat membrane (h=0)  ->  I = 1.0",
               f"max|I-1| = {err1:.2e}"))

    # ------------------------------------------------------------------
    # Test 2 — Energy conservation (Parseval: full field before crop)
    # ------------------------------------------------------------------
    h_bump = gaussian_bump(N=N, dx=dx, amplitude=500e-9, sigma=100e-6, device=device)
    with torch.no_grad():
        Ud_full = sensor.propagated_field(h_bump)

    E_in  = float(N * N)                              # |A|^2 = 1 uniform
    E_out = float(torch.abs(Ud_full).pow(2).sum())
    rel2  = abs(E_out - E_in) / E_in
    ok2   = rel2 < 1e-4
    all_pass &= ok2
    print(_fmt(ok2, "Energy conservation  sum|Ud|^2 = sum|U0|^2",
               f"E_in={E_in:.0f}  E_out={E_out:.4f}  rel_err={rel2:.2e}"))

    # ------------------------------------------------------------------
    # Test 3 — Symmetry: Gaussian h is symmetric -> I must be symmetric
    # ------------------------------------------------------------------
    with torch.no_grad():
        I_bump = sensor(h_bump)

    err3x = float((I_bump - I_bump.flip(0)).abs().max())
    err3y = float((I_bump - I_bump.flip(1)).abs().max())
    ok3   = max(err3x, err3y) < 1e-4
    all_pass &= ok3
    print(_fmt(ok3, "Symmetry (x-flip & y-flip)",
               f"max err x={err3x:.2e}  y={err3y:.2e}"))

    # ------------------------------------------------------------------
    # Test 4 — Phase periodicity: uniform h = lambda/2  ->  phase = 2pi
    #          exp(i*2pi) = 1, so identical to h=0  ->  I = 1.0
    # ------------------------------------------------------------------
    h_wrap = torch.full((N, N), float(lam / 2), device=device)
    with torch.no_grad():
        I_wrap = sensor(h_wrap)

    err4 = float((I_wrap - 1.0).abs().max())
    ok4  = err4 < 1e-4
    all_pass &= ok4
    print(_fmt(ok4, "Phase wrap  h=lam/2  ->  2pi  ->  I=1",
               f"max|I-1| = {err4:.2e}"))

    # ------------------------------------------------------------------
    # Test 5 — Gradient flow: backward() must not produce NaN / Inf
    # ------------------------------------------------------------------
    h_grad = gaussian_bump(N=N, dx=dx, amplitude=500e-9, sigma=100e-6,
                           device=device)
    h_grad.requires_grad_(True)
    I_g = sensor(h_grad)
    I_g.mean().backward()
    g = h_grad.grad
    ok5_notnone = g is not None
    ok5_finite  = ok5_notnone and (not g.isnan().any()) and (not g.isinf().any())
    ok5 = ok5_notnone and ok5_finite
    all_pass &= ok5
    detail5 = (f"grad range [{g.min():.3e}, {g.max():.3e}]"
               if ok5 else "gradient is None or non-finite")
    print(_fmt(ok5, "Gradient flow  (no NaN / Inf)", detail5))

    # ------------------------------------------------------------------
    # Test 6 -- Off-centre bump: peak in I shifts by expected amount
    # Shift in the COLUMN (Y / 2nd dimension) direction so we can compare
    # col positions simply via argmax % M.
    # ------------------------------------------------------------------
    shift_px = 50                          # pixels on membrane
    shift_m  = shift_px * dx              # physical shift [m]

    # center=(cx, cy): X is dim-0 (rows), Y is dim-1 (cols).
    # We shift in Y (cols) so the column position of the peak changes.
    h_shift = gaussian_bump(N=N, dx=dx, amplitude=500e-9, sigma=100e-6,
                            center=(0.0, shift_m), device=device)
    with torch.no_grad():
        I_shifted = sensor(h_shift)

    # Compare column position of peak in CMOS (M x M)
    col_orig  = int(I_bump.argmax())    % M
    col_shift = int(I_shifted.argmax()) % M
    delta_px  = col_shift - col_orig
    ok6 = abs(delta_px - shift_px) <= 3          # within 3 px tolerance
    all_pass &= ok6
    print(_fmt(ok6, f"Off-centre bump  ->  peak shifts +{shift_px} px (col)",
               f"detected shift = {delta_px:+d} px"))

    # ------------------------------------------------------------------
    # Test 7 — Two-bump symmetry: bumps at ±offset produce symmetric I
    # ------------------------------------------------------------------
    off_m = 200e-6
    h_two = (gaussian_bump(N=N, dx=dx, amplitude=400e-9, sigma=80e-6,
                           center=( off_m, 0.0), device=device) +
             gaussian_bump(N=N, dx=dx, amplitude=400e-9, sigma=80e-6,
                           center=(-off_m, 0.0), device=device))
    with torch.no_grad():
        I_two = sensor(h_two)

    err7 = float((I_two - I_two.flip(0)).abs().max())
    ok7  = err7 < 1e-4
    all_pass &= ok7
    print(_fmt(ok7, "Two symmetric bumps  ->  symmetric I",
               f"max err = {err7:.2e}"))

    print("=" * 64)
    verdict = "ALL PASS" if all_pass else "SOME TESTS FAILED"
    print(f"  Result: {verdict}\n")
    return all_pass


# ======================================================================
#  Height sweep
# ======================================================================

def _get_zoom_window(I_sample: torch.Tensor, dx: float):
    """Estimate a tight zoom window from a sample CMOS intensity image."""
    M   = I_sample.shape[0]
    row = I_sample[M // 2, :].cpu().numpy()
    bg  = float(np.percentile(I_sample.cpu().numpy(), 50))
    hi  = float(row.max())
    above = row > (bg + (hi - bg) * 0.5)
    spot_half = max(int(above.sum() // 2), 3)
    zoom_px   = min(max(spot_half * 20, 50), M // 4)
    ctr       = M // 2
    sc        = max(ctr - zoom_px, 0)
    ec        = min(ctr + zoom_px, M)
    ax        = (np.arange(M) - ctr) * dx * 1e6          # [um]
    return sc, ec, ax


def plot_height_sweep(
    sensor    : HolographicSensor,
    device    : str,
    sigma_m   : float = 100e-6,
    n_grid_h  : int   = 4,     # rows in image grid
    n_grid_w  : int   = 5,     # cols in image grid
    n_curve   : int   = 300,   # points on modulation curve
    save_prefix: str  = "height_sweep",
) -> None:
    """
    Two output figures:
      1. Grid of CMOS zoom images across heights 0 .. 5*lambda
      2. Modulation curve (I_peak vs h) + waterfall cross-section
    """
    lam = sensor.lam
    N   = sensor.mem_res
    dx  = sensor.mem_pitch
    M   = sensor.cmos_res

    h_max_m   = 5.0 * lam                         # [m]
    h_max_nm  = h_max_m * 1e9                      # [nm]

    # Normalised spatial bump (peak = 1 m)
    h_base = gaussian_bump(N=N, dx=dx, amplitude=1.0, sigma=sigma_m,
                           device=device)           # peak = 1 [m]

    # ---- Determine zoom window from h = 500 nm reference ---------------
    with torch.no_grad():
        I_ref = sensor(h_base * 500e-9)
    sc, ec, ax_c = _get_zoom_window(I_ref, dx)

    # ---- Grid: n_grid_h x n_grid_w images ------------------------------
    n_total     = n_grid_h * n_grid_w
    h_grid_nm   = np.linspace(0.0, h_max_nm, n_total)

    print(f"\nGenerating {n_total}-panel grid  (0 .. {h_max_nm:.0f} nm) ...")
    t0 = time.time()

    # ---- Collect all images first (fast, no matplotlib) ----------------
    grid_imgs = []
    grid_vmax_list = []
    for h_nm in h_grid_nm:
        h_t = h_base * float(h_nm * 1e-9)
        with torch.no_grad():
            I_c = sensor(h_t)
        crop = I_c[sc:ec, sc:ec].cpu().numpy()
        grid_imgs.append(crop)
        grid_vmax_list.append(float(crop.max()))

    vmax_global = max(grid_vmax_list)
    vmin_global = 0.0
    ext_z = [ax_c[sc], ax_c[ec - 1], ax_c[ec - 1], ax_c[sc]]

    # ---- Render grid with ONE shared colorbar (fast) -------------------
    fig1, axes = plt.subplots(
        n_grid_h, n_grid_w,
        figsize=(3.2 * n_grid_w, 3.0 * n_grid_h),
        constrained_layout=True,          # faster than tight_layout
    )
    fig1.suptitle(
        f"CMOS Intensity vs Gaussian Bump Height\n"
        f"(sigma={sigma_m*1e6:.0f} um, lambda={lam*1e9:.1f} nm, d=5 mm)",
        fontsize=12,
    )

    im_last = None
    for idx, (h_nm, img) in enumerate(zip(h_grid_nm, grid_imgs)):
        row, col = divmod(idx, n_grid_w)
        ax = axes[row, col]
        phase_pi = (4.0 * np.pi / lam) * h_nm * 1e-9 / np.pi
        im_last = ax.imshow(
            img, extent=ext_z, cmap="inferno", origin="upper",
            vmin=vmin_global, vmax=vmax_global,
        )
        ax.set_title(f"h={h_nm:.0f}nm  phase={phase_pi:.2f}pi", fontsize=7)
        ax.set_xlabel("x [um]", fontsize=6)
        ax.set_ylabel("y [um]", fontsize=6)
        ax.tick_params(labelsize=5)

    # Single colorbar placed outside the grid
    fig1.colorbar(im_last, ax=axes, location="right", shrink=0.6,
                  label="Intensity [a.u.]")

    path1 = f"output/{save_prefix}_grid.png"
    fig1.savefig(path1, dpi=120, bbox_inches="tight")
    plt.close(fig1)
    print(f"  [{time.time()-t0:.1f}s]  Grid saved -> '{path1}'")

    # ---- Modulation curve + waterfall ----------------------------------
    h_curve_nm = np.linspace(0.0, h_max_nm, n_curve)
    I_peaks    = []
    I_mins     = []
    I_means    = []
    profiles   = []   # centre row of CMOS zoom for each h

    print(f"Computing modulation curve ({n_curve} points) ...")
    t1 = time.time()
    for h_nm in h_curve_nm:
        h_t = h_base * float(h_nm * 1e-9)
        with torch.no_grad():
            I_c = sensor(h_t)
        I_np = I_c.cpu().numpy()
        I_peaks.append(float(I_np.max()))
        I_mins.append(float(I_np.min()))
        I_means.append(float(I_np.mean()))
        # Central cross-section (full CMOS width)
        profiles.append(I_np[M // 2, :].copy())

    print(f"  [{time.time()-t1:.1f}s]  Curve computed.")

    # Waterfall matrix: [n_curve, M]
    waterfall = np.stack(profiles, axis=0)   # [n_curve, M]

    fig2, axes2 = plt.subplots(3, 1, figsize=(12, 14))
    fig2.suptitle(
        f"Sensor Modulation Transfer: I_peak vs Bump Height\n"
        f"(sigma={sigma_m*1e6:.0f} um, d=5 mm, lambda={lam*1e9:.1f} nm)",
        fontsize=13,
    )

    # --- Sub-plot A: I_peak, I_min, I_mean vs h [nm] ---
    ax_a = axes2[0]
    ax_a.plot(h_curve_nm, I_peaks, "b-",  lw=1.8, label="I_peak")
    ax_a.plot(h_curve_nm, I_mins,  "c--", lw=1.2, label="I_min")
    ax_a.plot(h_curve_nm, I_means, "r:",  lw=1.2, label="I_mean (CMOS)")
    ax_a.axhline(1.0, color="gray", ls=":", lw=0.8, label="Background (flat)")

    # Mark phase-wrap lines (every lambda/2 in h)
    h_wrap_nm = lam * 1e9 / 2.0
    for n in range(1, int(h_max_nm / h_wrap_nm) + 2):
        xv = n * h_wrap_nm
        if xv <= h_max_nm:
            ax_a.axvline(xv, color="green", alpha=0.25, lw=1.0,
                         label="n*lambda/2" if n == 1 else None)
    ax_a.set_xlabel("Bump height  h [nm]")
    ax_a.set_ylabel("Intensity [a.u.]")
    ax_a.set_title("I_peak / I_min / I_mean vs height  (green: n*lambda/2 wrap lines)")
    ax_a.legend(loc="upper right", fontsize=9)
    ax_a.grid(True, alpha=0.3)

    # Secondary x-axis in phase units
    ax_a2 = ax_a.twiny()
    ax_a2.set_xlim(ax_a.get_xlim())
    phase_ticks_nm = np.arange(0, h_max_nm + 1, h_wrap_nm)
    ax_a2.set_xticks(phase_ticks_nm)
    ax_a2.set_xticklabels([f"{2*i}pi" for i in range(len(phase_ticks_nm))],
                          fontsize=7)
    ax_a2.set_xlabel("Peak phase  [rad]", fontsize=8)

    # --- Sub-plot B: same data with phase-x axis only ---
    phase_axis = (4.0 * np.pi / lam) * np.array(h_curve_nm) * 1e-9
    ax_b = axes2[1]
    ax_b.plot(phase_axis, I_peaks, "b-",  lw=1.8, label="I_peak")
    ax_b.plot(phase_axis, I_mins,  "c--", lw=1.2, label="I_min")
    ax_b.axhline(1.0, color="gray", ls=":", lw=0.8)
    for n in range(0, int(phase_axis[-1] / (2 * np.pi)) + 2):
        xv = n * 2 * np.pi
        if xv <= phase_axis[-1]:
            ax_b.axvline(xv, color="green", alpha=0.25, lw=1.0)
    ax_b.set_xlabel("Peak phase at bump centre  [rad]")
    ax_b.set_ylabel("Intensity [a.u.]")
    ax_b.set_title("I_peak vs phase  (period = 2pi  <=>  h = lambda/2)")
    ax_b.legend(fontsize=9)
    ax_b.grid(True, alpha=0.3)

    # --- Sub-plot C: Waterfall (height x spatial position) ---
    ax_c2 = axes2[2]
    spatial_um = (np.arange(M) - M // 2) * dx * 1e6   # [um]
    wf_extent  = [spatial_um[0], spatial_um[-1], h_curve_nm[-1], h_curve_nm[0]]
    im_wf = ax_c2.imshow(
        waterfall, aspect="auto", cmap="inferno",
        extent=wf_extent, origin="upper",
        norm=PowerNorm(gamma=0.4),
    )
    ax_c2.set_xlabel("x on CMOS  [um]")
    ax_c2.set_ylabel("Bump height  h [nm]")
    ax_c2.set_title("Waterfall: central cross-section  I(x, h)  "
                    "[sqrt-norm; brighter = higher intensity]")
    plt.colorbar(im_wf, ax=ax_c2, label="Intensity [a.u.]")
    # overlay phase-wrap horizontal lines
    for n in range(1, int(h_max_nm / h_wrap_nm) + 2):
        yv = n * h_wrap_nm
        if yv <= h_max_nm:
            ax_c2.axhline(yv, color="lime", alpha=0.4, lw=0.8,
                          label="n*lambda/2" if n == 1 else None)
    ax_c2.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    path2 = f"output/{save_prefix}_modulation.png"
    fig2.savefig(path2, dpi=130, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Modulation figure saved -> '{path2}'")


# ======================================================================
#  Main
# ======================================================================

def main() -> None:
    os.makedirs("output", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"PyTorch {torch.__version__}  |  device: {device}")

    sensor = HolographicSensor(
        wavelength = 632.8e-9,
        mem_res    = 1260,
        mem_pitch  = 10e-6,
        cmos_res   = 1024,
        grid_res   = 1260,
        distance   = 5e-3,
        device     = device,
    ).to(device)

    # 1. Sanity checks
    all_ok = run_sanity_checks(sensor, device)
    if not all_ok:
        print("WARNING: one or more sanity checks failed - inspect output above.")

    # 2. Height sweep visualisation
    plot_height_sweep(
        sensor       = sensor,
        device       = device,
        sigma_m      = 100e-6,
        n_grid_h     = 4,
        n_grid_w     = 5,
        n_curve      = 300,
        save_prefix  = "height_sweep",
    )

    print("\nDone.  Output files:")
    print("  output/height_sweep_grid.png")
    print("  output/height_sweep_modulation.png")


if __name__ == "__main__":
    main()
