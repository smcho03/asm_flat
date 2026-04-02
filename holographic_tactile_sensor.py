"""
Differentiable Forward Simulation Pipeline
for a Lensless Holographic Tactile Sensor

Physics Pipeline:
  1. Phase Modulation  : U0(x,y) = A(x,y) * exp(i * 4pi/lam * h(x,y))
                         (reflection -> round-trip factor 4pi/lam)
  2. ASM Propagation   : U_d = FFT2(U0) * H(fx, fy)
                         H = exp(i*2pi*d*sqrt(1/lam2-fx2-fy2)) with band-limiting
  3. CMOS Intensity    : I = |IFFT2(U_d)|2 , cropped to sensor size

All operations are PyTorch-based -> fully differentiable w.r.t. h(x,y).

Reference for band-limited ASM:
  Matsushima & Shimobaba, Opt. Express 17, 19662 (2009)
"""

from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")            # non-interactive: saves to file without blocking
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 (activates 3-D projection)
from typing import Optional


# =====================================================================
#  Core Model
# ======================================================================

class HolographicSensor(nn.Module):
    """
    Differentiable forward model of a lensless holographic tactile sensor.

    Parameters
    ----------
    wavelength : float
        Illumination wavelength [m].  Default: 632.8 nm (He-Ne laser).
    mem_res : int
        Physical membrane pixel count (square).  Default: 512.
    mem_pitch : float
        Membrane pixel pitch [m].  Default: 10 um.
    cmos_res : int
        CMOS pixel count (square).  Default: 1024.
    grid_res : int
        Simulation grid pixel count (square).  Must be >= cmos_res.
        Default: 1536.  Increase to reduce wrap-around contamination.
    distance : float
        Free-space propagation distance from membrane to sensor [m].
        Default: 5 mm.
    device : str
        'cpu' or 'cuda'.
    """

    def __init__(
        self,
        wavelength : float = 632.8e-9,
        mem_res    : int   = 512,
        mem_pitch  : float = 10e-6,
        cmos_res   : int   = 1024,
        grid_res   : int   = 1536,
        distance   : float = 5e-3,
        device     : str   = "cpu",
    ) -> None:
        super().__init__()

        assert grid_res >= cmos_res, "grid_res must be >= cmos_res."
        assert grid_res >= mem_res,  "grid_res must be >= mem_res."

        self.lam       = wavelength
        self.mem_res   = mem_res
        self.mem_pitch = mem_pitch
        self.cmos_res  = cmos_res
        self.grid_res  = grid_res
        self.distance  = distance
        self.device    = torch.device(device)

        # Precomputed (band-limited) ASM transfer function - not a learnable parameter
        H = self._build_transfer_function()          # [grid_res, grid_res] complex64
        self.register_buffer("H", H)

        # Membrane region within the grid (centred)
        ms = (grid_res - mem_res) // 2
        self._ms = ms                                # membrane start
        self._me = ms + mem_res                      # membrane end

        # CMOS crop indices from grid (centred)
        cs = (grid_res - cmos_res) // 2
        self._s = cs                                 # crop start
        self._e = cs + cmos_res                      # crop end

    # ------------------------------------------------------------------
    # Private: Angular Spectrum Transfer Function
    # ------------------------------------------------------------------

    def _build_transfer_function(self) -> torch.Tensor:
        """
        Construct the band-limited ASM transfer function H(fx, fy).

        H(fx, fy) = exp(i * 2pi * d * sqrt(1/lam2 - fx2 - fy2)) * mask

        Two masks are applied:
          1. Evanescent-wave filter : removes fx2+fy2 >= 1/lam2
          2. Propagation-aliasing filter (Matsushima 2009):
               |fx|, |fy| <= sin(theta_max)/lam
             where sin(theta_max) = (L/2) / sqrt(d2 + (L/2)2)
             and L = N*Deltax is the physical extent of the field.

        Returns
        -------
        H : complex64 tensor of shape [grid_res, grid_res], FFT-ordered.
        """
        N   = self.grid_res
        dx  = self.mem_pitch
        d   = self.distance
        lam = self.lam

        # ---- Spatial frequency grid (FFT-natural ordering: 0, +, ..., -) ----
        # fftfreq(N, d=dx)[k] = k/(N*dx)  [cycles/m]
        f   = torch.fft.fftfreq(N, d=dx)            # [N]
        Fx, Fy = torch.meshgrid(f, f, indexing="ij") # [N, N]
        F2  = Fx**2 + Fy**2                          # [N, N]

        # ---- Mask 1: Evanescent wave suppression ----
        prop_mask = F2 < (1.0 / lam**2)

        # ---- Mask 2: Propagation-aliasing limit (Matsushima 2009) ----
        # Critical angle whose group delay just fits inside the field window:
        #   max |dphi/dfx| = d*|fx|/sqrt(1/lam2-F2) <= L/2  ->  |fx| <= f_max
        L       = N * dx
        sin_max = (L / 2.0) / np.sqrt(d**2 + (L / 2.0)**2)
        f_max   = sin_max / lam                      # [cycles/m]
        alias_mask = (torch.abs(Fx) <= f_max) & (torch.abs(Fy) <= f_max)

        mask = prop_mask & alias_mask                # combined boolean mask

        # ---- Transfer function phase ----
        sqrt_arg = torch.clamp(1.0 / lam**2 - F2, min=0.0)
        phase_H  = 2.0 * np.pi * d * torch.sqrt(sqrt_arg)

        # Build complex H; zeroed wherever mask is False
        mask_f = mask.float()
        H = torch.polar(mask_f, phase_H)             # complex64, [N, N]

        return H

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        h : torch.Tensor,
        A : Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the CMOS intensity pattern.

        Parameters
        ----------
        h : Tensor [mem_res, mem_res]
            Membrane height map in *metres*.  Must be on the same device as
            the model.  Gradient-tracking is preserved.
        A : Tensor [mem_res, mem_res], optional
            Amplitude map over the membrane (default: uniform 1.0).
            Outside the membrane boundary A is always 0.

        Returns
        -------
        I_cmos : Tensor [cmos_res, cmos_res]  - real-valued intensity.
        """
        if A is None:
            A = torch.ones_like(h)

        # Pad membrane region to full grid (zero outside = no illumination)
        pad = (self._ms, self.grid_res - self._me,
               self._ms, self.grid_res - self._me)
        h_grid = F.pad(h, pad)                      # [grid_res, grid_res]
        A_grid = F.pad(A, pad)                      # [grid_res, grid_res]

        # Step 1 - Phase modulation (reflection: double-pass -> factor 4pi/lam)
        phi = (4.0 * np.pi / self.lam) * h_grid    # [grid_res, grid_res] real
        U0  = A_grid * torch.exp(1j * phi)          # [grid_res, grid_res] complex

        # Step 2 - Angular Spectrum propagation
        Ud  = self._asm_propagate(U0)               # [grid_res, grid_res] complex

        # Step 3 - Intensity & CMOS crop
        I_full  = torch.abs(Ud).pow(2)              # [grid_res, grid_res] real
        I_cmos  = I_full[self._s:self._e, self._s:self._e]   # [cmos_res, cmos_res]
        return I_cmos

    def _asm_propagate(self, U0: torch.Tensor) -> torch.Tensor:
        """IFFT2( FFT2(U0) * H ) - one-line ASM."""
        return torch.fft.ifft2(torch.fft.fft2(U0) * self.H)

    # ------------------------------------------------------------------
    # Helpers for analysis / visualisation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def propagated_field(
        self,
        h : torch.Tensor,
        A : Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return full [grid_res, grid_res] complex field at sensor plane."""
        if A is None:
            A = torch.ones_like(h)
        pad = (self._ms, self.grid_res - self._me,
               self._ms, self.grid_res - self._me)
        h_grid = F.pad(h, pad)
        A_grid = F.pad(A, pad)
        phi = (4.0 * np.pi / self.lam) * h_grid
        U0  = A_grid * torch.exp(1j * phi)
        return self._asm_propagate(U0)

    def crop(self, field: torch.Tensor) -> torch.Tensor:
        """Centre-crop [mem_res, mem_res] field to [cmos_res, cmos_res]."""
        return field[self._s:self._e, self._s:self._e]

    def print_params(self) -> None:
        """Print a summary of the optical system parameters."""
        dx  = self.mem_pitch
        d   = self.distance
        lam = self.lam
        L   = self.grid_res * dx
        sin_max = (L / 2.0) / np.sqrt(d**2 + (L / 2.0)**2)
        f_nyq   = 1.0 / (2.0 * dx)
        f_max   = sin_max / lam

        print("=" * 56)
        print(" Holographic Tactile Sensor - System Parameters")
        print("=" * 56)
        print(f"  Wavelength          : {lam*1e9:.1f} nm")
        print(f"  Membrane            : {self.mem_res}x{self.mem_res} px  ({self.mem_res*dx*1e3:.2f} mm)")
        print(f"  CMOS                : {self.cmos_res}x{self.cmos_res} px  ({self.cmos_res*dx*1e3:.2f} mm)")
        print(f"  Simulation grid     : {self.grid_res}x{self.grid_res} px  ({L*1e3:.2f} mm)")
        print(f"  Propagation dist.   : {d*1e3:.1f} mm")
        print(f"  Nyquist freq        : {f_nyq/1e3:.0f} cycles/mm")
        print(f"  ASM band-limit freq : {f_max/1e3:.0f} cycles/mm")
        print(f"  Max half-angle      : {np.degrees(np.arcsin(sin_max)):.1f} deg")
        print("=" * 56)


# ======================================================================
#  Test input: Gaussian bump
# ======================================================================

def gaussian_bump(
    N         : int             = 1260,
    dx        : float           = 10e-6,
    amplitude : float           = 500e-9,        # [m] peak height
    sigma     : float           = 100e-6,        # [m] 1-sigma lateral width
    center    : tuple           = (0.0, 0.0),    # [m] (cx, cy) offset from centre
    device    : str             = "cpu",
) -> torch.Tensor:
    """
    Create a Gaussian membrane deformation centred at (cx, cy).

    Returns
    -------
    h : float32 tensor [N, N], height in metres.
    """
    # Use (N-1)/2 so coords are exactly symmetric under flip for any N (odd or even).
    # For N=1260: coords = [-629.5, -628.5, ..., +629.5] * dx  -> flip(0) maps i -> N-1-i,
    # which negates each coordinate, preserving Gaussian symmetry exactly.
    coords = (torch.arange(N, dtype=torch.float32, device=device) - (N - 1) * 0.5) * dx
    X, Y   = torch.meshgrid(coords, coords, indexing="ij")
    cx, cy = center
    h      = amplitude * torch.exp(-((X - cx)**2 + (Y - cy)**2) / (2.0 * sigma**2))
    return h


# ======================================================================
#  Visualisation
# ======================================================================

def visualize(
    h       : torch.Tensor,   # [N, N] float,   membrane height [m]
    I_cmos  : torch.Tensor,   # [M, M] float,   CMOS intensity
    Ud_crop : torch.Tensor,   # [M, M] complex, propagated field (CMOS region)
    dx      : float = 10e-6,
    save_path: str  = "output/holographic_tactile_result.png",
) -> None:
    """
    Four-panel figure:
      (1) 3-D surface of h(x,y) - zoomed to bump region
      (2) CMOS intensity - full field, sqrt-scaled to reveal ring structure
      (3) CMOS intensity - zoomed centre, linear scale
      (4) Propagated field phase - full CMOS region
    """
    h_np     = h.cpu().float().numpy()
    I_np     = I_cmos.cpu().float().numpy()
    phase_np = torch.angle(Ud_crop).cpu().float().numpy()

    N_h = h_np.shape[0]
    N_c = I_np.shape[0]

    ax_h = (np.arange(N_h) - N_h // 2) * dx * 1e6   # [um]
    ax_c = (np.arange(N_c) - N_c // 2) * dx * 1e6   # [um]
    ext_c = [ax_c[0], ax_c[-1], ax_c[-1], ax_c[0]]

    fig = plt.figure(figsize=(22, 6))
    fig.suptitle("Holographic Tactile Sensor - Forward Simulation", fontsize=14)

    # ---- Panel 1: 3-D surface — zoomed to bump region -------------------
    # Estimate bump width from the height data and zoom ±8*sigma_px around centre
    h_thresh = h_np.max() * np.exp(-8)          # exp(-r^2/2sigma^2) at r=4sigma
    mask_1d = (h_np[N_h // 2, :] > h_thresh)
    bump_half_width_px = max(int(mask_1d.sum() // 2), 20)
    pad = max(bump_half_width_px * 4, 40)        # show ±4x the bump half-width
    ctr = N_h // 2
    s_z  = max(0, ctr - pad)
    e_z  = min(N_h, ctr + pad)

    ax1 = fig.add_subplot(141, projection="3d")
    ax_zoom = ax_h[s_z:e_z]
    h_zoom  = h_np[s_z:e_z, s_z:e_z]
    step_z  = max(1, len(ax_zoom) // 80)
    Xg, Yg = np.meshgrid(ax_zoom[::step_z], ax_zoom[::step_z], indexing="ij")
    surf = ax1.plot_surface(
        Xg, Yg, h_zoom[::step_z, ::step_z] * 1e9,
        cmap="viridis", linewidth=0, antialiased=True,
    )
    ax1.set_xlabel("x [um]")
    ax1.set_ylabel("y [um]")
    ax1.set_zlabel("h [nm]")
    ax1.set_title("h(x,y) — bump region")
    fig.colorbar(surf, ax=ax1, shrink=0.5, pad=0.12, label="nm")

    # ---- Panel 2: CMOS intensity — full field, sqrt-norm -----------------
    # PowerNorm(gamma<1) compresses bright spots so ring structure is visible
    ax2 = fig.add_subplot(142)
    im2 = ax2.imshow(
        I_np, extent=ext_c, cmap="inferno", origin="upper",
        norm=PowerNorm(gamma=0.35, vmin=I_np.min(), vmax=I_np.max()),
    )
    ax2.set_xlabel("x [um]")
    ax2.set_ylabel("y [um]")
    ax2.set_title("CMOS I(x,y)  [sqrt-norm, full]")
    fig.colorbar(im2, ax=ax2, label="Intensity [a.u.]")

    # ---- Panel 3: CMOS intensity — zoomed centre, linear -----------------
    # Zoom tightly: estimate sigma from the CMOS intensity peak width (FWHM -> sigma)
    # Use the central row of I_np to estimate the diffracted spot half-width
    I_row     = I_np[N_c // 2, :]
    I_bg      = float(np.percentile(I_np, 50))   # median ≈ background
    above_bg  = I_row > (I_bg + (I_row.max() - I_bg) * 0.5)
    spot_half = max(int(above_bg.sum() // 2), 3)
    # Show ±20× spot half-width, at least ±50 px, at most quarter of sensor
    zoom_px   = min(max(spot_half * 20, 50), N_c // 4)
    ctr_c   = N_c // 2
    sc, ec  = ctr_c - zoom_px, ctr_c + zoom_px
    sc, ec  = max(sc, 0), min(ec, N_c)
    I_zoom  = I_np[sc:ec, sc:ec]
    ext_z   = [ax_c[sc], ax_c[ec - 1], ax_c[ec - 1], ax_c[sc]]

    ax3 = fig.add_subplot(143)
    im3 = ax3.imshow(
        I_zoom, extent=ext_z, cmap="inferno", origin="upper",
    )
    ax3.set_xlabel("x [um]")
    ax3.set_ylabel("y [um]")
    ax3.set_title("CMOS I(x,y)  [linear, centre zoom]")
    fig.colorbar(im3, ax=ax3, label="Intensity [a.u.]")

    # ---- Panel 4: Phase of propagated field (CMOS region) ----------------
    ax4 = fig.add_subplot(144)
    im4 = ax4.imshow(
        phase_np, extent=ext_c, cmap="hsv", origin="upper",
        vmin=-np.pi, vmax=np.pi,
    )
    ax4.set_xlabel("x [um]")
    ax4.set_ylabel("y [um]")
    ax4.set_title("Propagated Phase (CMOS region)")
    fig.colorbar(im4, ax=ax4, label="Phase [rad]")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  -> Saved to '{save_path}'")
    # plt.show() intentionally omitted - using Agg (file-only) backend


# ======================================================================
#  Main: Sanity check
# ======================================================================

def main() -> None:
    os.makedirs("output", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"PyTorch {torch.__version__}  |  device: {device}\n")

    # ---- Build sensor model -------------------------------------------
    # mem_res(512px=5.12mm) < cmos_res(1024px=10.24mm):
    # CMOS sees the membrane rectangle + edge diffraction around it
    sensor = HolographicSensor(
        wavelength = 632.8e-9,
        mem_res    = 512,
        mem_pitch  = 10e-6,
        cmos_res   = 1024,
        grid_res   = 1536,
        distance   = 5e-3,
        device     = device,
    ).to(device)

    sensor.print_params()

    # ---- Gaussian bump test input -------------------------------------
    # amplitude = 500 nm  ->  phase swing ~= 4pi/lam * h_max ~= 9.94 rad  (~1.58 x 2pi)
    # sigma     = 100 um  ->  bump FWHM ~= 235 um (about 23.5 membrane pixels)
    h = gaussian_bump(
        N         = 512,
        dx        = 10e-6,
        amplitude = 500e-9,
        sigma     = 100e-6,
        device    = device,
    )
    h.requires_grad_(True)   # enable autograd

    phi_max = float((4 * np.pi / 632.8e-9) * 500e-9)
    print(f"\nGaussian bump:"
          f"\n  Peak height      : 500 nm"
          f"\n  sigma (1-sigma)  : 100 um  (FWHM ~= {100e-6*2.355*1e6:.0f} um)"
          f"\n  Max phase        : {phi_max:.3f} rad  ({phi_max/(2*np.pi):.2f} x 2pi)"
          f"\n  [CMOS sees 10.24mm, membrane is 5.12mm -> edge diffraction visible]\n")

    # ---- Forward pass ------------------------------------------------
    print("Running forward pass ...")
    I_cmos = sensor(h)
    print(f"  h      : {tuple(h.shape)}  dtype={h.dtype}")
    print(f"  I_cmos : {tuple(I_cmos.shape)}  dtype={I_cmos.dtype}")
    print(f"  I min/max : {I_cmos.min().item():.4e}  /  {I_cmos.max().item():.4e}")

    # ---- Backpropagation check ----------------------------------------
    loss = I_cmos.mean()
    loss.backward()
    g = h.grad
    print(f"\nGradient check (dL/dh):")
    print(f"  grad is not None : {g is not None}")
    if g is not None:
        print(f"  grad min/max     : {g.min().item():.4e}  /  {g.max().item():.4e}")
        print(f"  grad has NaN     : {g.isnan().any().item()}")
        print(f"  grad has Inf     : {g.isinf().any().item()}")

    # ---- Visualise ---------------------------------------------------
    print("\nRendering visualisation ...")
    with torch.no_grad():
        Ud_full = sensor.propagated_field(h.detach())
        Ud_crop = sensor.crop(Ud_full)

    visualize(
        h        = h.detach(),
        I_cmos   = I_cmos.detach(),
        Ud_crop  = Ud_crop,
        dx       = sensor.mem_pitch,
    )
    print("Done.")


if __name__ == "__main__":
    main()
