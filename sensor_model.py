"""
sensor_model.py
---------------
Differentiable forward model of a lensless holographic tactile sensor.

Physics pipeline:
  1. Phase modulation : U0(x,y) = A(x,y) * exp(i * 4*pi/lam * h(x,y))
                        (reflection -> round-trip factor 4pi/lam)
  2. ASM propagation  : Ud = IFFT2( FFT2(U0) * H(fx,fy) )
                        H = exp(i*2*pi*d*sqrt(1/lam^2-fx^2-fy^2)) + band-limiting
  3. CMOS intensity   : I = |Ud|^2, centre-cropped to cmos_res x cmos_res

Reference for band-limited ASM:
  Matsushima & Shimobaba, Opt. Express 17, 19662 (2009)

All operations are PyTorch-based -> fully differentiable w.r.t. h(x,y).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ======================================================================
#  Core Model
# ======================================================================

class HolographicSensor(nn.Module):
    """
    Differentiable forward model of a lensless holographic tactile sensor.

    Parameters
    ----------
    wavelength : float
        Illumination wavelength [m].
    mem_res : int
        Membrane pixel count (square).
    mem_pitch : float
        Membrane pixel pitch [m].
    cmos_res : int
        CMOS pixel count (square).
    grid_res : int
        Simulation grid pixel count.  Must be >= cmos_res and >= mem_res.
        Larger values reduce wrap-around contamination.
    distance : float
        Free-space propagation distance from membrane to CMOS [m].
    device : str
        'cpu' or 'cuda'.
    """

    def __init__(
        self,
        wavelength : float = 632.8e-9,
        mem_res    : int   = 1260,
        mem_pitch  : float = 10e-6,
        cmos_res   : int   = 1024,
        grid_res   : int   = 1260,
        distance   : float = 5e-3,
        device     : str   = "cpu",
    ) -> None:
        super().__init__()

        assert grid_res >= cmos_res, "grid_res must be >= cmos_res"
        assert grid_res >= mem_res,  "grid_res must be >= mem_res"

        self.lam       = wavelength
        self.mem_res   = mem_res
        self.mem_pitch = mem_pitch
        self.cmos_res  = cmos_res
        self.grid_res  = grid_res
        self.distance  = distance
        self.device    = torch.device(device)

        # Precomputed band-limited ASM transfer function
        H = self._build_transfer_function()   # [grid_res, grid_res] complex64
        self.register_buffer("H", H)

        # Membrane region within the grid (centred)
        ms = (grid_res - mem_res) // 2
        self._ms = ms
        self._me = ms + mem_res

        # CMOS crop indices (centred)
        cs = (grid_res - cmos_res) // 2
        self._s = cs
        self._e = cs + cmos_res

    # ------------------------------------------------------------------
    # Band-limited ASM transfer function
    # ------------------------------------------------------------------

    def _build_transfer_function(self) -> torch.Tensor:
        """
        H(fx,fy) = exp(i*2*pi*d*sqrt(1/lam^2-fx^2-fy^2)) * mask

        Masks:
          1. Evanescent: fx^2+fy^2 < 1/lam^2
          2. Aliasing limit (Matsushima 2009):
               |fx|,|fy| <= sin(theta_max)/lam
             sin(theta_max) = (L/2) / sqrt(d^2 + (L/2)^2)
             L = grid_res * mem_pitch
        """
        N   = self.grid_res
        dx  = self.mem_pitch
        d   = self.distance
        lam = self.lam

        f        = torch.fft.fftfreq(N, d=dx)          # [N]  FFT-natural order
        Fx, Fy   = torch.meshgrid(f, f, indexing="ij")
        F2       = Fx**2 + Fy**2

        # Mask 1 — evanescent suppression
        prop_mask = F2 < (1.0 / lam**2)

        # Mask 2 — propagation-aliasing limit
        L        = N * dx
        sin_max  = (L / 2.0) / np.sqrt(d**2 + (L / 2.0)**2)
        f_max    = sin_max / lam
        alias_mask = (F2 <= f_max**2)

        mask     = prop_mask & alias_mask
        sqrt_arg = torch.clamp(1.0 / lam**2 - F2, min=0.0)
        phase_H  = 2.0 * np.pi * d * torch.sqrt(sqrt_arg)
        H        = torch.polar(mask.float(), phase_H)  # complex64
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
        Parameters
        ----------
        h : Tensor [mem_res, mem_res]  — height map in metres
        A : Tensor [mem_res, mem_res]  — amplitude (default: uniform 1.0)

        Returns
        -------
        I_cmos : Tensor [cmos_res, cmos_res]  — real-valued intensity
        """
        if A is None:
            A = torch.ones_like(h)

        pad    = (self._ms, self.grid_res - self._me,
                  self._ms, self.grid_res - self._me)
        h_grid = F.pad(h, pad)
        A_grid = F.pad(A, pad)

        phi   = -(4.0 * np.pi / self.lam) * h_grid
        U0    = A_grid * torch.exp(1j * phi)
        Ud    = torch.fft.ifft2(torch.fft.fft2(U0) * self.H)

        I_full = torch.abs(Ud).pow(2)
        return I_full[self._s:self._e, self._s:self._e]

    # ------------------------------------------------------------------
    # Analysis helpers
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
        pad    = (self._ms, self.grid_res - self._me,
                  self._ms, self.grid_res - self._me)
        h_grid = F.pad(h, pad)
        A_grid = F.pad(A, pad)
        phi    = -(4.0 * np.pi / self.lam) * h_grid
        U0     = A_grid * torch.exp(1j * phi)
        return torch.fft.ifft2(torch.fft.fft2(U0) * self.H)

    def crop(self, field: torch.Tensor) -> torch.Tensor:
        """Centre-crop [grid_res, grid_res] field to [cmos_res, cmos_res]."""
        return field[self._s:self._e, self._s:self._e]

    def print_params(self) -> None:
        dx  = self.mem_pitch
        d   = self.distance
        lam = self.lam
        L   = self.grid_res * dx
        sin_max = (L / 2.0) / np.sqrt(d**2 + (L / 2.0)**2)
        f_nyq   = 1.0 / (2.0 * dx)
        f_max   = sin_max / lam
        print("=" * 56)
        print(" HolographicSensor - System Parameters")
        print("=" * 56)
        print(f"  Wavelength      : {lam*1e9:.1f} nm")
        print(f"  Membrane        : {self.mem_res}x{self.mem_res} px  ({self.mem_res*dx*1e3:.2f} mm)")
        print(f"  CMOS            : {self.cmos_res}x{self.cmos_res} px  ({self.cmos_res*dx*1e3:.2f} mm)")
        print(f"  Grid            : {self.grid_res}x{self.grid_res} px  ({L*1e3:.2f} mm)")
        print(f"  Distance        : {d*1e3:.1f} mm")
        print(f"  Nyquist freq    : {f_nyq/1e3:.0f} cycles/mm")
        print(f"  ASM band-limit  : {f_max/1e3:.0f} cycles/mm")
        print(f"  Max half-angle  : {np.degrees(np.arcsin(sin_max)):.1f} deg")
        print("=" * 56)


# ======================================================================
#  Deformation generators
# ======================================================================

def gaussian_bump(
    N         : int   = 1260,
    dx        : float = 10e-6,
    amplitude : float = 500e-9,
    sigma     : float = 100e-6,
    center    : tuple = (0.0, 0.0),
    device    : str   = "cpu",
) -> torch.Tensor:
    """
    Gaussian membrane deformation centred at (cx, cy).

    Uses (N-1)/2 so coordinates are exactly symmetric under flip
    for both odd and even N.

    Returns
    -------
    h : float32 tensor [N, N], height in metres
    """
    coords = (torch.arange(N, dtype=torch.float32, device=device) - (N - 1) * 0.5) * dx
    X, Y   = torch.meshgrid(coords, coords, indexing="ij")
    cx, cy = center
    return amplitude * torch.exp(-((X - cx)**2 + (Y - cy)**2) / (2.0 * sigma**2))


def make_h_flat(N: int, dx: float, device: str) -> torch.Tensor:
    return torch.zeros(N, N, dtype=torch.float32, device=device)


def make_h_single_pixel(
    N: int, dx: float, device: str,
    amplitude: float = 158e-9,
) -> torch.Tensor:
    """Single-pixel perturbation at membrane centre (lambda/4 -> pi phase shift)."""
    h    = torch.zeros(N, N, dtype=torch.float32, device=device)
    c    = (N - 1) // 2
    h[c, c] = amplitude
    return h


def make_h_single_bump(N: int, dx: float, device: str) -> torch.Tensor:
    return gaussian_bump(N=N, dx=dx, amplitude=500e-9, sigma=100e-6, device=device)


def make_h_multi_bump(N: int, dx: float, device: str) -> torch.Tensor:
    # Centers must be within ±(N/2)*dx of membrane centre.
    # For N=512, dx=10um: limit = ±2560um (all safe).
    # For N=128, dx=10um: limit = ±640um  (use reconstruct_sweep.make_h instead).
    specs = [
        dict(amplitude=500e-9, sigma= 80e-6, center=(     0.0,      0.0)),
        dict(amplitude=400e-9, sigma=120e-6, center=( 300e-6,   200e-6)),
        dict(amplitude=300e-9, sigma= 60e-6, center=(-250e-6,  -300e-6)),
        dict(amplitude=450e-9, sigma= 90e-6, center=( 200e-6,  -400e-6)),
        dict(amplitude=350e-9, sigma=150e-6, center=(-400e-6,   300e-6)),
    ]
    h = torch.zeros(N, N, dtype=torch.float32, device=device)
    for s in specs:
        h = h + gaussian_bump(N=N, dx=dx, device=device, **s)
    return h


def make_h_random(
    N: int, dx: float, device: str,
    seed: int = 42, amplitude: float = 400e-9, sigma_filter: float = 200e-6,
) -> torch.Tensor:
    """Band-limited random membrane roughness."""
    torch.manual_seed(seed)
    noise  = torch.randn(N, N, dtype=torch.float32, device=device)
    f      = torch.fft.fftfreq(N, d=dx)
    Fx, Fy = torch.meshgrid(f, f, indexing="ij")
    lpf    = torch.exp(-2.0 * (np.pi * sigma_filter)**2 * (Fx**2 + Fy**2))
    nc     = torch.view_as_complex(torch.stack([noise, torch.zeros_like(noise)], -1))
    fil    = torch.fft.ifft2(torch.fft.fft2(nc) * lpf).real
    return (fil / fil.abs().max().clamp(min=1e-30)) * amplitude
