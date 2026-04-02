"""
sensor_utils.py
---------------
Shared dark-theme visualization helpers for the holographic tactile sensor.

Axis convention (matching rs_sanity):
  h panel   : xlabel = y [um],  ylabel = x [um]
  CMOS panel: xlabel = y'[um],  ylabel = x'[um]

Phase display policy:
  - Individual phase : unwrap_phase(angle(U))
  - delta_phi        : angle(U_def * conj(U_ref)) -> unwrap_phase
    (simple subtraction phi_def - phi_ref is forbidden: gives [-2pi, 2pi]
     which confuses the unwrapper)
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
from skimage.restoration import unwrap_phase

STYLE = {
    'figure.facecolor' : '#0d1117',
    'axes.facecolor'   : '#0d1117',
    'text.color'       : '#e6edf3',
    'axes.labelcolor'  : '#e6edf3',
    'xtick.color'      : '#8b949e',
    'ytick.color'      : '#8b949e',
    'axes.titlesize'   : 9,
    'axes.labelsize'   : 8,
    'xtick.labelsize'  : 7,
    'ytick.labelsize'  : 7,
    'font.family'      : 'monospace',
}


# ---- Internal helpers ------------------------------------------------

def _cb(fig, ax, im, label):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(label, color='#8b949e', fontsize=7)
    cb.ax.yaxis.set_tick_params(color='#8b949e', labelsize=6)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='#8b949e')


def _grid(ax):
    ax.grid(True, color='#30363d', linewidth=0.4, linestyle=':')
    for sp in ax.spines.values():
        sp.set_edgecolor('#30363d')


# ---- Public plot functions -------------------------------------------

def plot_h(fig, ax, h_nm: np.ndarray, mem_coords: np.ndarray, title: str) -> None:
    """
    2-D false-colour image of membrane height h [nm].

    Parameters
    ----------
    h_nm       : [N, N] float array, height in nm, indexing [x, y]
    mem_coords : [N] float array, physical coordinates in metres
    """
    c_um = mem_coords * 1e6
    vmin = min(float(h_nm.min()), -1e-6)
    vmax = max(float(h_nm.max()),  1e-6)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    ext  = [c_um.min(), c_um.max(), c_um.min(), c_um.max()]

    # Transpose: rows of image = y, cols = x  (xlabel=y, ylabel=x)
    im = ax.imshow(h_nm.T, extent=ext, origin='lower',
                   aspect='auto', cmap='RdBu', norm=norm)
    _cb(fig, ax, im, 'h [nm]')
    ax.set_title(title)
    ax.set_xlabel('y [um]')
    ax.set_ylabel('x [um]')
    _grid(ax)


def plot_intensity(
    fig, ax,
    I: np.ndarray,
    cmos_coords: np.ndarray,
    title: str,
    vmax: float | None = None,
) -> None:
    """
    2-D false-colour image of CMOS intensity I [a.u.].

    Parameters
    ----------
    I           : [M, M] float array, indexing [x', y']
    cmos_coords : [M] float array, physical coordinates in metres
    """
    c_um = cmos_coords * 1e6
    ext  = [c_um.min(), c_um.max(), c_um.min(), c_um.max()]
    im   = ax.imshow(I.T, extent=ext, origin='lower',
                     aspect='auto', cmap='inferno',
                     vmin=0, vmax=vmax if vmax is not None else float(I.max()))
    _cb(fig, ax, im, 'I [a.u.]')
    ax.set_title(title)
    ax.set_xlabel("y' [um]")
    ax.set_ylabel("x' [um]")
    _grid(ax)


def plot_phase(
    fig, ax,
    phi_raw: np.ndarray,
    cmos_coords: np.ndarray,
    title: str,
    sym: bool = False,
) -> None:
    """
    2-D phase image (with 2-D unwrapping).

    Parameters
    ----------
    phi_raw     : [M, M] float, must be in [-pi, pi] (i.e. np.angle(U))
                  For delta_phi pass np.angle(U_def * conj(U_ref)).
    cmos_coords : [M] float array in metres
    sym         : True -> RdBu symmetric colormap (use for delta_phi)
    """
    c_um = cmos_coords * 1e6
    ext  = [c_um.min(), c_um.max(), c_um.min(), c_um.max()]

    phi  = unwrap_phase(phi_raw.astype(np.float64)).astype(np.float32)
    span = float(phi.max() - phi.min())
    vmin_d, vmax_d = float(phi.min()), float(phi.max())

    if sym:
        abs_max = max(abs(vmin_d), abs(vmax_d), 1e-12)
        cmap, vmin, vmax = 'RdBu', -abs_max, abs_max
        cb_label = 'delta_phi [rad]  (unwrapped)'
    elif span <= 2 * np.pi + 1e-6:
        cmap, vmin, vmax = 'hsv', -np.pi, np.pi
        cb_label = 'phi [rad]  (<=2pi, hsv)'
    else:
        cmap, vmin, vmax = 'viridis', vmin_d, vmax_d
        cb_label = f'phi [rad]  (unwrapped, span={span:.1f})'

    im = ax.imshow(phi.T, extent=ext, origin='lower',
                   aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)

    if cmap == 'hsv':
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                          ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        cb.ax.set_yticklabels(['-pi', '-pi/2', '0', '+pi/2', '+pi'],
                              color='#8b949e', fontsize=6)
        cb.set_label(cb_label, color='#8b949e', fontsize=7)
    else:
        _cb(fig, ax, im, cb_label)

    ax.set_title(title)
    ax.set_xlabel("y' [um]")
    ax.set_ylabel("x' [um]")
    _grid(ax)


# ---- Composite figures -----------------------------------------------

def plot_1x3(
    title       : str,
    h_nm        : np.ndarray,
    I           : np.ndarray,
    phi_raw     : np.ndarray,
    mem_coords  : np.ndarray,
    cmos_coords : np.ndarray,
    out_path    : Path,
) -> None:
    """
    1 row x 3 cols:  h(x,y)  |  CMOS Intensity  |  CMOS Phase (unwrapped)

    phi_raw = np.angle(U_CMOS)  — pass raw angle, unwrapping done inside.
    """
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(title, fontsize=11, color='#e6edf3')
    fig.patch.set_facecolor('#0d1117')

    plot_h(fig, axes[0], h_nm, mem_coords, 'Surface h(x,y)')
    plot_intensity(fig, axes[1], I, cmos_coords, 'CMOS Intensity')
    plot_phase(fig, axes[2], phi_raw, cmos_coords, 'CMOS Phase  (unwrapped)')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"  saved -> {out_path.name}")


def plot_3x3(
    title         : str,
    h_ref_nm      : np.ndarray,
    h_def_nm      : np.ndarray,
    I_ref         : np.ndarray,
    I_def         : np.ndarray,
    U_ref_cmos    : np.ndarray,   # complex, [M, M]
    U_def_cmos    : np.ndarray,   # complex, [M, M]
    mem_coords    : np.ndarray,
    cmos_coords   : np.ndarray,
    out_path      : Path,
    cmos_zoom_um  : float | None = None,  # if set, zoom CMOS panels to +-N um
    h_zoom_um     : float | None = None,  # if set, zoom h panels to +-N um
) -> None:
    """
    3 rows x 3 cols:
      row 0: h=0 (ref)     | I_ref    | phi_ref  (unwrapped)
      row 1: h_def         | I_def    | phi_def  (unwrapped)
      row 2: delta_h       | delta_I  | delta_phi (complex product -> unwrap)

    delta_phi = angle(U_def * conj(U_ref))  -> unwrap
      (NOT phi_def - phi_ref, which gives [-2pi,2pi] and confuses unwrapper)
    """
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    fig.suptitle(title, fontsize=11, color='#e6edf3')
    fig.patch.set_facecolor('#0d1117')

    phi_ref      = np.angle(U_ref_cmos).astype(np.float32)
    phi_def      = np.angle(U_def_cmos).astype(np.float32)
    dphi_wrapped = np.angle(U_def_cmos * np.conj(U_ref_cmos)).astype(np.float32)

    I_max = max(float(I_ref.max()), float(I_def.max()))
    dI    = I_def - I_ref
    dh_nm = h_def_nm - h_ref_nm

    # ---- optional zoom: crop cmos arrays to centre window -----------
    def _crop(arr2d, coords, zoom_um):
        """Return (cropped_arr, cropped_coords) limited to +-zoom_um."""
        if zoom_um is None:
            return arr2d, coords
        c_um  = coords * 1e6
        mask  = np.abs(c_um) <= zoom_um
        idx   = np.where(mask)[0]
        s, e  = idx[0], idx[-1] + 1
        return arr2d[s:e, s:e], coords[s:e]

    c_I_ref,  cc_I = _crop(I_ref,        cmos_coords, cmos_zoom_um)
    c_I_def,  _    = _crop(I_def,        cmos_coords, cmos_zoom_um)
    c_dI,     _    = _crop(dI,           cmos_coords, cmos_zoom_um)
    c_phi_ref, _   = _crop(phi_ref,      cmos_coords, cmos_zoom_um)
    c_phi_def, _   = _crop(phi_def,      cmos_coords, cmos_zoom_um)
    c_dphi,    _   = _crop(dphi_wrapped, cmos_coords, cmos_zoom_um)

    c_hR, hc_h = _crop(h_ref_nm, mem_coords, h_zoom_um)
    c_hD, _    = _crop(h_def_nm, mem_coords, h_zoom_um)
    c_dh, _    = _crop(dh_nm,    mem_coords, h_zoom_um)

    I_max_z = max(float(c_I_ref.max()), float(c_I_def.max()))

    # ---- row 0: reference -------------------------------------------
    plot_h(fig, axes[0, 0], c_hR, hc_h, 'h = 0  (reference)')
    plot_intensity(fig, axes[0, 1], c_I_ref, cc_I,
                   'CMOS Intensity  (ref)', vmax=I_max_z)
    plot_phase(fig, axes[0, 2], c_phi_ref, cc_I,
               'CMOS Phase  (ref, unwrapped)')

    # ---- row 1: deformed --------------------------------------------
    plot_h(fig, axes[1, 0], c_hD, hc_h, 'h  (deformed)')
    plot_intensity(fig, axes[1, 1], c_I_def, cc_I,
                   'CMOS Intensity  (def)', vmax=I_max_z)
    plot_phase(fig, axes[1, 2], c_phi_def, cc_I,
               'CMOS Phase  (def, unwrapped)')

    # ---- row 2: difference ------------------------------------------
    plot_h(fig, axes[2, 0], c_dh, hc_h, 'delta_h  (def - ref)')

    c_um   = cc_I * 1e6
    ext    = [c_um.min(), c_um.max(), c_um.min(), c_um.max()]
    dI_min = min(float(c_dI.min()), -1e-20)
    dI_max = max(float(c_dI.max()),  1e-20)
    ax     = axes[2, 1]
    im     = ax.imshow(c_dI.T, extent=ext, origin='lower', aspect='auto',
                       cmap='RdBu',
                       norm=TwoSlopeNorm(vmin=dI_min, vcenter=0, vmax=dI_max))
    _cb(fig, ax, im, 'dI [a.u.]')
    ax.set_title('delta_I  (def - ref)')
    ax.set_xlabel("y' [um]")
    ax.set_ylabel("x' [um]")
    _grid(ax)

    plot_phase(fig, axes[2, 2], c_dphi, cc_I,
               'delta_phi  (unwrapped)', sym=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"  saved -> {out_path.name}")
