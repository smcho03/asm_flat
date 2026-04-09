"""
sanity_05_animation.py
----------------------
Animated GIF: CMOS response as Gaussian bump height h sweeps 0 -> 5*lambda

4-panel layout per frame:
  [h map]  |  [CMOS Intensity]  |  [delta_I]  |  [delta_phi]

Outputs (sanity_results/):
  sanity_05_anim_single_bump.gif   -- sigma=100um (tight zoom, shows rings)
  sanity_05_anim_random.gif        -- band-limited random pattern scaled by h
"""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import TwoSlopeNorm, Normalize
from skimage.restoration import unwrap_phase
import torch

from sensor_model  import HolographicSensor, gaussian_bump, make_h_random
from sensor_params import wavelength, mem_res, mem_pitch, cmos_res, grid_res, distance

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

def _grid(ax):
    ax.grid(True, color='#30363d', linewidth=0.4, linestyle=':')
    for sp in ax.spines.values():
        sp.set_edgecolor('#30363d')

def _cb(fig, ax, im, label):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(label, color='#8b949e', fontsize=7)
    cb.ax.yaxis.set_tick_params(color='#8b949e', labelsize=6)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='#8b949e')
    return cb

def _crop(arr2d, coords, zoom_um):
    """Centre-crop arr2d and coords to +-zoom_um. Returns (arr, coords)."""
    if zoom_um is None:
        return arr2d, coords
    mask = np.abs(coords * 1e6) <= zoom_um
    idx  = np.where(mask)[0]
    s, e = idx[0], idx[-1] + 1
    return arr2d[s:e, s:e], coords[s:e]

def _unwrap(phi_raw):
    return unwrap_phase(phi_raw.astype(np.float64)).astype(np.float32)


# ======================================================================
def make_animation(
    out_path     : Path,
    sensor       : HolographicSensor,
    h_shapes     : list,          # list of (h_tensor, label_str) at each frame
    h_max_nm     : float,         # for h colorbar fixed range
    cmos_zoom_um : float | None,  # CMOS zoom half-width [um]
    h_zoom_um    : float | None,  # h map zoom half-width [um]
    fps          : int = 10,
    title        : str = "",
    device       : str = "cpu",
) -> None:
    N  = sensor.mem_res
    dx = sensor.mem_pitch
    M  = sensor.cmos_res

    mem_coords  = (np.arange(N) - (N - 1) * 0.5) * dx
    cmos_coords = (np.arange(M) - (M - 1) * 0.5) * dx

    # ---- Precompute reference (h=0) ---------------------------------
    h_ref = torch.zeros(N, N, device=device)
    with torch.no_grad():
        Ud_ref = sensor.propagated_field(h_ref)
    I_ref      = sensor.crop(torch.abs(Ud_ref).pow(2)).cpu().numpy()
    U_ref_cmos = sensor.crop(Ud_ref).cpu().numpy()

    # ---- Pre-render all frames' data --------------------------------
    print(f"  Computing {len(h_shapes)} frames ...", flush=True)
    frames_data = []
    for h_def, label in h_shapes:
        with torch.no_grad():
            Ud_def = sensor.propagated_field(h_def)
        I_def      = sensor.crop(torch.abs(Ud_def).pow(2)).cpu().numpy()
        U_def_cmos = sensor.crop(Ud_def).cpu().numpy()
        h_nm       = h_def.cpu().numpy() * 1e9
        dI         = I_def - I_ref
        dphi_raw   = np.angle(U_def_cmos * np.conj(U_ref_cmos)).astype(np.float32)
        frames_data.append((h_nm, I_def, dI, dphi_raw, label))

    # ---- Cropped coordinate helpers ---------------------------------
    _, cc    = _crop(I_ref, cmos_coords, cmos_zoom_um)
    _, hc    = _crop(frames_data[0][0], mem_coords, h_zoom_um)
    c_um     = cc * 1e6
    h_um     = hc * 1e6
    ext_c    = [c_um.min(), c_um.max(), c_um.min(), c_um.max()]
    ext_h    = [h_um.min(), h_um.max(), h_um.min(), h_um.max()]

    # Fixed colourmap limits computed across all frames
    all_I   = np.stack([_crop(d[1], cmos_coords, cmos_zoom_um)[0] for d in frames_data])
    all_dI  = np.stack([_crop(d[2], cmos_coords, cmos_zoom_um)[0] for d in frames_data])
    I_vmax  = float(np.percentile(all_I, 99.5))
    dI_abs  = float(np.percentile(np.abs(all_dI), 99.5))
    dI_abs  = max(dI_abs, 1e-12)

    # delta_phi range: use 99th percentile of unwrapped magnitudes
    all_dphi = []
    for d in frames_data:
        c_dp, _ = _crop(d[3], cmos_coords, cmos_zoom_um)
        all_dphi.append(_unwrap(c_dp))
    dphi_abs = float(np.percentile([np.abs(p).max() for p in all_dphi], 95))
    dphi_abs = max(dphi_abs, 1e-12)

    # ---- Build figure -----------------------------------------------
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), facecolor='#0d1117',
                             constrained_layout=True)
    sup = fig.suptitle(title, fontsize=11, color='#e6edf3')

    # Initial dummy images (will be updated each frame)
    dummy_h = np.zeros((len(hc), len(hc)))
    dummy_c = np.zeros((len(cc), len(cc)))

    im0 = axes[0].imshow(dummy_h.T, extent=ext_h, origin='lower', aspect='auto',
                         cmap='RdBu',
                         norm=TwoSlopeNorm(vmin=-h_max_nm, vcenter=0, vmax=h_max_nm))
    axes[0].set_title('h(x,y)'); axes[0].set_xlabel('y [um]'); axes[0].set_ylabel('x [um]')
    _grid(axes[0]); _cb(fig, axes[0], im0, 'h [nm]')

    im1 = axes[1].imshow(dummy_c.T, extent=ext_c, origin='lower', aspect='auto',
                         cmap='inferno', vmin=0, vmax=I_vmax)
    axes[1].set_title('CMOS Intensity'); axes[1].set_xlabel("y' [um]"); axes[1].set_ylabel("x' [um]")
    _grid(axes[1]); _cb(fig, axes[1], im1, 'I [a.u.]')

    im2 = axes[2].imshow(dummy_c.T, extent=ext_c, origin='lower', aspect='auto',
                         cmap='RdBu',
                         norm=TwoSlopeNorm(vmin=-dI_abs, vcenter=0, vmax=dI_abs))
    axes[2].set_title('delta_I  (def - ref)'); axes[2].set_xlabel("y' [um]"); axes[2].set_ylabel("x' [um]")
    _grid(axes[2]); _cb(fig, axes[2], im2, 'dI [a.u.]')

    im3 = axes[3].imshow(dummy_c.T, extent=ext_c, origin='lower', aspect='auto',
                         cmap='RdBu', vmin=-dphi_abs, vmax=dphi_abs)
    axes[3].set_title('delta_phi  (unwrapped)'); axes[3].set_xlabel("y' [um]"); axes[3].set_ylabel("x' [um]")
    _grid(axes[3]); _cb(fig, axes[3], im3, 'dphi [rad]')

    for ax in axes:
        ax.set_facecolor('#0d1117')

    # ---- Update function --------------------------------------------
    def update(frame_idx):
        h_nm, I_def, dI, dphi_raw, label = frames_data[frame_idx]

        c_h,  _ = _crop(h_nm,    mem_coords,  h_zoom_um)
        c_I,  _ = _crop(I_def,   cmos_coords, cmos_zoom_um)
        c_dI, _ = _crop(dI,      cmos_coords, cmos_zoom_um)
        c_dp, _ = _crop(dphi_raw, cmos_coords, cmos_zoom_um)
        c_dp_uw = _unwrap(c_dp)

        im0.set_data(c_h.T)
        im1.set_data(c_I.T)
        im2.set_data(c_dI.T)
        im3.set_data(c_dp_uw.T)
        sup.set_text(f"{title}   |   {label}")
        return im0, im1, im2, im3, sup

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames_data),
        interval=1000 // fps, blit=False,
    )

    print(f"  Saving GIF -> {out_path.name} ...", flush=True)
    writer = animation.PillowWriter(fps=fps)
    anim.save(str(out_path), writer=writer, dpi=100,
              savefig_kwargs=dict(facecolor='#0d1117'))
    plt.close()
    print(f"  saved -> {out_path.name}", flush=True)


# ======================================================================
def run(out_dir: Path, device: str = "cpu") -> None:
    print("[05] Animation ...", flush=True)

    sensor = HolographicSensor(
        wavelength=wavelength, mem_res=mem_res, mem_pitch=mem_pitch,
        cmos_res=cmos_res, grid_res=grid_res, distance=distance, device=device,
    ).to(device)

    N   = sensor.mem_res
    dx  = sensor.mem_pitch
    lam = sensor.lam

    n_frames  = 40
    h_values  = np.linspace(0, 5 * lam, n_frames)  # 0 .. 5*lambda

    # ---- Animation 1: Single Gaussian bump --------------------------
    print("  [single_bump] building frames ...", flush=True)
    h_shapes_bump = []
    for h_val in h_values:
        h = gaussian_bump(N=N, dx=dx, amplitude=float(h_val),
                          sigma=100e-6, device=device)
        label = f"h_max = {h_val*1e9:.1f} nm  ({h_val/lam:.2f} lambda)"
        h_shapes_bump.append((h, label))

    make_animation(
        out_path     = out_dir / 'sanity_05_anim_single_bump.gif',
        sensor       = sensor,
        h_shapes     = h_shapes_bump,
        h_max_nm     = float(5 * lam * 1e9),
        cmos_zoom_um = 600.0,
        h_zoom_um    = 400.0,
        fps          = 10,
        title        = 'Single Gaussian Bump  (sigma=100um)  --  h sweep 0->5*lambda',
        device       = device,
    )

    # ---- Animation 2: Random pattern scaled by h --------------------
    print("  [random] building frames ...", flush=True)
    h_base = make_h_random(N=N, dx=dx, device=device,
                           amplitude=1.0)   # normalised to 1.0 max
    h_shapes_rnd = []
    for h_val in h_values:
        h     = h_base * float(h_val)
        label = f"h_max = {h_val*1e9:.1f} nm  ({h_val/lam:.2f} lambda)"
        h_shapes_rnd.append((h, label))

    make_animation(
        out_path     = out_dir / 'sanity_05_anim_random.gif',
        sensor       = sensor,
        h_shapes     = h_shapes_rnd,
        h_max_nm     = float(5 * lam * 1e9),
        cmos_zoom_um = None,
        h_zoom_um    = None,
        fps          = 10,
        title        = 'Random Band-limited  (sigma_filter=200um)  --  h sweep 0->5*lambda',
        device       = device,
    )


if __name__ == '__main__':
    out_dir = Path(__file__).parent / 'sanity_results'
    out_dir.mkdir(exist_ok=True)
    run(out_dir, device='cuda' if torch.cuda.is_available() else 'cpu')
