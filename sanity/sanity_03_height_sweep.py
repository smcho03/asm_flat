"""
sanity_03_height_sweep.py
-------------------------
Gaussian bump height sweep  h = 0 .. 5*lambda
  A. 4x5 grid of CMOS zoom images (centre crop)
  B. Modulation curve: I_peak vs h  (shows lambda/2 periodicity)
  C. Waterfall: spatial cross-section vs h (2-D false colour)
"""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import torch

from sensor_model  import HolographicSensor, gaussian_bump
from sensor_params import wavelength, mem_res, mem_pitch, cmos_res, grid_res, distance

STYLE = {
    'figure.facecolor' : '#0d1117',
    'axes.facecolor'   : '#0d1117',
    'text.color'       : '#e6edf3',
    'axes.labelcolor'  : '#e6edf3',
    'xtick.color'      : '#8b949e',
    'ytick.color'      : '#8b949e',
    'axes.titlesize'   : 8,
    'axes.labelsize'   : 7,
    'xtick.labelsize'  : 6,
    'ytick.labelsize'  : 6,
    'font.family'      : 'monospace',
}


def run(out_dir: Path, device: str = "cpu") -> None:
    print("[03] Height sweep ...", flush=True)

    sensor = HolographicSensor(
        wavelength=wavelength, mem_res=mem_res, mem_pitch=mem_pitch,
        cmos_res=cmos_res, grid_res=grid_res, distance=distance, device=device,
    ).to(device)

    N   = sensor.mem_res
    dx  = sensor.mem_pitch
    M   = sensor.cmos_res
    lam = sensor.lam

    n_steps    = 20
    h_values   = np.linspace(0, 5 * lam, n_steps)   # [m]
    zoom_px    = 60     # half-width of CMOS zoom for the grid panels

    ax_c     = (np.arange(M) - (M - 1) * 0.5) * dx * 1e6   # [um]
    ctr      = (M - 1) // 2
    sc, ec   = max(0, ctr - zoom_px), min(M, ctr + zoom_px + 1)
    ax_zoom  = ax_c[sc:ec]

    images   = []
    I_peaks  = []
    profiles = []

    for h_val in h_values:
        h = gaussian_bump(N=N, dx=dx, amplitude=float(h_val), sigma=100e-6, device=device)
        with torch.no_grad():
            I = sensor(h).cpu().numpy()
        images.append(I[sc:ec, sc:ec])
        I_peaks.append(float(I.max()))
        profiles.append(I[ctr, :])   # central row

    profiles = np.stack(profiles, axis=0)   # [n_steps, M]

    # ---- Figure A: 4x5 grid -----------------------------------------
    plt.rcParams.update(STYLE)
    fig_a, axes_a = plt.subplots(4, 5, figsize=(16, 13),
                                 constrained_layout=True,
                                 facecolor='#0d1117')
    fig_a.suptitle('Sanity 03 - Height Sweep  (Gaussian bump, CMOS centre zoom)',
                   fontsize=10, color='#e6edf3')

    vmax_all = max(im.max() for im in images)
    ext_z = [ax_zoom[0], ax_zoom[-1], ax_zoom[-1], ax_zoom[0]]

    for idx, ax in enumerate(axes_a.flat):
        ax.set_facecolor('#0d1117')
        im = ax.imshow(images[idx], extent=ext_z, origin='upper',
                       cmap='inferno', vmin=0, vmax=vmax_all,
                       aspect='auto', interpolation='nearest')
        ax.set_title(f'h = {h_values[idx]*1e9:.0f} nm', color='#e6edf3', fontsize=7)
        ax.tick_params(colors='#8b949e', labelsize=5)
        for sp in ax.spines.values():
            sp.set_edgecolor('#30363d')

    plt.savefig(out_dir / 'sanity_03a_sweep_grid.png',
                dpi=130, facecolor='#0d1117', bbox_inches='tight')
    plt.close()
    print("  saved -> sanity_03a_sweep_grid.png", flush=True)

    # ---- Figure B: modulation curve ---------------------------------
    plt.rcParams.update(STYLE)
    fig_b, ax_b = plt.subplots(figsize=(8, 4), facecolor='#0d1117')
    ax_b.set_facecolor('#0d1117')
    h_nm = h_values * 1e9
    ax_b.plot(h_nm, I_peaks, color='#58a6ff', linewidth=1.5, marker='o',
              markersize=3, label='I_peak')
    for k in range(1, 6):
        ax_b.axvline(k * lam / 2 * 1e9, color='#f78166', linewidth=0.8,
                     linestyle='--', alpha=0.6)
    ax_b.set_xlabel('h [nm]')
    ax_b.set_ylabel('Peak intensity [a.u.]')
    ax_b.set_title('Sanity 03B - I_peak vs h  (lambda/2 periodicity)',
                   color='#e6edf3')
    ax_b.grid(True, color='#30363d', linewidth=0.4, linestyle=':')
    for sp in ax_b.spines.values():
        sp.set_edgecolor('#30363d')
    ax_b.legend(fontsize=7, labelcolor='#e6edf3', facecolor='#0d1117')
    plt.tight_layout()
    plt.savefig(out_dir / 'sanity_03b_modulation_curve.png',
                dpi=130, facecolor='#0d1117', bbox_inches='tight')
    plt.close()
    print("  saved -> sanity_03b_modulation_curve.png", flush=True)

    # ---- Figure C: waterfall ----------------------------------------
    plt.rcParams.update(STYLE)
    fig_c, ax_c2 = plt.subplots(figsize=(10, 5), facecolor='#0d1117')
    ax_c2.set_facecolor('#0d1117')
    ext_w = [ax_c[0], ax_c[-1], h_values[-1] * 1e9, h_values[0] * 1e9]
    im_w  = ax_c2.imshow(profiles, extent=ext_w, origin='upper',
                         cmap='inferno', aspect='auto',
                         norm=Normalize(vmin=0, vmax=float(profiles.max())))
    cb = fig_c.colorbar(im_w, ax=ax_c2, fraction=0.025, pad=0.02)
    cb.set_label('I [a.u.]', color='#8b949e', fontsize=7)
    cb.ax.yaxis.set_tick_params(color='#8b949e', labelsize=6)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='#8b949e')
    ax_c2.set_xlabel("y' [um]")
    ax_c2.set_ylabel('h [nm]')
    ax_c2.set_title('Sanity 03C - Waterfall: I(y\') vs h', color='#e6edf3')
    ax_c2.grid(True, color='#30363d', linewidth=0.4, linestyle=':')
    for sp in ax_c2.spines.values():
        sp.set_edgecolor('#30363d')
    plt.tight_layout()
    plt.savefig(out_dir / 'sanity_03c_waterfall.png',
                dpi=130, facecolor='#0d1117', bbox_inches='tight')
    plt.close()
    print("  saved -> sanity_03c_waterfall.png", flush=True)


if __name__ == '__main__':
    out_dir = Path(__file__).parent / 'sanity_results'
    out_dir.mkdir(exist_ok=True)
    run(out_dir, device='cuda' if torch.cuda.is_available() else 'cpu')
