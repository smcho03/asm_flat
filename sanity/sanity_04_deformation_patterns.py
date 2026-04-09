"""
sanity_04_deformation_patterns.py
----------------------------------
5 deformation patterns, each saved as a 3x3 figure:
  1. flat         - h = 0 (baseline)
  2. single_pixel - single pixel at centre  (lambda/4 = 158 nm)
  3. single_bump  - Gaussian  sigma=100 um, h=500 nm
  4. multi_bump   - 5 Gaussian bumps at various positions
  5. random       - band-limited random roughness
"""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from sensor_model  import (HolographicSensor,
                           make_h_flat, make_h_single_pixel,
                           make_h_single_bump, make_h_multi_bump, make_h_random)
from sensor_utils  import plot_3x3
from sensor_params import wavelength, mem_res, mem_pitch, cmos_res, grid_res, distance

# (key, generator, label, cmos_zoom_um, h_zoom_um)
# cmos_zoom_um : half-width of CMOS zoom window [um], None = full view
# h_zoom_um    : half-width of h zoom window [um],    None = full membrane
PATTERNS = [
    ('flat',
     make_h_flat,
     'Flat  (h = 0)',
     None, None),

    ('single_pixel',
     make_h_single_pixel,
     'Single Pixel  (lambda/4, centre)',
     400.0, 50.0),       # Fresnel ring spacing ~56 um -> show +/-400 um

    ('single_bump',
     make_h_single_bump,
     'Single Gaussian Bump  (sigma=100um, h=500nm)',
     600.0, 400.0),      # diffraction spread ~32 um -> zoom to see rings

    ('multi_bump',
     make_h_multi_bump,
     'Multi Bump  (5 bumps)',
     2500.0, 2000.0),    # bumps reach +/-900 um -> show broader field

    ('random',
     make_h_random,
     'Random Band-limited  (sigma_filter=200um)',
     None, None),        # full view to show speckle over whole membrane
]


def run(out_dir: Path, device: str = "cpu") -> None:
    print("[04] Deformation patterns ...", flush=True)

    sensor = HolographicSensor(
        wavelength=wavelength, mem_res=mem_res, mem_pitch=mem_pitch,
        cmos_res=cmos_res, grid_res=grid_res, distance=distance, device=device,
    ).to(device)

    N  = sensor.mem_res
    dx = sensor.mem_pitch
    M  = sensor.cmos_res

    mem_coords  = (np.arange(N) - (N - 1) * 0.5) * dx
    cmos_coords = (np.arange(M) - (M - 1) * 0.5) * dx

    # Reference: flat membrane
    h_ref = make_h_flat(N, dx, device)
    with torch.no_grad():
        Ud_ref_full = sensor.propagated_field(h_ref)
    I_ref      = sensor.crop(torch.abs(Ud_ref_full).pow(2)).cpu().numpy()
    U_ref_cmos = sensor.crop(Ud_ref_full).cpu().numpy()
    h_ref_nm   = h_ref.cpu().numpy() * 1e9

    for key, h_gen, label, cmos_zoom, h_zoom in PATTERNS:
        print(f"  [{key}] ...", flush=True)
        h_def = h_gen(N, dx, device)
        with torch.no_grad():
            Ud_def_full = sensor.propagated_field(h_def)

        I_def      = sensor.crop(torch.abs(Ud_def_full).pow(2)).cpu().numpy()
        U_def_cmos = sensor.crop(Ud_def_full).cpu().numpy()
        h_def_nm   = h_def.cpu().numpy() * 1e9

        plot_3x3(
            title        = f'Sanity 04 - {label}',
            h_ref_nm     = h_ref_nm,
            h_def_nm     = h_def_nm,
            I_ref        = I_ref,
            I_def        = I_def,
            U_ref_cmos   = U_ref_cmos,
            U_def_cmos   = U_def_cmos,
            mem_coords   = mem_coords,
            cmos_coords  = cmos_coords,
            out_path     = out_dir / f'sanity_04_{key}.png',
            cmos_zoom_um = cmos_zoom,
            h_zoom_um    = h_zoom,
        )


if __name__ == '__main__':
    out_dir = Path(__file__).parent / 'sanity_results'
    out_dir.mkdir(exist_ok=True)
    run(out_dir, device='cuda' if torch.cuda.is_available() else 'cpu')
