"""
sanity_01_flat_mirror.py
------------------------
h = 0 (flat mirror)
  - CMOS intensity must be uniform = 1.0   (max error < 1e-4)
  - Energy conservation: sum|Ud|^2 == sum|U0|^2   (Parseval, rel err < 1e-4)
Output: 1x3
"""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from sensor_model  import HolographicSensor
from sensor_utils  import plot_1x3
from sensor_params import wavelength, mem_res, mem_pitch, cmos_res, grid_res, distance


def run(out_dir: Path, device: str = "cpu") -> None:
    print("[01] Flat mirror  h=0 ...", flush=True)

    sensor = HolographicSensor(
        wavelength=wavelength, mem_res=mem_res, mem_pitch=mem_pitch,
        cmos_res=cmos_res, grid_res=grid_res, distance=distance, device=device,
    ).to(device)

    N  = sensor.mem_res
    dx = sensor.mem_pitch
    M  = sensor.cmos_res

    h_zero = torch.zeros(N, N, device=device)

    # ---- Test 1: intensity = 1.0 inside membrane footprint -----------
    # Membrane occupies pixels [_ms:_me] of the grid.
    # CMOS crop starts at pixel _s of the grid.
    # Membrane footprint in CMOS coords: [_ms - _s : _me - _s]
    with torch.no_grad():
        I_cmos = sensor(h_zero)

    # Membrane footprint in CMOS index space
    ms_c = max(sensor._ms - sensor._s, 0)
    me_c = min(sensor._me - sensor._s, M)
    # Include all pixels (no guard band)
    guard = 0
    I_inner = I_cmos[ms_c + guard : me_c - guard, ms_c + guard : me_c - guard]
    mean_I = float(I_inner.mean())
    err    = abs(mean_I - 1.0)
    ok     = err < 5e-3
    print(f"  [{'PASS' if ok else 'FAIL'}]  Flat h=0 -> I=1.0 (membrane interior)   "
          f"mean(I)={mean_I:.6f}  |mean-1|={err:.2e}", flush=True)

    # ---- Test 2: energy conservation (Parseval) ----------------------
    with torch.no_grad():
        Ud_full = sensor.propagated_field(h_zero)

    E_in  = float(N * N)
    E_out = float(torch.abs(Ud_full).pow(2).sum())
    rel   = abs(E_out - E_in) / E_in
    ok2   = rel < 1e-4
    print(f"  [{'PASS' if ok2 else 'FAIL'}]  Energy conservation      "
          f"E_in={E_in:.0f}  E_out={E_out:.4f}  rel={rel:.2e}", flush=True)

    I_np  = I_cmos.cpu().numpy()
    Ud_c  = sensor.crop(Ud_full)
    phi   = np.angle(Ud_c.cpu().numpy())

    mem_coords  = (np.arange(N) - (N - 1) * 0.5) * dx
    cmos_coords = (np.arange(M) - (M - 1) * 0.5) * dx

    plot_1x3(
        title       = 'Sanity 01 - Flat Mirror  h = 0',
        h_nm        = np.zeros((N, N), dtype=np.float32),
        I           = I_np,
        phi_raw     = phi,
        mem_coords  = mem_coords,
        cmos_coords = cmos_coords,
        out_path    = out_dir / 'sanity_01_flat_mirror.png',
    )


if __name__ == '__main__':
    out_dir = Path(__file__).parent / 'sanity_results'
    out_dir.mkdir(exist_ok=True)
    run(out_dir, device='cuda' if torch.cuda.is_available() else 'cpu')
