"""
sanity_02_symmetry.py
---------------------
  1. Symmetry     : symmetric h -> symmetric I  (x-flip, y-flip)
  2. Phase period : uniform h = lambda/2 -> 2*pi phase shift -> I = 1.0
  3. Gradient     : loss.backward() gives finite, non-NaN gradients
  4. Shift        : off-centre bump shifts the CMOS pattern accordingly
Output: 1x3 for gradient test case
"""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from sensor_model  import HolographicSensor, gaussian_bump
from sensor_utils  import plot_1x3
from sensor_params import wavelength, mem_res, mem_pitch, cmos_res, grid_res, distance


def run(out_dir: Path, device: str = "cpu") -> None:
    print("[02] Symmetry & phase periodicity ...", flush=True)

    sensor = HolographicSensor(
        wavelength=wavelength, mem_res=mem_res, mem_pitch=mem_pitch,
        cmos_res=cmos_res, grid_res=grid_res, distance=distance, device=device,
    ).to(device)

    N  = sensor.mem_res
    dx = sensor.mem_pitch
    M  = sensor.cmos_res
    lam = sensor.lam

    mem_coords  = (np.arange(N) - (N - 1) * 0.5) * dx
    cmos_coords = (np.arange(M) - (M - 1) * 0.5) * dx

    # ---- Test 1: Symmetry -------------------------------------------
    h_bump = gaussian_bump(N=N, dx=dx, amplitude=500e-9, sigma=100e-6, device=device)
    with torch.no_grad():
        I_bump = sensor(h_bump)

    err_x = float((I_bump - I_bump.flip(0)).abs().max())
    err_y = float((I_bump - I_bump.flip(1)).abs().max())
    ok1   = err_x < 1e-4 and err_y < 1e-4
    print(f"  [{'PASS' if ok1 else 'FAIL'}]  Symmetry (Gaussian bump)  "
          f"flip_x err={err_x:.2e}  flip_y err={err_y:.2e}", flush=True)

    # ---- Test 2: Phase periodicity ----------------------------------
    # h = lambda/2 -> phi = 4*pi/lam * lam/2 = 2*pi -> exp(i*2*pi) = 1 -> I = 1
    # Check only inside the membrane footprint on the CMOS.
    h_half = torch.full((N, N), lam / 2.0, device=device)
    with torch.no_grad():
        I_half = sensor(h_half)

    ms_c  = max(sensor._ms - sensor._s, 0)
    me_c  = min(sensor._me - sensor._s, M)
    guard = 5
    I_inner2 = I_half[ms_c + guard : me_c - guard, ms_c + guard : me_c - guard]
    mean_I2  = float(I_inner2.mean())
    err2     = abs(mean_I2 - 1.0)
    ok2      = err2 < 5e-3
    print(f"  [{'PASS' if ok2 else 'FAIL'}]  Phase period  h=lam/2 -> I=1 (membrane interior)  "
          f"mean(I)={mean_I2:.6f}  |mean-1|={err2:.2e}", flush=True)

    # ---- Test 3: Gradient flow --------------------------------------
    h_grad = gaussian_bump(N=N, dx=dx, amplitude=500e-9, sigma=100e-6, device=device)
    h_grad.requires_grad_(True)
    I_g = sensor(h_grad)
    I_g.mean().backward()
    g   = h_grad.grad
    ok3 = (g is not None) and not g.isnan().any() and not g.isinf().any()
    g_max = float(g.abs().max()) if g is not None else float('nan')
    print(f"  [{'PASS' if ok3 else 'FAIL'}]  Gradient flow             "
          f"grad_max={g_max:.2e}  nan={g.isnan().any().item() if g is not None else True}",
          flush=True)

    # ---- Test 4: Off-centre shift -----------------------------------
    shift_m = 500e-6   # 500 um in y-direction
    h_ctr   = gaussian_bump(N=N, dx=dx, amplitude=500e-9, sigma=100e-6, device=device)
    h_sft   = gaussian_bump(N=N, dx=dx, amplitude=500e-9, sigma=100e-6,
                            center=(0.0, shift_m), device=device)
    h_flat = torch.zeros(N, N, device=device)
    with torch.no_grad():
        I_flat = sensor(h_flat).cpu().numpy()
        I_ctr  = sensor(h_ctr).cpu().numpy()
        I_sft  = sensor(h_sft).cpu().numpy()

    # Convex mirror: bump creates a dip -> argmin of delta_I tracks bump location
    dI_ctr   = I_ctr - I_flat
    dI_sft   = I_sft - I_flat
    peak_ctr = int(dI_ctr.argmin() % M)
    peak_sft = int(dI_sft.argmin() % M)
    delta_px  = peak_sft - peak_ctr
    delta_um  = delta_px * dx * 1e6
    ok4 = abs(delta_um - shift_m * 1e6) < 5 * dx * 1e6
    print(f"  [{'PASS' if ok4 else 'FAIL'}]  Off-centre shift          "
          f"expected={shift_m*1e6:.0f}um  measured={delta_um:.1f}um", flush=True)

    # ---- Save 1x3 for the gradient-test bump ------------------------
    with torch.no_grad():
        Ud_full = sensor.propagated_field(h_bump)
    Ud_c  = sensor.crop(Ud_full)

    plot_1x3(
        title       = 'Sanity 02 - Symmetry check  (Gaussian bump)',
        h_nm        = h_bump.cpu().numpy() * 1e9,
        I           = I_bump.cpu().numpy(),
        phi_raw     = np.angle(Ud_c.cpu().numpy()),
        mem_coords  = mem_coords,
        cmos_coords = cmos_coords,
        out_path    = out_dir / 'sanity_02_symmetry.png',
    )


if __name__ == '__main__':
    out_dir = Path(__file__).parent / 'sanity_results'
    out_dir.mkdir(exist_ok=True)
    run(out_dir, device='cuda' if torch.cuda.is_available() else 'cpu')
