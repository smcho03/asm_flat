"""
light_source_comparison.py
--------------------------
Compare reconstruction quality across different illumination sources
of varying coherence:

  (1) Laser              — single wavelength, single point source     (delta lam = 0)
  (2) Monochromatic LED  — narrow spectrum, narrow source             (~1 nm, 1 angle)
  (3) LED (bare)         — broader spectrum, finite extent            (~10 nm, ~5 angles)
  (4) White LED          — broadband, finite extent                   (~80 nm, ~5 angles)
  (5) Three-color LED    — RGB, photometric-stereo style              (3 channels, no fringe)

Forward model
-------------
For partially coherent illumination, total intensity is the incoherent
sum over independent realizations:
  I_total = (1/N) sum_(lam, theta) |U(h ; lam, theta)|^2
A plane-wave at angle (kx, ky) is implemented as a tilt phase ramp
prepended to the membrane field; the same band-limited ASM kernel is
re-built per wavelength.

For (5), the LEDs are decomposed into three Lambertian intensity
channels (no fringe at all) and reconstruction takes those scalars.

Outputs
-------
  output/light_source_comparison/
    summary.png
    summary.txt
    summary.json
"""

import sys, time, json, functools
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print = functools.partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).parent.parent))
from sensor_model  import HolographicSensor, gaussian_bump
from sensor_params import wavelength, mem_pitch
from sensor_utils  import STYLE

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "output" / "light_source_comparison"
OUT.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

# ----------------------------------------------------------------------
# Sensor + scene
# ----------------------------------------------------------------------
MEM   = 128
CMOS  = 256
GRID  = 384
DIST  = 5e-3
AMP   = 200e-9
SIGMA = 100e-6
H_SCALE = 400e-9
LR    = 3e-3
N_ITER = 1500

dx = mem_pitch

# A laser-baseline sensor for reference
sensor_ref = HolographicSensor(
    wavelength=wavelength, mem_res=MEM, mem_pitch=dx,
    cmos_res=CMOS, grid_res=GRID, distance=DIST, device=device,
).to(device)

h_true = gaussian_bump(N=MEM, dx=dx, amplitude=AMP, sigma=SIGMA, device=device)

# ----------------------------------------------------------------------
# Helpers — partially coherent forward model
# ----------------------------------------------------------------------
def make_sensor(lam):
    return HolographicSensor(
        wavelength=lam, mem_res=MEM, mem_pitch=dx,
        cmos_res=CMOS, grid_res=GRID, distance=DIST, device=device,
    ).to(device)

# cache of sensors per wavelength to avoid rebuilding ASM kernel every iter
_sensor_cache = {}
def get_sensor(lam):
    key = round(lam * 1e12)  # picometre key
    if key not in _sensor_cache:
        _sensor_cache[key] = make_sensor(lam)
    return _sensor_cache[key]

def _tilt_phase(N, dx, theta_x, theta_y, lam, device):
    """Tilted plane wave phase ramp at the membrane plane."""
    coords = (torch.arange(N, dtype=torch.float32, device=device) - (N-1)*0.5) * dx
    X, Y = torch.meshgrid(coords, coords, indexing="ij")
    kx = 2*np.pi/lam * np.sin(theta_x)
    ky = 2*np.pi/lam * np.sin(theta_y)
    return kx * X + ky * Y

def forward_partial_coherent(h, lams, angles_deg, device):
    """
    Sum of intensities over (wavelength, illumination angle) realizations.
    `angles_deg`: list of (theta_x_deg, theta_y_deg)
    """
    acc = None
    n   = len(lams) * len(angles_deg)
    for lam in lams:
        sensor = get_sensor(lam)
        for tx_deg, ty_deg in angles_deg:
            tx = np.deg2rad(tx_deg); ty = np.deg2rad(ty_deg)
            tilt = _tilt_phase(MEM, dx, tx, ty, lam, device)
            A = torch.exp(1j * tilt)
            # Need a complex amplitude path through the sensor:
            # we monkey-patch by injecting A via custom forward.
            pad = (sensor._ms, sensor.grid_res - sensor._me,
                   sensor._ms, sensor.grid_res - sensor._me)
            h_grid = F.pad(h, pad)
            A_grid = F.pad(A, pad)
            phi  = -(4.0*np.pi/sensor.lam) * h_grid
            U0   = A_grid * torch.exp(1j * phi)
            Ud   = torch.fft.ifft2(torch.fft.fft2(U0) * sensor.H)
            I    = torch.abs(Ud).pow(2)
            I    = I[sensor._s:sensor._e, sensor._s:sensor._e]
            acc  = I if acc is None else acc + I
    return acc / n

# ----------------------------------------------------------------------
# Light source definitions
# ----------------------------------------------------------------------
def lams_in_band(center_nm, width_nm, n=5):
    if width_nm < 1e-6:
        return [center_nm * 1e-9]
    return list((np.linspace(center_nm - width_nm/2,
                              center_nm + width_nm/2, n) * 1e-9))

def angles_in_disc(extent_deg, n=5):
    if extent_deg < 1e-6 or n == 1:
        return [(0.0, 0.0)]
    out = [(0.0, 0.0)]
    for k in range(1, n):
        ang = 2*np.pi*k/(n-1)
        out.append((extent_deg*np.cos(ang), extent_deg*np.sin(ang)))
    return out

# Source descriptors: (name, [wavelengths_m], [(theta_x_deg, theta_y_deg)])
sources = {
    "laser_HeNe"     : ("Laser HeNe (632.8 nm, point)",
                         lams_in_band(632.8, 0.0,  n=1),
                         angles_in_disc(0.0,     n=1)),
    "led_mono_pin"   : ("Mono LED + pinhole (632 nm ±0.5 nm, ~point)",
                         lams_in_band(632.0, 1.0,  n=5),
                         angles_in_disc(0.0,     n=1)),
    "led_mono_bare"  : ("Mono LED bare (632 nm ±5 nm, ext 1°)",
                         lams_in_band(632.0, 10.0, n=5),
                         angles_in_disc(1.0,     n=5)),
    "led_white"      : ("White LED (550 nm ±50 nm, ext 1°)",
                         lams_in_band(550.0, 100.0, n=5),
                         angles_in_disc(1.0,     n=5)),
}

# Render targets
print("\nForward rendering ...")
targets = {}
for k, (label, lams, angs) in sources.items():
    t0 = time.time()
    with torch.no_grad():
        I = forward_partial_coherent(h_true, lams, angs, device)
    targets[k] = I
    print(f"  {k:18s} {len(lams)*len(angs):3d} realizations  in {time.time()-t0:.1f}s")

# Photometric Gelsight target — separate, returns 3 channels
# Includes physical Gelsight limits: elastomer blur + camera pixel pitch.
# (Matches gelsight_photometric_compare.py: CAM_PIXEL=30um, BLUR_SIGMA=60um.)
CAM_PIXEL_UM = 30.0
BLUR_SIGMA_UM = 60.0

def photometric_target(h):
    from scipy.ndimage import gaussian_filter
    h_np = h.detach().cpu().numpy()
    gy_h, gx_h = np.gradient(h_np, dx)
    nz_d = np.sqrt(gx_h**2 + gy_h**2 + 1.0)
    nx = -gx_h/nz_d; ny = -gy_h/nz_d; nz = 1.0/nz_d
    th = np.deg2rad(45.0)
    leds = np.array([
        [ np.sin(th), 0.0, np.cos(th)],
        [-np.sin(th)/2,  np.sin(th)*np.sqrt(3)/2, np.cos(th)],
        [-np.sin(th)/2, -np.sin(th)*np.sqrt(3)/2, np.cos(th)],
    ])
    # Render Lambertian intensities
    Is = [np.clip(nx*l[0]+ny*l[1]+nz*l[2], 0, None) for l in leds]
    # Elastomer + optical blur
    blur_px = BLUR_SIGMA_UM * 1e-6 / dx
    Is = [gaussian_filter(I, sigma=blur_px) for I in Is]
    # Downsample to camera pixel pitch (block-average)
    ds = max(int(round(CAM_PIXEL_UM / (dx*1e6))), 1)
    H2 = (MEM // ds) * ds
    Is = [I[:H2, :H2].reshape(H2//ds, ds, H2//ds, ds).mean(axis=(1,3)) for I in Is]
    return Is, leds

I_photo, photo_leds = photometric_target(h_true)
sources["photo_3led"] = ("3-LED Lambertian (cam 30um + blur 60um)", None, None)

# ----------------------------------------------------------------------
# Reconstruction
# ----------------------------------------------------------------------
def reconstruct_partial(forward_fn, I_tgt, n_iter=N_ITER, lr=LR):
    # squared reparam (h >= 0) — removes in-line sign / twin-image ambiguity.
    raw = torch.full((MEM, MEM), 0.2, dtype=torch.float32, device=device,
                     requires_grad=True)
    opt = torch.optim.Adam([raw], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iter)
    losses = []
    for _ in range(n_iter):
        opt.zero_grad()
        h = (raw * raw) * H_SCALE
        I = forward_fn(h)
        loss = torch.mean((I - I_tgt)**2)
        loss.backward(); opt.step(); sched.step()
        losses.append(float(loss.detach()))
    with torch.no_grad():
        h_final = (raw * raw) * H_SCALE
    return h_final, losses

def reconstruct_photometric(I_targets_np, leds, n_iter=2000, lr=1e-2):
    """Gelsight-style photometric stereo: 3-LED Lambertian → normals →
    Frankot-Chellappa integration → h_rec. The intensities are assumed to
    have already been through the camera-pipeline limits (blur+downsample),
    so the native grid here is the camera grid (smaller than MEM).
    We upsample back to MEM×MEM for fair PSNR comparison against h_true."""
    from scipy.ndimage import zoom
    I_stack = np.stack(I_targets_np, axis=-1)  # H_cam, W_cam, 3
    L_pinv = np.linalg.pinv(leds)
    n_est = I_stack @ L_pinv.T
    nrm   = np.linalg.norm(n_est, axis=-1, keepdims=True).clip(min=1e-12)
    n_est = n_est / nrm
    nx = n_est[..., 0]; ny = n_est[..., 1]; nz = n_est[..., 2].clip(min=1e-3)
    gx = -nx/nz; gy = -ny/nz
    H, W = gx.shape
    # gradients are per camera-pixel; convert to per-metre using cam pitch
    cam_dx = CAM_PIXEL_UM * 1e-6
    fx = np.fft.fftfreq(W).reshape(1, W)
    fy = np.fft.fftfreq(H).reshape(H, 1)
    denom = (fx**2 + fy**2); denom[0,0] = 1.0
    Gx = np.fft.fft2(gx); Gy = np.fft.fft2(gy)
    F_ = -1j*(fx*Gx + fy*Gy) / (2*np.pi*denom + 1e-30)
    F_[0, 0] = 0
    z_cam = np.real(np.fft.ifft2(F_)) * cam_dx
    z_cam = z_cam - np.median(z_cam)
    # Upsample (bilinear) back to MEM×MEM so we can compare with h_true
    z_up = zoom(z_cam, (MEM / H, MEM / W), order=1)
    z_up = z_up[:MEM, :MEM]
    if z_up.shape != (MEM, MEM):
        out = np.zeros((MEM, MEM), dtype=z_up.dtype)
        out[:z_up.shape[0], :z_up.shape[1]] = z_up
        z_up = out
    return z_up

def psnr_db(a, b):
    mse = float(torch.mean((a - b)**2))
    peak = float(b.abs().max())
    if mse < 1e-30: return 100.0
    if peak < 1e-30: return 0.0
    return 10.0 * np.log10(peak**2 / mse)

# Run all reconstructions
recons = {}
losses = {}
print("\nReconstructing ...")
for k, (label, lams, angs) in sources.items():
    if k == "photo_3led":
        t0 = time.time()
        h_rec_np = reconstruct_photometric(I_photo, photo_leds)
        h_rec = torch.tensor(h_rec_np, dtype=torch.float32, device=device)
        p = psnr_db(h_rec, h_true)
        recons[k] = h_rec.cpu().numpy()
        losses[k] = None
        print(f"  {k:18s} PSNR={p:6.2f} dB  ({time.time()-t0:.1f}s)")
    else:
        t0 = time.time()
        forward_fn = (lambda lams_=lams, angs_=angs:
                      lambda h: forward_partial_coherent(h, lams_, angs_, device))()
        h_rec, L = reconstruct_partial(forward_fn, targets[k])
        p = psnr_db(h_rec, h_true)
        recons[k] = h_rec.cpu().numpy()
        losses[k] = L
        print(f"  {k:18s} PSNR={p:6.2f} dB  ({time.time()-t0:.1f}s)")

# ----------------------------------------------------------------------
# Fringe visibility (rough)
# ----------------------------------------------------------------------
def visibility(I):
    I = I.detach().cpu().numpy() if hasattr(I, "detach") else I
    Imax = float(I.max()); Imin = float(I.min())
    return (Imax - Imin) / (Imax + Imin + 1e-30)

vis = {k: visibility(targets[k]) for k in targets}

# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
plt.rcParams.update(STYLE)
n_src = len(sources)
fig, axes = plt.subplots(3, n_src, figsize=(4.0*n_src, 11))
fig.suptitle("Light source comparison — coherent → incoherent",
             fontsize=13, color="#e6edf3")
fig.patch.set_facecolor("#0d1117")

def _style(ax):
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e", labelsize=7)
    ax.grid(color="#30363d", lw=0.4, ls=":")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

m_um = (np.arange(MEM) - (MEM-1)*0.5) * dx * 1e6

# Row 0: intensity at sensor (or RGB triple for photometric)
for j, (k, (label, _, _)) in enumerate(sources.items()):
    ax = axes[0, j]
    if k == "photo_3led":
        # Show LED1 channel only
        im = ax.imshow(I_photo[0].T, origin="lower", aspect="equal", cmap="inferno")
        ax.set_title(f"{label}\nLED1 channel  V={visibility(I_photo[0]):.2f}",
                     fontsize=8, color="#e6edf3")
    else:
        I = targets[k].cpu().numpy()
        im = ax.imshow(I.T, origin="lower", aspect="equal", cmap="inferno")
        ax.set_title(f"{label}\nVisibility={vis[k]:.3f}",
                     fontsize=8, color="#e6edf3")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _style(ax)

# Row 1: reconstructed h
h_true_np = h_true.cpu().numpy()*1e9
vm = max(abs(h_true_np).max(), 1.0)
for j, k in enumerate(sources.keys()):
    ax = axes[1, j]
    h_rec_np = recons[k] * 1e9
    im = ax.imshow(h_rec_np.T, origin="lower", aspect="equal",
                   cmap="RdBu", vmin=-vm, vmax=vm)
    p = psnr_db(torch.tensor(recons[k]), h_true)
    ax.set_title(f"h_rec  PSNR={p:.1f} dB", fontsize=9, color="#e6edf3")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _style(ax)

# Row 2: centre cross-section
mid = MEM // 2
for j, k in enumerate(sources.keys()):
    ax = axes[2, j]
    ax.plot(m_um, h_true_np[mid,:], color="#8b949e", lw=2, ls=":", label="GT")
    ax.plot(m_um, recons[k][mid,:]*1e9, color="#58a6ff", lw=1.0, label="rec")
    ax.set_xlabel("x [um]", fontsize=8, color="#8b949e")
    ax.set_ylabel("h [nm]", fontsize=8, color="#8b949e")
    ax.legend(fontsize=7, labelcolor="#e6edf3", framealpha=0.3)
    _style(ax)

plt.tight_layout()
plt.savefig(OUT / "summary.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()

# ----------------------------------------------------------------------
# Text + JSON
# ----------------------------------------------------------------------
results = []
for k, (label, lams, angs) in sources.items():
    if k == "photo_3led":
        n_real = 3
        v = visibility(I_photo[0])
    else:
        n_real = len(lams) * len(angs)
        v = vis[k]
    p = psnr_db(torch.tensor(recons[k]), h_true)
    results.append(dict(name=k, label=label, n_realizations=n_real,
                        visibility=v, psnr_db=p))

txt = ["light_source_comparison.py — partial-coherence sweep",
       "=" * 56,
       f"  Phantom: Gaussian bump A={AMP*1e9:.0f} nm, sigma={SIGMA*1e6:.0f} um",
       f"  Distance={DIST*1e3:.0f} mm  mem={MEM}px@{dx*1e6:.0f}um  cmos={CMOS}",
       f"  Recon: Adam lr={LR}, {N_ITER} iters",
       "",
       f"  {'source':22s}  {'realiz':>7s}  {'visibility':>11s}  {'PSNR':>8s}"]
for r in results:
    txt.append(f"  {r['name']:22s}  {r['n_realizations']:>7d}  "
               f"{r['visibility']:>11.3f}  {r['psnr_db']:>8.2f}")
txt += ["",
        "Reading the table",
        "  * Visibility (Imax-Imin)/(Imax+Imin) measures fringe contrast",
        "    at the sensor.  Coherent -> ~1.  Fully incoherent -> ~0.",
        "  * PSNR drops as visibility drops because phase information is",
        "    averaged out across realizations.",
        "  * 3-LED photometric (Gelsight) skips the diffraction channel",
        "    entirely; recovers smooth shape but no fine fringe info."]
txt = "\n".join(txt)
(OUT / "summary.txt").write_text(txt, encoding="utf-8")
print("\n" + txt)

(OUT / "summary.json").write_text(json.dumps(dict(
    config=dict(amp_nm=AMP*1e9, sigma_um=SIGMA*1e6, distance_mm=DIST*1e3,
                mem=MEM, cmos=CMOS, grid=GRID,
                lr=LR, n_iter=N_ITER, h_scale_nm=H_SCALE*1e9),
    sources=results,
), indent=2))
print(f"\nSaved -> {OUT/'summary.png'}")
