"""
gelsight_mitsuba_compare.py
---------------------------
Compare holographic tactile sensor reconstruction with CMU's physics-based
GelSight simulation (Mitsuba3 renderer).

Improvements over v1:
  - LED-by-LED rendering (separate R/G/B passes)
  - Exact LED direction vectors from XML geometry
  - Proper least-squares photometric stereo
  - Multiple deformation scales (nm, um, tens-of-um)
  - Spatial resolution matching between render and heightmap

GelSight LED geometry (from XML):
  All LEDs at same radius ~15.4 scene-units from surface, 120 deg apart.
  Surface (elastomer) at y=13.  LED base at y=4.207, z=-12.59.
    Red:   no Y-rotation  → pos ≈ ( 0.0,   4.2, -12.6)
    Green: 240° Y-rotation → pos ≈ ( 10.9,  4.2,   6.3)
    Blue:  120° Y-rotation → pos ≈ (-10.9,  4.2,   6.3)

Dependencies:
  pip install mitsuba==3.3.0
"""

import sys, os, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import zoom as scipy_zoom

sys.path.insert(0, str(Path(__file__).parent.parent))
from sensor_model import HolographicSensor, gaussian_bump
from sensor_params import wavelength, mem_pitch
from sensor_utils import STYLE

# Mitsuba repo (must be ASCII path - Mitsuba3 C++ cannot handle Korean)
MITSUBA_REPO = Path("e:/USER/Documents/tactile_optical_sim")

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

# ======================================================================
#  GelSight LED geometry (computed from XML transforms)
# ======================================================================
# Surface centre in scene coords: (0, 13, 0)
# LED base: translate(0, 4.207, -12.59), then rotate Y by 0/240/120 deg

SURFACE_Y = 13.0

def _led_position(y_rot_deg):
    """LED position after Y-rotation of the base translated position."""
    base = np.array([0.0, 4.207, -12.59])
    theta = np.radians(y_rot_deg)
    Ry = np.array([
        [np.cos(theta),  0, np.sin(theta)],
        [0,              1, 0             ],
        [-np.sin(theta), 0, np.cos(theta)],
    ])
    return Ry @ base

LED_POSITIONS = {
    "red":   _led_position(0),
    "green": _led_position(240),
    "blue":  _led_position(120),
}

def _led_direction(led_name):
    """Unit vector from surface centre to LED (illumination direction)."""
    surface = np.array([0.0, SURFACE_Y, 0.0])
    d = LED_POSITIONS[led_name] - surface
    return d / np.linalg.norm(d)

# Light direction matrix L: [3, 3] - rows are l_red, l_green, l_blue
L_MATRIX = np.stack([_led_direction("red"),
                     _led_direction("green"),
                     _led_direction("blue")], axis=0).astype(np.float64)

print("LED directions (surface → LED):")
for name in ["red", "green", "blue"]:
    d = _led_direction(name)
    print(f"  {name:5s}: ({d[0]:+.3f}, {d[1]:+.3f}, {d[2]:+.3f})")
print(f"L condition number: {np.linalg.cond(L_MATRIX):.2f}")


# ======================================================================
#  Heightmap -> OBJ mesh
# ======================================================================

def heightmap_to_obj(h: np.ndarray, dx: float, filepath: str) -> None:
    """
    Convert heightmap [N, N] in metres to OBJ with XY in [-1,1].
    Z is in the same normalised units as XY.
    """
    N = h.shape[0]
    L_phys = N * dx
    half_L = L_phys / 2.0
    coords = (np.arange(N) - (N - 1) * 0.5) * dx

    with open(filepath, 'w') as f:
        for i in range(N):
            for j in range(N):
                x = coords[i] / half_L
                y = coords[j] / half_L
                z = h[i, j] / half_L  # same scale as XY
                f.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")
        for i in range(N - 1):
            for j in range(N - 1):
                v00 = i * N + j + 1
                v10 = (i + 1) * N + j + 1
                v01 = i * N + (j + 1) + 1
                v11 = (i + 1) * N + (j + 1) + 1
                f.write(f"f {v00} {v10} {v01}\n")
                f.write(f"f {v10} {v11} {v01}\n")


# ======================================================================
#  Mitsuba3 rendering - per-LED
# ======================================================================

def render_per_led(h: np.ndarray, dx: float, name: str,
                   spp: int = 64, res: int = 400) -> dict:
    """
    Render 3 separate images (one per LED) for photometric stereo.

    Returns dict with keys 'red', 'green', 'blue', each [H, W] float array.
    """
    import mitsuba as mi
    mi.set_variant("scalar_rgb")

    # Save mesh
    mesh_dir = MITSUBA_REPO / "models" / "meshes"
    obj_path = mesh_dir / f"custom_{name}.obj"
    heightmap_to_obj(h, dx, str(obj_path))

    model_fn = str(MITSUBA_REPO / "models" / "flatgel_with_mesh.xml")
    mi.Thread.thread().file_resolver().append(str(MITSUBA_REPO / "models"))
    mi.Thread.thread().file_resolver().append(str(mesh_dir))

    # Original RGB intensity values from all_lights.xml
    INTENSITIES = {
        "red":   {"redIntensity": "5.237216,0,0",
                  "greenIntensity": "0,0,0",
                  "blueIntensity": "0,0,0"},
        "green": {"redIntensity": "0,0,0",
                  "greenIntensity": "0.171378,6.734968,0.0",
                  "blueIntensity": "0,0,0"},
        "blue":  {"redIntensity": "0,0,0",
                  "greenIntensity": "0,0,0",
                  "blueIntensity": "0,0,6.839277"},
    }

    images = {}
    for led, overrides in INTENSITIES.items():
        scene = mi.load_file(
            model_fn,
            num_samples=spp, num_bounces=5,
            resW=res, resH=res,
            cropW=res, cropH=res,
            cropX=0, cropY=0,
            hfName=f"custom_{name}",
            **overrides,
        )
        img = np.array(mi.render(scene, spp=spp))

        # Extract the relevant channel
        if led == "red":
            images[led] = img[:, :, 0].astype(np.float64)
        elif led == "green":
            images[led] = img[:, :, 1].astype(np.float64)
        elif led == "blue":
            images[led] = img[:, :, 2].astype(np.float64)

    return images


# ======================================================================
#  Photometric stereo with known light directions
# ======================================================================

def photometric_stereo(images: dict, dx_render: float) -> np.ndarray:
    """
    Recover height from 3 LED images using proper photometric stereo.

    1. Normalise images to remove LED intensity / albedo imbalance
    2. Stack images -> I_vec = [I_r, I_g, I_b] per pixel
    3. Solve n_tilde = L^{-1} * I_vec  (albedo * normal)
    4. Normalise -> unit normal
    5. Map scene-space gradients to image-space axes
    6. Frankot-Chellappa integration -> height

    Camera coordinate mapping (from perspective_cam.xml):
      Camera looks in +Y direction, up = (0,0,-1)
      Image axis 0 (rows, top->bottom) -> scene +Z
      Image axis 1 (cols, left->right) -> scene -X

    Returns height map [H, W] in arbitrary units (needs calibration).
    """
    H, W = images["red"].shape

    # --- Intensity normalisation ---
    # LED intensities and channel albedos differ, which corrupts PS normals.
    # Dividing each image by its mean equalises the effective scaling.
    I_r = images["red"]   / np.mean(images["red"]).clip(min=1e-12)
    I_g = images["green"] / np.mean(images["green"]).clip(min=1e-12)
    I_b = images["blue"]  / np.mean(images["blue"]).clip(min=1e-12)

    L_inv = np.linalg.inv(L_MATRIX)  # [3, 3]

    # Stack: [3, H*W]
    I_stack = np.stack([I_r.flatten(),
                        I_g.flatten(),
                        I_b.flatten()], axis=0)

    # n_tilde = L_inv @ I  -> [3, H*W]
    n_tilde = L_inv @ I_stack

    # Normalise per pixel
    norms = np.linalg.norm(n_tilde, axis=0, keepdims=True).clip(min=1e-12)
    normals = (n_tilde / norms).reshape(3, H, W)

    # normals[0] = n_x, normals[1] = n_y (flat-surface direction), normals[2] = n_z
    # In GelSight coords, flat normal is (0, -1, 0) -> ny is NEGATIVE
    ny = normals[1]
    ny = np.where(np.abs(ny) > 1e-6, ny, -1e-6)  # preserve sign, avoid div-by-zero

    # Surface: scene_y = g(scene_x, scene_z).
    # Normal (towards camera, -y): n propto (dg/dx, -1, dg/dz)
    #   => dg/dx = -nx/ny,  dg/dz = -nz/ny
    #
    # Physical height: h_cam = 13 - g  (positive for bump)
    #   dh_cam/d(scene_x) = -dg/dx =  nx/ny
    #   dh_cam/d(scene_z) = -dg/dz =  nz/ny
    #
    # Image axes (from camera XML):
    #   axis 0 (rows, top->bottom) -> scene +Z
    #   axis 1 (cols, left->right) -> scene -X
    #
    # FC integration needs p = dh/d(row_dir), q = dh/d(col_dir):
    #   p = dh_cam/d(+scene_z) =  nz/ny
    #   q = dh_cam/d(-scene_x) = -nx/ny
    p =  normals[2] / ny   # nz/ny = dh/d(row) = dh/d(scene_z)
    q = -normals[0] / ny   # -nx/ny = dh/d(col) = dh/d(-scene_x)

    # Frankot-Chellappa integration
    fx = np.fft.fftfreq(H, d=dx_render)
    fy = np.fft.fftfreq(W, d=dx_render)
    Fx, Fy = np.meshgrid(fx, fy, indexing="ij")

    P = np.fft.fft2(p)
    Q = np.fft.fft2(q)

    denom = (2j * np.pi * Fx)**2 + (2j * np.pi * Fy)**2
    denom[0, 0] = 1.0

    Z = (2j * np.pi * Fx * P + 2j * np.pi * Fy * Q) / denom
    Z[0, 0] = 0.0

    h_rec = np.real(np.fft.ifft2(Z))
    return h_rec.astype(np.float32)


# ======================================================================
#  Holographic reconstruction
# ======================================================================

MEM_RES  = 128
CMOS_RES = 256
GRID_RES = 384
N_ITER   = 2000
LR       = 5e-10
HOLO_DIST = 5e-3

def holographic_reconstruct(h_true_t: torch.Tensor):
    sensor = HolographicSensor(
        wavelength=wavelength, mem_res=MEM_RES, mem_pitch=mem_pitch,
        cmos_res=CMOS_RES, grid_res=GRID_RES, distance=HOLO_DIST, device=device,
    ).to(device)

    with torch.no_grad():
        I_target = sensor(h_true_t)

    h_pred = torch.zeros(MEM_RES, MEM_RES, dtype=torch.float32,
                         device=device, requires_grad=True)
    opt = torch.optim.Adam([h_pred], lr=LR)

    losses = []
    for i in range(N_ITER):
        opt.zero_grad()
        I_pred = sensor(h_pred)
        loss = torch.mean((I_pred - I_target)**2)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))

    return h_pred.detach().cpu().numpy(), losses


# ======================================================================
#  Metrics
# ======================================================================

def rmse(a, b):
    return float(np.sqrt(np.mean((a - b)**2)))

def psnr_db(a, b):
    mse_val = float(np.mean((a - b)**2))
    peak = float(np.abs(b).max())
    if mse_val < 1e-30: return 100.0
    return 10.0 * np.log10(peak**2 / mse_val)

def normalised_cross_correlation(a, b):
    """NCC - shape similarity independent of amplitude, in [-1, 1]."""
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a**2) * np.sum(b**2))
    if denom < 1e-30: return 0.0
    return float(np.sum(a * b) / denom)


# ======================================================================
#  Test heightmaps
# ======================================================================

def make_h(name: str, amp: float, N: int = MEM_RES, dx: float = mem_pitch):
    coords = (np.arange(N, dtype=np.float32) - (N-1)*0.5) * dx
    X, Y = np.meshgrid(coords, coords, indexing="ij")
    if name == "single_bump":
        return amp * np.exp(-(X**2 + Y**2) / (2*(150e-6)**2))
    elif name == "multi_bump":
        h = np.zeros((N, N), dtype=np.float32)
        for a, s, cx, cy in [(1.0, 80e-6, 0, 0),
                              (0.8, 100e-6, 300e-6, 200e-6),
                              (0.6, 60e-6, -250e-6, -300e-6)]:
            h += a * np.exp(-((X-cx)**2 + (Y-cy)**2) / (2*s**2))
        return h * (amp / max(abs(h).max(), 1e-30))
    elif name == "fine_grating":
        period = 40e-6
        env = np.exp(-(X**2 + Y**2) / (2*(400e-6)**2))
        return amp * env * (0.5 + 0.5 * np.cos(2*np.pi*X/period))
    else:
        raise ValueError(name)


# ======================================================================
#  Main comparison
# ======================================================================

print("\n" + "="*60)
print("  GelSight (Mitsuba3, per-LED) vs Holographic Sensor")
print("="*60)

# Test at multiple amplitude scales
CONFIGS = [
    # (def_type, amplitude, note)
    ("single_bump", 200e-9,  "holographic regime, sub-wavelength"),
    ("single_bump", 10e-6,   "both regimes, micrometer scale"),
    ("single_bump", 50e-6,   "GelSight regime, tens of um"),
    ("multi_bump",  10e-6,   "multi-bump at um scale"),
    ("fine_grating", 10e-6,  "grating, tests resolution limit"),
]

results = []

for def_type, amp, note in CONFIGS:
    amp_str = f"{amp*1e9:.0f}nm" if amp < 1e-6 else f"{amp*1e6:.0f}um"
    label = f"{def_type} {amp_str}"
    print(f"\n--- {label} ({note}) ---")

    h_true = make_h(def_type, amp)

    # Holographic
    print("  Holographic ...", end="", flush=True)
    t0 = time.time()
    h_true_t = torch.tensor(h_true, device=device)
    h_holo, losses = holographic_reconstruct(h_true_t)
    dt_h = time.time() - t0
    p_h = psnr_db(h_holo, h_true)
    ncc_h = normalised_cross_correlation(h_holo, h_true)
    print(f"  PSNR={p_h:.1f}dB  NCC={ncc_h:.3f}  ({dt_h:.1f}s)")

    # GelSight Mitsuba3
    print("  GelSight (per-LED render) ...", end="", flush=True)
    t0 = time.time()
    try:
        led_images = render_per_led(h_true, mem_pitch, f"{def_type}_{amp_str}",
                                     spp=64, res=400)

        # --- Geometry: locate mesh in the rendered image ---
        # Camera: origin (-1.1029, -28.05, -1.3235), target (-1.1029, 8.65, -1.3235)
        # Camera looks in +Y, up=(0,0,-1), right=(-1,0,0)
        # Surface at y=13, camera distance = 13 - (-28.05) = 41.05
        # FOV 53.3 deg on larger axis (400 px square)
        cam_dist = 41.05
        fov_rad = np.radians(53.3)
        half_width_scene = cam_dist * np.tan(fov_rad / 2)  # half-FOV width in scene units
        scene_per_px = 2 * half_width_scene / 400  # scene units per pixel

        # Mesh spans [-20, 20] in scene X and Z (after scale 20)
        mesh_half = 20.0  # scene units
        mesh_px = int(round(2 * mesh_half / scene_per_px))  # pixels covered by mesh

        # Camera target is at scene (-1.1029, ?, -1.3235), mesh centre at (0, 13, 0)
        # Offset in scene coords: mesh_centre - cam_target (at surface plane)
        # Image col -> scene -X, so scene X offset -> col offset (negated)
        # Image row -> scene +Z, so scene Z offset -> row offset
        offset_x_scene = 0.0 - (-1.1029)  # = +1.1029
        offset_z_scene = 0.0 - (-1.3235)  # = +1.3235
        offset_col = -offset_x_scene / scene_per_px  # scene -X -> col
        offset_row = offset_z_scene / scene_per_px   # scene +Z -> row

        # Mesh centre in pixel coords (relative to image centre)
        cx = 200 + offset_col  # column of mesh centre
        cy = 200 + offset_row  # row of mesh centre

        # Physical pixel pitch for FC integration
        phys_size = MEM_RES * mem_pitch  # 1.28 mm
        dx_render = phys_size / mesh_px

        h_gs_full = photometric_stereo(led_images, dx_render)

        # Crop mesh region (row, col)
        H_r, W_r = h_gs_full.shape
        r0 = int(round(cy - mesh_px / 2))
        c0 = int(round(cx - mesh_px / 2))
        r0 = max(0, min(r0, H_r - mesh_px))
        c0 = max(0, min(c0, W_r - mesh_px))
        h_gs_crop = h_gs_full[r0:r0+mesh_px, c0:c0+mesh_px]

        # Resize to MEM_RES
        scale_factor = MEM_RES / h_gs_crop.shape[0]
        h_gs = scipy_zoom(h_gs_crop, scale_factor)[:MEM_RES, :MEM_RES]

        # Scale to match GT amplitude (calibration)
        if abs(h_gs).max() > 1e-30:
            h_gs = h_gs * (amp / abs(h_gs).max())

        p_gs = psnr_db(h_gs, h_true)
        ncc_gs = normalised_cross_correlation(h_gs, h_true)
        gs_ok = True
    except Exception as e:
        print(f"\n    FAILED: {e}")
        import traceback; traceback.print_exc()
        h_gs = np.zeros_like(h_true)
        led_images = None
        p_gs, ncc_gs = 0.0, 0.0
        gs_ok = False

    dt_gs = time.time() - t0
    if gs_ok:
        print(f"  PSNR={p_gs:.1f}dB  NCC={ncc_gs:.3f}  ({dt_gs:.1f}s)")

    results.append(dict(
        def_type=def_type, amp=amp, amp_str=amp_str, label=label, note=note,
        h_true=h_true,
        h_holo=h_holo, losses=losses, psnr_holo=p_h, ncc_holo=ncc_h,
        h_gs=h_gs, led_images=led_images, psnr_gs=p_gs, ncc_gs=ncc_gs,
        gs_ok=gs_ok,
    ))


# ======================================================================
#  Visualization
# ======================================================================

plt.rcParams.update(STYLE)
m_um = (np.arange(MEM_RES) - (MEM_RES-1)*0.5) * mem_pitch * 1e6
ext = [m_um[0], m_um[-1], m_um[0], m_um[-1]]
mid = MEM_RES // 2

n = len(results)

def _style(ax):
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e", labelsize=6)
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

# --- Fig 1: Full comparison grid ---
fig, axes = plt.subplots(n, 5, figsize=(25, 4*n), squeeze=False)
fig.suptitle("GelSight (Mitsuba3, per-LED PS) vs Holographic - Fixed Pipeline",
             fontsize=14, color="#e6edf3")
fig.patch.set_facecolor("#0d1117")

for row, r in enumerate(results):
    # Determine display units and scale
    if r["amp"] < 1e-6:
        unit = "nm"
        scale = 1e9
    else:
        unit = "μm"
        scale = 1e6

    ht = r["h_true"] * scale
    hh = r["h_holo"] * scale
    hg = r["h_gs"] * scale
    vmax = max(abs(ht).max(), 1e-6)

    # Col 0: GT
    ax = axes[row, 0]
    im = ax.imshow(ht.T, extent=ext, origin="lower", cmap="RdBu",
                   vmin=-vmax, vmax=vmax, aspect="equal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"GT: {r['label']}", fontsize=8, color="#e6edf3")
    _style(ax)

    # Col 1: Holographic
    ax = axes[row, 1]
    im = ax.imshow(hh.T, extent=ext, origin="lower", cmap="RdBu",
                   vmin=-vmax, vmax=vmax, aspect="equal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Holo PSNR={r['psnr_holo']:.1f} NCC={r['ncc_holo']:.3f}",
                 fontsize=7, color="#58a6ff")
    _style(ax)

    # Col 2: GelSight LED images (composite)
    ax = axes[row, 2]
    if r["led_images"] is not None:
        composite = np.stack([
            r["led_images"]["red"] / max(r["led_images"]["red"].max(), 1e-10),
            r["led_images"]["green"] / max(r["led_images"]["green"].max(), 1e-10),
            r["led_images"]["blue"] / max(r["led_images"]["blue"].max(), 1e-10),
        ], axis=-1)
        ax.imshow(np.clip(composite, 0, 1))
        ax.set_title("GelSight LED images (RGB)", fontsize=7, color="#3fb950")
    else:
        ax.text(0.5, 0.5, "FAILED", transform=ax.transAxes,
                ha="center", color="#f85149")
    _style(ax)

    # Col 3: GelSight recon
    ax = axes[row, 3]
    im = ax.imshow(hg.T, extent=ext, origin="lower", cmap="RdBu",
                   vmin=-vmax, vmax=vmax, aspect="equal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    t = f"GS PSNR={r['psnr_gs']:.1f} NCC={r['ncc_gs']:.3f}" if r["gs_ok"] else "FAILED"
    ax.set_title(t, fontsize=7, color="#3fb950" if r["gs_ok"] else "#f85149")
    _style(ax)

    # Col 4: Cross-section
    ax = axes[row, 4]
    ax.plot(m_um, ht[mid, :], color="#8b949e", lw=2, ls=":", label="GT")
    ax.plot(m_um, hh[mid, :], color="#58a6ff", lw=1.5, label="Holographic")
    if r["gs_ok"]:
        ax.plot(m_um, hg[mid, :], color="#3fb950", lw=1.5, label="GelSight")
    ax.set_xlabel(f"x [μm]", fontsize=7, color="#8b949e")
    ax.set_ylabel(f"h [{unit}]", fontsize=7, color="#8b949e")
    ax.set_title("Cross-section", fontsize=8, color="#e6edf3")
    ax.legend(fontsize=6, labelcolor="#e6edf3", framealpha=0.3)
    ax.grid(color="#30363d", lw=0.4, ls=":")
    _style(ax)

plt.tight_layout()
plt.savefig(OUT / "mitsuba_gelsight_vs_holographic.png", dpi=150,
            bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"\nSaved -> {OUT / 'mitsuba_gelsight_vs_holographic.png'}")


# --- Fig 2: Summary bars ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Summary: Holographic vs GelSight (Mitsuba3, proper PS)",
             fontsize=13, color="#e6edf3")
fig.patch.set_facecolor("#0d1117")

labels = [r["label"] for r in results]
x = np.arange(len(labels))
w = 0.35

# PSNR
ax = axes[0]
ax.set_facecolor("#0d1117")
ax.bar(x - w/2, [r["psnr_holo"] for r in results], w,
       label="Holographic", color="#58a6ff", edgecolor="#30363d")
ax.bar(x + w/2, [r["psnr_gs"] for r in results], w,
       label="GelSight", color="#3fb950", edgecolor="#30363d")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=5, color="#8b949e", rotation=25, ha="right")
ax.set_ylabel("PSNR [dB]", fontsize=8, color="#8b949e")
ax.set_title("PSNR (higher=better)", fontsize=9, color="#e6edf3")
ax.legend(fontsize=7, labelcolor="#e6edf3", framealpha=0.3)
ax.grid(color="#30363d", lw=0.4, ls=":", axis="y")
_style(ax)

# NCC
ax = axes[1]
ax.set_facecolor("#0d1117")
ax.bar(x - w/2, [r["ncc_holo"] for r in results], w,
       label="Holographic", color="#58a6ff", edgecolor="#30363d")
ax.bar(x + w/2, [r["ncc_gs"] for r in results], w,
       label="GelSight", color="#3fb950", edgecolor="#30363d")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=5, color="#8b949e", rotation=25, ha="right")
ax.set_ylabel("NCC", fontsize=8, color="#8b949e")
ax.set_title("Normalised Cross-Correlation (higher=better)", fontsize=9, color="#e6edf3")
ax.legend(fontsize=7, labelcolor="#e6edf3", framealpha=0.3)
ax.grid(color="#30363d", lw=0.4, ls=":", axis="y")
_style(ax)

# Advantage chart (holographic PSNR - GelSight PSNR)
ax = axes[2]
ax.set_facecolor("#0d1117")
advantage = [r["psnr_holo"] - r["psnr_gs"] for r in results]
colors = ["#58a6ff" if a > 0 else "#3fb950" for a in advantage]
ax.bar(x, advantage, color=colors, edgecolor="#30363d")
ax.axhline(0, color="#8b949e", lw=0.5)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=5, color="#8b949e", rotation=25, ha="right")
ax.set_ylabel("ΔPSNR [dB]", fontsize=8, color="#8b949e")
ax.set_title("Holographic advantage (blue=holo wins)", fontsize=9, color="#e6edf3")
ax.grid(color="#30363d", lw=0.4, ls=":", axis="y")
_style(ax)

plt.tight_layout()
plt.savefig(OUT / "mitsuba_comparison_summary.png", dpi=150,
            bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"Saved -> {OUT / 'mitsuba_comparison_summary.png'}")


# --- Print summary table ---
print("\n" + "="*80)
print(f"{'Config':<25s} {'Holo PSNR':>10s} {'GS PSNR':>10s} {'Holo NCC':>10s} {'GS NCC':>10s}")
print("-"*80)
for r in results:
    gs_p = f"{r['psnr_gs']:.1f}" if r['gs_ok'] else "FAIL"
    gs_n = f"{r['ncc_gs']:.3f}" if r['gs_ok'] else "FAIL"
    print(f"{r['label']:<25s} {r['psnr_holo']:>10.1f} {gs_p:>10s} "
          f"{r['ncc_holo']:>10.3f} {gs_n:>10s}")
print("="*80)
print("\nDone.")
