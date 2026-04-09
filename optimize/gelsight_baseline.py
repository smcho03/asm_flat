"""
gelsight_baseline.py
--------------------
Simulate a GelSight-style LED + lens tactile sensor and compare
reconstruction quality with the lensless holographic approach.

GelSight physics:
  1. Elastomer membrane with reflective coating
  2. Multiple LEDs illuminate from different angles
  3. Camera with lens captures image -> diffraction-limited PSF
  4. Photometric stereo: recover surface normals from multi-LED images
  5. Integrate normals -> height map

Key limitation: the lens introduces a diffraction limit.
  Rayleigh resolution = 1.22 * lambda / (2 * NA)
  For typical GelSight: NA ~ 0.1-0.2, lambda ~ 550nm (green LED)
  -> resolution ~ 1.7-3.4 um per pixel (but often lens-limited to ~20-50um)

We compare:
  - GelSight (photometric stereo, diffraction-limited)
  - Holographic (gradient descent from coherent intensity pattern)

Output:
  output/gelsight_vs_holographic.png  – side-by-side comparison
"""

import sys, os, time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).parent.parent))
from sensor_model import HolographicSensor, gaussian_bump, make_h_multi_bump, make_h_random
from sensor_params import wavelength, mem_pitch
from sensor_utils import STYLE

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

# ======================================================================
#  GelSight Forward Model
# ======================================================================

class GelSightSensor:
    """
    Simplified GelSight forward model.

    Physics:
      - Lambertian surface with surface normals derived from h(x,y)
      - LED illumination from K directions (unit vectors)
      - Lens PSF modelled as Gaussian blur (diffraction limit)
      - Camera captures intensity = albedo * max(n . l_k, 0)

    Parameters
    ----------
    N : int          - pixel count of membrane (square)
    dx : float       - pixel pitch [m]
    led_wavelength : float - LED wavelength [m] (for diffraction calc)
    lens_NA : float  - numerical aperture of imaging lens
    lens_mag : float - magnification (determines effective pixel size)
    n_leds : int     - number of LED directions
    noise_std : float - sensor noise std (fraction of max intensity)
    """

    def __init__(
        self,
        N: int = 256,
        dx: float = 10e-6,
        led_wavelength: float = 550e-9,    # green LED
        lens_NA: float = 0.12,             # typical GelSight lens
        lens_mag: float = 1.0,
        n_leds: int = 4,
        noise_std: float = 0.005,
    ):
        self.N = N
        self.dx = dx
        self.led_wavelength = led_wavelength
        self.lens_NA = lens_NA
        self.lens_mag = lens_mag
        self.n_leds = n_leds
        self.noise_std = noise_std

        # Rayleigh resolution: 0.61 * lambda / NA  (radius of Airy disk)
        self.rayleigh_res = 0.61 * led_wavelength / lens_NA  # [m]
        # Convert to sigma for Gaussian PSF (Airy ~ Gaussian with sigma ~ 0.42 * rayleigh)
        self.psf_sigma_m = 0.42 * self.rayleigh_res
        self.psf_sigma_px = self.psf_sigma_m / dx

        # LED directions: elevated from grazing, spread azimuthally
        self.led_dirs = self._make_led_directions(n_leds)

        print(f"GelSight sensor:")
        print(f"  LED wavelength  : {led_wavelength*1e9:.0f} nm")
        print(f"  Lens NA         : {lens_NA}")
        print(f"  Rayleigh res    : {self.rayleigh_res*1e6:.2f} um")
        print(f"  PSF sigma       : {self.psf_sigma_px:.2f} px ({self.psf_sigma_m*1e6:.2f} um)")
        print(f"  N LEDs          : {n_leds}")

    def _make_led_directions(self, K):
        """K LED directions at elevation ~30 deg, evenly spaced in azimuth."""
        elevation = np.radians(30)
        dirs = []
        for i in range(K):
            azimuth = 2 * np.pi * i / K
            lx = np.cos(elevation) * np.cos(azimuth)
            ly = np.cos(elevation) * np.sin(azimuth)
            lz = np.sin(elevation)
            dirs.append(np.array([lx, ly, lz], dtype=np.float32))
        return dirs

    def surface_normals(self, h: np.ndarray) -> np.ndarray:
        """
        Compute surface normals from height map h [N, N] in metres.
        Returns normals [N, N, 3] (unit vectors).
        """
        # Central differences
        dhdx = np.zeros_like(h)
        dhdy = np.zeros_like(h)
        dhdx[1:-1, :] = (h[2:, :] - h[:-2, :]) / (2 * self.dx)
        dhdy[:, 1:-1] = (h[:, 2:] - h[:, :-2]) / (2 * self.dx)
        # Boundary: forward/backward
        dhdx[0, :]  = (h[1, :] - h[0, :]) / self.dx
        dhdx[-1, :] = (h[-1, :] - h[-2, :]) / self.dx
        dhdy[:, 0]  = (h[:, 1] - h[:, 0]) / self.dx
        dhdy[:, -1] = (h[:, -1] - h[:, -2]) / self.dx

        # Normal = (-dh/dx, -dh/dy, 1) normalised
        normals = np.stack([-dhdx, -dhdy, np.ones_like(h)], axis=-1)
        norms = np.linalg.norm(normals, axis=-1, keepdims=True)
        normals /= norms.clip(min=1e-12)
        return normals

    def forward(self, h: np.ndarray):
        """
        Generate K LED images from height map h.
        Each image = diffraction-limited capture of Lambertian shading.

        Returns list of K images [N, N], each in [0, 1].
        """
        normals = self.surface_normals(h)
        images = []
        for l_dir in self.led_dirs:
            # Lambertian: I = max(n . l, 0)
            shading = np.clip(np.sum(normals * l_dir, axis=-1), 0, None)

            # Apply lens PSF (diffraction limit) via Gaussian blur
            blurred = gaussian_filter(shading, sigma=self.psf_sigma_px)

            # Add sensor noise
            noisy = blurred + self.noise_std * np.random.randn(*blurred.shape)
            noisy = np.clip(noisy, 0, None)
            images.append(noisy.astype(np.float32))

        return images

    def photometric_stereo(self, images) -> np.ndarray:
        """
        Recover surface normals from K LED images via least-squares
        photometric stereo.

        I_k(x,y) = albedo * n(x,y) . l_k

        Returns normals [N, N, 3].
        """
        K = len(images)
        N = images[0].shape[0]

        # L matrix: [K, 3]
        L = np.stack(self.led_dirs, axis=0)  # [K, 3]

        # I matrix: [K, N*N]
        I_mat = np.stack([img.flatten() for img in images], axis=0)  # [K, N*N]

        # Least squares: n_tilde = (L^T L)^{-1} L^T I
        # n_tilde = albedo * n
        LtL_inv = np.linalg.inv(L.T @ L)
        n_tilde = LtL_inv @ L.T @ I_mat  # [3, N*N]

        # Normalise to get unit normals (albedo cancels)
        norms = np.linalg.norm(n_tilde, axis=0, keepdims=True).clip(min=1e-12)
        normals = (n_tilde / norms).T.reshape(N, N, 3)

        return normals.astype(np.float32)

    def integrate_normals(self, normals: np.ndarray) -> np.ndarray:
        """
        Integrate normals to recover height via Frankot-Chellappa (1988).
        Frequency-domain integration of (p, q) = (-nx/nz, -ny/nz).
        """
        nz = normals[:, :, 2].clip(min=1e-6)
        p = -normals[:, :, 0] / nz   # dh/dx
        q = -normals[:, :, 1] / nz   # dh/dy

        N = p.shape[0]
        fx = np.fft.fftfreq(N, d=self.dx)
        fy = np.fft.fftfreq(N, d=self.dx)
        Fx, Fy = np.meshgrid(fx, fy, indexing="ij")

        P = np.fft.fft2(p)
        Q = np.fft.fft2(q)

        # Frankot-Chellappa: H = (j*fx*P + j*fy*Q) / (fx^2 + fy^2)
        denom = (2j * np.pi * Fx)**2 + (2j * np.pi * Fy)**2
        denom[0, 0] = 1.0  # avoid division by zero at DC

        H = (2j * np.pi * Fx * P + 2j * np.pi * Fy * Q) / denom
        H[0, 0] = 0.0  # zero mean

        h_rec = np.real(np.fft.ifft2(H))
        return h_rec.astype(np.float32)

    def reconstruct(self, h: np.ndarray) -> np.ndarray:
        """Full pipeline: forward -> photometric stereo -> integrate."""
        images = self.forward(h)
        normals = self.photometric_stereo(images)
        h_rec = self.integrate_normals(normals)
        return h_rec


# ======================================================================
#  Holographic reconstruction (same as reconstruct_sweep)
# ======================================================================

MEM_RES  = 128
CMOS_RES = 256
GRID_RES = 384
N_ITER   = 500
LR       = 5e-10
LAM_TV   = 1e-20

def holographic_reconstruct(h_true_t: torch.Tensor, distance: float):
    sensor = HolographicSensor(
        wavelength=wavelength, mem_res=MEM_RES, mem_pitch=mem_pitch,
        cmos_res=CMOS_RES, grid_res=GRID_RES, distance=distance, device=device,
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
        mse = torch.mean((I_pred - I_target)**2)
        tv = (torch.mean(torch.abs(h_pred[1:, :] - h_pred[:-1, :])) +
              torch.mean(torch.abs(h_pred[:, 1:] - h_pred[:, :-1])))
        loss = mse + LAM_TV * tv
        loss.backward()
        opt.step()
        losses.append(float(loss))

    return h_pred.detach().cpu().numpy(), losses


# ======================================================================
#  Metrics
# ======================================================================

def rmse_nm(h_pred, h_true):
    return float(np.sqrt(np.mean((h_pred - h_true)**2))) * 1e9

def psnr_db(h_pred, h_true):
    mse = float(np.mean((h_pred - h_true)**2))
    peak = float(np.abs(h_true).max())
    if mse < 1e-30:
        return 100.0
    return 10.0 * np.log10(peak**2 / mse)


# ======================================================================
#  Comparison experiments
# ======================================================================

print("\n" + "="*60)
print("  GelSight vs Holographic Tactile Sensor - Baseline Comparison")
print("="*60)

# Test deformations
def make_h_np(def_type, amplitude, N=MEM_RES, dx=mem_pitch):
    if def_type == "single_bump":
        coords = (np.arange(N, dtype=np.float32) - (N-1)*0.5) * dx
        X, Y = np.meshgrid(coords, coords, indexing="ij")
        return amplitude * np.exp(-(X**2 + Y**2) / (2*(150e-6)**2))
    elif def_type == "multi_bump":
        h = np.zeros((N, N), dtype=np.float32)
        specs = [
            (1.0,  80e-6, 0, 0),
            (0.8, 120e-6, 300e-6, 200e-6),
            (0.6,  60e-6, -250e-6, -300e-6),
        ]
        coords = (np.arange(N, dtype=np.float32) - (N-1)*0.5) * dx
        X, Y = np.meshgrid(coords, coords, indexing="ij")
        for a, s, cx, cy in specs:
            h += a * np.exp(-((X-cx)**2 + (Y-cy)**2) / (2*s**2))
        h = h * (amplitude / max(abs(h).max(), 1e-30))
        return h
    elif def_type == "fine_grating":
        # Fine periodic structure to test diffraction limit
        coords = (np.arange(N, dtype=np.float32) - (N-1)*0.5) * dx
        X, Y = np.meshgrid(coords, coords, indexing="ij")
        period = 30e-6  # 30 um period — near GelSight diffraction limit
        envelope = np.exp(-(X**2 + Y**2) / (2*(500e-6)**2))
        return amplitude * envelope * (0.5 + 0.5 * np.cos(2*np.pi*X/period))
    else:
        raise ValueError(f"Unknown: {def_type}")


# Configurations to compare
CONFIGS = [
    ("single_bump",  200e-9),
    ("multi_bump",   200e-9),
    ("fine_grating", 200e-9),  # tests diffraction limit
]

# GelSight sensor variants (different NA -> different resolution)
GELSIGHT_NAs = [0.05, 0.12, 0.25]

results_gs = {}  # (def_type, NA) -> dict
results_holo = {}  # def_type -> dict

HOLO_DISTANCE = 5e-3

for def_type, amp in CONFIGS:
    print(f"\n--- Deformation: {def_type}  amp={amp*1e9:.0f}nm ---")
    h_true_np = make_h_np(def_type, amp)

    # Holographic reconstruction
    print("  Holographic ...", end="", flush=True)
    t0 = time.time()
    h_true_t = torch.tensor(h_true_np, device=device)
    h_holo, losses_holo = holographic_reconstruct(h_true_t, HOLO_DISTANCE)
    dt = time.time() - t0
    r_holo = rmse_nm(h_holo, h_true_np)
    p_holo = psnr_db(h_holo, h_true_np)
    print(f"  RMSE={r_holo:.2f}nm  PSNR={p_holo:.1f}dB  ({dt:.1f}s)")
    results_holo[def_type] = dict(
        h_pred=h_holo, losses=losses_holo, rmse=r_holo, psnr=p_holo,
        h_true=h_true_np,
    )

    # GelSight for each NA
    for NA in GELSIGHT_NAs:
        print(f"  GelSight NA={NA} ...", end="", flush=True)
        t0 = time.time()
        gs = GelSightSensor(N=MEM_RES, dx=mem_pitch, lens_NA=NA)
        h_gs = gs.reconstruct(h_true_np)
        dt = time.time() - t0
        r_gs = rmse_nm(h_gs, h_true_np)
        p_gs = psnr_db(h_gs, h_true_np)
        print(f"  RMSE={r_gs:.2f}nm  PSNR={p_gs:.1f}dB  ({dt:.1f}s)")
        results_gs[(def_type, NA)] = dict(
            h_pred=h_gs, rmse=r_gs, psnr=p_gs,
            rayleigh=gs.rayleigh_res, psf_sigma=gs.psf_sigma_m,
        )


# ======================================================================
#  Visualization
# ======================================================================

plt.rcParams.update(STYLE)
m_um = (np.arange(MEM_RES) - (MEM_RES - 1) * 0.5) * mem_pitch * 1e6
ext = [m_um[0], m_um[-1], m_um[0], m_um[-1]]
mid = MEM_RES // 2

n_configs = len(CONFIGS)
n_methods = 1 + len(GELSIGHT_NAs)  # holographic + GelSight variants

# --- Figure 1: Big comparison grid ---
fig, axes = plt.subplots(n_configs, n_methods + 1, figsize=(5*(n_methods+1), 4*n_configs),
                         squeeze=False)
fig.suptitle("GelSight vs Holographic — Reconstruction Comparison",
             fontsize=14, color="#e6edf3")
fig.patch.set_facecolor("#0d1117")

for row, (def_type, amp) in enumerate(CONFIGS):
    h_true = results_holo[def_type]["h_true"] * 1e9
    vmax = max(abs(h_true).max(), 1)

    # Col 0: ground truth
    ax = axes[row, 0]
    ax.set_facecolor("#0d1117")
    im = ax.imshow(h_true.T, extent=ext, origin="lower", cmap="RdBu",
                   vmin=-vmax, vmax=vmax, aspect="equal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"GT: {def_type}", fontsize=8, color="#e6edf3")
    ax.tick_params(colors="#8b949e", labelsize=5)
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

    # Col 1: holographic
    h_pred = results_holo[def_type]["h_pred"] * 1e9
    r = results_holo[def_type]["rmse"]
    p = results_holo[def_type]["psnr"]
    ax = axes[row, 1]
    ax.set_facecolor("#0d1117")
    im = ax.imshow(h_pred.T, extent=ext, origin="lower", cmap="RdBu",
                   vmin=-vmax, vmax=vmax, aspect="equal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Holographic\nRMSE={r:.1f}nm  PSNR={p:.1f}dB",
                 fontsize=7, color="#58a6ff")
    ax.tick_params(colors="#8b949e", labelsize=5)
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

    # Cols 2+: GelSight variants
    for j, NA in enumerate(GELSIGHT_NAs):
        key = (def_type, NA)
        h_pred = results_gs[key]["h_pred"] * 1e9
        r = results_gs[key]["rmse"]
        p = results_gs[key]["psnr"]
        ray = results_gs[key]["rayleigh"] * 1e6

        ax = axes[row, 2 + j]
        ax.set_facecolor("#0d1117")
        im = ax.imshow(h_pred.T, extent=ext, origin="lower", cmap="RdBu",
                       vmin=-vmax, vmax=vmax, aspect="equal")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"GelSight NA={NA}\nRayleigh={ray:.1f}μm  RMSE={r:.1f}nm  PSNR={p:.1f}dB",
                     fontsize=7, color="#3fb950")
        ax.tick_params(colors="#8b949e", labelsize=5)
        for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

plt.tight_layout()
plt.savefig(OUT / "gelsight_vs_holographic_grid.png", dpi=150, bbox_inches="tight",
            facecolor="#0d1117")
plt.close()
print(f"\nSaved -> {OUT / 'gelsight_vs_holographic_grid.png'}")


# --- Figure 2: Cross-section comparison ---
fig, axes = plt.subplots(n_configs, 2, figsize=(14, 4*n_configs), squeeze=False)
fig.suptitle("Cross-section Comparison — GelSight vs Holographic",
             fontsize=13, color="#e6edf3")
fig.patch.set_facecolor("#0d1117")

for row, (def_type, amp) in enumerate(CONFIGS):
    h_true = results_holo[def_type]["h_true"] * 1e9

    # Left: cross-section
    ax = axes[row, 0]
    ax.set_facecolor("#0d1117")
    ax.plot(m_um, h_true[mid, :], color="#8b949e", lw=2, ls=":", label="Ground truth")
    ax.plot(m_um, results_holo[def_type]["h_pred"][mid, :]*1e9,
            color="#58a6ff", lw=1.5, label="Holographic")
    for NA in GELSIGHT_NAs:
        h_gs = results_gs[(def_type, NA)]["h_pred"] * 1e9
        ax.plot(m_um, h_gs[mid, :], lw=1.2,
                label=f"GelSight NA={NA}")
    ax.set_title(f"{def_type} — centre cross-section", fontsize=9, color="#e6edf3")
    ax.set_xlabel("x [μm]", fontsize=8, color="#8b949e")
    ax.set_ylabel("h [nm]", fontsize=8, color="#8b949e")
    ax.legend(fontsize=7, labelcolor="#e6edf3", framealpha=0.3)
    ax.tick_params(colors="#8b949e", labelsize=7)
    ax.grid(color="#30363d", lw=0.4, ls=":")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

    # Right: PSNR bar chart
    ax = axes[row, 1]
    ax.set_facecolor("#0d1117")
    names = ["Holographic"] + [f"GelSight\nNA={NA}" for NA in GELSIGHT_NAs]
    psnrs = [results_holo[def_type]["psnr"]]
    psnrs += [results_gs[(def_type, NA)]["psnr"] for NA in GELSIGHT_NAs]
    colors = ["#58a6ff"] + ["#3fb950"]*len(GELSIGHT_NAs)
    bars = ax.bar(names, psnrs, color=colors, edgecolor="#30363d", lw=0.5)
    for bar, val in zip(bars, psnrs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", fontsize=7, color="#e6edf3")
    ax.set_title(f"{def_type} — PSNR comparison", fontsize=9, color="#e6edf3")
    ax.set_ylabel("PSNR [dB]", fontsize=8, color="#8b949e")
    ax.tick_params(colors="#8b949e", labelsize=7)
    ax.grid(color="#30363d", lw=0.4, ls=":", axis="y")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

plt.tight_layout()
plt.savefig(OUT / "gelsight_vs_holographic_profile.png", dpi=150, bbox_inches="tight",
            facecolor="#0d1117")
plt.close()
print(f"Saved -> {OUT / 'gelsight_vs_holographic_profile.png'}")


# --- Figure 3: Diffraction limit demonstration (fine_grating) ---
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Diffraction Limit — Fine Grating (30μm period)\n"
             "GelSight resolution degrades with smaller NA; Holographic preserves fine detail",
             fontsize=12, color="#e6edf3")
fig.patch.set_facecolor("#0d1117")

# zoom to centre
zoom = 60  # ±60 pixels from centre
s, e = mid - zoom, mid + zoom
ext_z = [m_um[s], m_um[e-1], m_um[s], m_um[e-1]]

h_true_grating = results_holo["fine_grating"]["h_true"] * 1e9
vmax = max(abs(h_true_grating[s:e, s:e]).max(), 1)

# Row 0: 2D zoomed images
panels_top = [
    ("Ground Truth", h_true_grating, "#8b949e"),
    ("Holographic", results_holo["fine_grating"]["h_pred"]*1e9, "#58a6ff"),
    (f"GelSight NA={GELSIGHT_NAs[0]}",
     results_gs[("fine_grating", GELSIGHT_NAs[0])]["h_pred"]*1e9, "#3fb950"),
]
for col, (title, data, clr) in enumerate(panels_top):
    ax = axes[0, col]
    ax.set_facecolor("#0d1117")
    im = ax.imshow(data[s:e, s:e].T, extent=ext_z, origin="lower",
                   cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="equal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=9, color=clr)
    ax.tick_params(colors="#8b949e", labelsize=6)
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

# Row 1: cross-section, loss curve, summary bar
ax = axes[1, 0]
ax.set_facecolor("#0d1117")
ax.plot(m_um[s:e], h_true_grating[mid, s:e], color="#8b949e", lw=2, ls=":", label="GT")
ax.plot(m_um[s:e], results_holo["fine_grating"]["h_pred"][mid, s:e]*1e9,
        color="#58a6ff", lw=1.5, label="Holographic")
for NA in GELSIGHT_NAs:
    h_gs = results_gs[("fine_grating", NA)]["h_pred"] * 1e9
    ax.plot(m_um[s:e], h_gs[mid, s:e], lw=1.2, label=f"GelSight NA={NA}")
ax.set_title("Cross-section (zoomed)", fontsize=9, color="#e6edf3")
ax.set_xlabel("x [μm]", fontsize=8, color="#8b949e")
ax.set_ylabel("h [nm]", fontsize=8, color="#8b949e")
ax.legend(fontsize=6, labelcolor="#e6edf3", framealpha=0.3)
ax.tick_params(colors="#8b949e", labelsize=7)
ax.grid(color="#30363d", lw=0.4, ls=":")
for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

# Loss curve (holographic only)
ax = axes[1, 1]
ax.set_facecolor("#0d1117")
ax.plot(results_holo["fine_grating"]["losses"], color="#58a6ff", lw=1.0)
ax.set_yscale("log")
ax.set_title("Holographic Loss Curve", fontsize=9, color="#e6edf3")
ax.set_xlabel("iteration", fontsize=8, color="#8b949e")
ax.set_ylabel("loss", fontsize=8, color="#8b949e")
ax.tick_params(colors="#8b949e", labelsize=7)
ax.grid(color="#30363d", lw=0.4, ls=":")
for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

# RMSE bar chart for fine_grating
ax = axes[1, 2]
ax.set_facecolor("#0d1117")
names = ["Holographic"] + [f"GelSight\nNA={NA}" for NA in GELSIGHT_NAs]
rmses = [results_holo["fine_grating"]["rmse"]]
rmses += [results_gs[("fine_grating", NA)]["rmse"] for NA in GELSIGHT_NAs]
colors = ["#58a6ff"] + ["#3fb950"]*len(GELSIGHT_NAs)
bars = ax.bar(names, rmses, color=colors, edgecolor="#30363d", lw=0.5)
for bar, val in zip(bars, rmses):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.1f}", ha="center", fontsize=7, color="#e6edf3")
ax.set_title("RMSE [nm] — Fine Grating", fontsize=9, color="#e6edf3")
ax.set_ylabel("RMSE [nm]", fontsize=8, color="#8b949e")
ax.tick_params(colors="#8b949e", labelsize=7)
ax.grid(color="#30363d", lw=0.4, ls=":", axis="y")
for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

plt.tight_layout()
plt.savefig(OUT / "gelsight_diffraction_limit.png", dpi=150, bbox_inches="tight",
            facecolor="#0d1117")
plt.close()
print(f"Saved -> {OUT / 'gelsight_diffraction_limit.png'}")


# --- Figure 4: Overall summary ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Overall Comparison Summary — All Deformation Types",
             fontsize=13, color="#e6edf3")
fig.patch.set_facecolor("#0d1117")

def_names = [d for d, _ in CONFIGS]
x = np.arange(len(def_names))
width = 0.18

# PSNR
ax = axes[0]
ax.set_facecolor("#0d1117")
bars_holo = ax.bar(x - width*1.5, [results_holo[d]["psnr"] for d in def_names],
                   width, label="Holographic", color="#58a6ff", edgecolor="#30363d")
for i, NA in enumerate(GELSIGHT_NAs):
    vals = [results_gs[(d, NA)]["psnr"] for d in def_names]
    ax.bar(x - width*0.5 + i*width, vals, width,
           label=f"GelSight NA={NA}", edgecolor="#30363d", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(def_names, fontsize=7, color="#8b949e")
ax.set_ylabel("PSNR [dB]", fontsize=8, color="#8b949e")
ax.set_title("PSNR (higher is better)", fontsize=10, color="#e6edf3")
ax.legend(fontsize=7, labelcolor="#e6edf3", framealpha=0.3)
ax.tick_params(colors="#8b949e", labelsize=7)
ax.grid(color="#30363d", lw=0.4, ls=":", axis="y")
for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

# RMSE
ax = axes[1]
ax.set_facecolor("#0d1117")
ax.bar(x - width*1.5, [results_holo[d]["rmse"] for d in def_names],
       width, label="Holographic", color="#58a6ff", edgecolor="#30363d")
for i, NA in enumerate(GELSIGHT_NAs):
    vals = [results_gs[(d, NA)]["rmse"] for d in def_names]
    ax.bar(x - width*0.5 + i*width, vals, width,
           label=f"GelSight NA={NA}", edgecolor="#30363d", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(def_names, fontsize=7, color="#8b949e")
ax.set_ylabel("RMSE [nm]", fontsize=8, color="#8b949e")
ax.set_title("RMSE (lower is better)", fontsize=10, color="#e6edf3")
ax.legend(fontsize=7, labelcolor="#e6edf3", framealpha=0.3)
ax.tick_params(colors="#8b949e", labelsize=7)
ax.grid(color="#30363d", lw=0.4, ls=":", axis="y")
for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

plt.tight_layout()
plt.savefig(OUT / "gelsight_vs_holographic_summary.png", dpi=150, bbox_inches="tight",
            facecolor="#0d1117")
plt.close()
print(f"Saved -> {OUT / 'gelsight_vs_holographic_summary.png'}")

print("\n=== All comparison plots saved ===")
