"""
random_pattern_reconstruction.py
--------------------------------
Verify that the corrected algorithm (squared reparam + Adam, h >= 0)
reconstructs *arbitrary* smooth height maps, not just a single Gaussian bump.

Strategy
--------
We generate band-limited random height maps by:
  1. Drawing iid Gaussian noise on the MEM grid
  2. Low-pass filtering with a Gaussian kernel of width sigma_smooth
     (so per-pixel slope stays below lam/4 — no phase wrap)
  3. Rescaling to a target peak amplitude A
  4. Shifting to be non-negative (h >= 0) — physical press-in constraint

For several (seed, A, sigma_smooth) combinations we run the corrected
reconstruction and report PSNR / RMSE. We also display 2D ground-truth
vs reconstruction vs error maps for qualitative comparison.

Output:
  output/random_pattern_reconstruction/
    random_pattern.png
    random_pattern.txt
    random_pattern.json
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
from sensor_model  import HolographicSensor
from sensor_params import wavelength, mem_pitch
from sensor_utils  import STYLE

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "output" / "random_pattern_reconstruction"
OUT.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

MEM, CMOS, GRID, DIST = 128, 256, 384, 5e-3
LR     = 3e-3
N_ITER = 15000

lam = wavelength
dx  = mem_pitch
PHASE_WRAP_NM = lam / 4.0 * 1e9

sensor = HolographicSensor(
    wavelength=lam, mem_res=MEM, mem_pitch=dx,
    cmos_res=CMOS, grid_res=GRID, distance=DIST, device=device,
).to(device)


def random_height(seed, A_m, sigma_smooth_um):
    """Smooth non-negative random height map.
    A_m              : target peak amplitude in metres
    sigma_smooth_um  : low-pass Gaussian sigma in um (feature scale)
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((MEM, MEM)).astype(np.float32)

    sigma_px = sigma_smooth_um * 1e-6 / dx
    k_half = int(max(3 * sigma_px, 3))
    kx = np.arange(-k_half, k_half + 1)
    kernel_1d = np.exp(-(kx ** 2) / (2 * sigma_px ** 2))
    kernel_1d /= kernel_1d.sum()
    kernel_2d = np.outer(kernel_1d, kernel_1d).astype(np.float32)

    noise_t = torch.from_numpy(noise).unsqueeze(0).unsqueeze(0)
    kernel_t = torch.from_numpy(kernel_2d).unsqueeze(0).unsqueeze(0)
    pad = k_half
    smooth = F.conv2d(F.pad(noise_t, (pad, pad, pad, pad), mode="reflect"),
                      kernel_t).squeeze().numpy()

    smooth -= smooth.min()
    peak = float(smooth.max())
    if peak < 1e-30:
        smooth = np.zeros_like(smooth)
    else:
        smooth = smooth * (A_m / peak)
    return torch.tensor(smooth, dtype=torch.float32, device=device)


def reconstruct(I_tgt, h_scale):
    raw = torch.full((MEM, MEM), 0.2, device=device,
                     dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([raw], lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_ITER)
    for _ in range(N_ITER):
        opt.zero_grad()
        h = (raw * raw) * h_scale
        loss = torch.mean((sensor(h) - I_tgt) ** 2)
        loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        h_final = (raw * raw) * h_scale
        l = float(torch.mean((sensor(h_final) - I_tgt) ** 2))
    return h_final, l


def psnr_db(a, b):
    mse = float(torch.mean((a - b) ** 2))
    peak = float(b.abs().max())
    if mse < 1e-30: return 200.0
    if peak < 1e-30: return 0.0
    return 10.0 * np.log10(peak ** 2 / mse)


# (A_nm, sigma_smooth_um, seed) — 5 random patterns at two feature scales
cases = [
    dict(tag="r1", A_nm=100, sigma=150, seed=1),
    dict(tag="r2", A_nm=100, sigma=150, seed=2),
    dict(tag="r3", A_nm=100, sigma=150, seed=3),
    dict(tag="r4", A_nm=200, sigma=150, seed=4),
    dict(tag="r5", A_nm=100, sigma=300, seed=5),
]

print(f"\nphase-wrap pixel limit lam/4 = {PHASE_WRAP_NM:.2f} nm")
print(f"\n{'tag':>4}  {'A[nm]':>6}  {'sig[um]':>8}  {'seed':>4}  "
      f"{'slope':>7}  {'pix[nm]':>8}  {'PSNR':>8}  {'RMSE[nm]':>10}  {'t[s]':>6}")

results = []
keep = {}
for c in cases:
    A_m = c["A_nm"] * 1e-9
    h_scale = max(2.0 * A_m, 50e-9)
    h_true = random_height(c["seed"], A_m, c["sigma"])
    # empirical per-pixel slope
    with torch.no_grad():
        gy = (h_true[1:, :] - h_true[:-1, :]).abs().max().item()
        gx = (h_true[:, 1:] - h_true[:, :-1]).abs().max().item()
    per_pix_nm = max(gx, gy) * 1e9
    slope = per_pix_nm / (dx * 1e9)  # rise/run in m/m
    with torch.no_grad():
        I_tgt = sensor(h_true)
    t0 = time.time()
    h_rec, loss = reconstruct(I_tgt, h_scale)
    p = psnr_db(h_rec, h_true)
    rmse = float(torch.sqrt(torch.mean((h_rec - h_true) ** 2))) * 1e9
    dt = time.time() - t0
    print(f"{c['tag']:>4}  {c['A_nm']:>6.0f}  {c['sigma']:>8.0f}  "
          f"{c['seed']:>4}  {slope:>7.4f}  {per_pix_nm:>8.2f}  "
          f"{p:>8.2f}  {rmse:>10.3f}  {dt:>6.1f}")
    results.append(dict(tag=c["tag"], A_nm=c["A_nm"], sigma_um=c["sigma"],
                        seed=c["seed"], per_pix_nm=per_pix_nm,
                        slope=slope, psnr=p, rmse_nm=rmse, loss=loss, t=dt))
    keep[c["tag"]] = dict(h_true=h_true.cpu().numpy() * 1e9,
                           h_rec =h_rec.cpu().numpy() * 1e9)


# ---------------- plot ----------------
plt.rcParams.update(STYLE)
ncase = len(cases)
fig = plt.figure(figsize=(4 * ncase, 10))
gs = fig.add_gridspec(3, ncase)
fig.suptitle(
    "Random pattern reconstruction  (squared h>=0 + Adam 15k)",
    fontsize=13, color="#e6edf3",
)
fig.patch.set_facecolor("#0d1117")


def _style(ax):
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e", labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")


extent_um = (MEM * dx / 2) * 1e6
for j, r in enumerate(results):
    k = keep[r["tag"]]
    vmax = float(np.max(k["h_true"]))
    vmin = 0.0
    err = k["h_rec"] - k["h_true"]
    emax = float(max(abs(err.min()), abs(err.max())))

    ax = fig.add_subplot(gs[0, j])
    im = ax.imshow(k["h_true"], cmap="magma", vmin=vmin, vmax=vmax,
                   extent=[-extent_um, extent_um, -extent_um, extent_um])
    ax.set_title(f"{r['tag']}  GT  A={r['A_nm']}nm σ={r['sigma_um']}um",
                 fontsize=9, color="#e6edf3")
    _style(ax)
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02).ax.tick_params(
        colors="#8b949e", labelsize=7)

    ax = fig.add_subplot(gs[1, j])
    im = ax.imshow(k["h_rec"], cmap="magma", vmin=vmin, vmax=vmax,
                   extent=[-extent_um, extent_um, -extent_um, extent_um])
    ax.set_title(f"{r['tag']}  rec  PSNR={r['psnr']:.1f} dB",
                 fontsize=9, color="#e6edf3")
    _style(ax)
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02).ax.tick_params(
        colors="#8b949e", labelsize=7)

    ax = fig.add_subplot(gs[2, j])
    im = ax.imshow(err, cmap="RdBu_r", vmin=-emax, vmax=emax,
                   extent=[-extent_um, extent_um, -extent_um, extent_um])
    ax.set_title(f"err  RMSE={r['rmse_nm']:.2f}nm", fontsize=9, color="#e6edf3")
    _style(ax)
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02).ax.tick_params(
        colors="#8b949e", labelsize=7)

plt.tight_layout()
plt.savefig(OUT / "random_pattern.png", dpi=150, bbox_inches="tight",
            facecolor="#0d1117")
plt.close()

lines = [
    "random_pattern_reconstruction.py — beyond the single Gaussian bump",
    "=" * 64,
    f"  wavelength   = {lam*1e9:.1f} nm",
    f"  mem pitch    = {dx*1e6:.1f} um",
    f"  distance     = {DIST*1e3:.1f} mm",
    f"  phase wrap   = {PHASE_WRAP_NM:.2f} nm per pixel",
    f"  algorithm    = squared (h>=0) + Adam {N_ITER} iters, cosine decay",
    "",
    f"  {'tag':>4}  {'A[nm]':>6}  {'sig[um]':>8}  {'pix[nm]':>8}  "
    f"{'PSNR':>8}  {'RMSE[nm]':>10}",
]
for r in results:
    lines.append(
        f"  {r['tag']:>4}  {r['A_nm']:>6.0f}  {r['sigma_um']:>8.0f}  "
        f"{r['per_pix_nm']:>8.2f}  {r['psnr']:>8.2f}  {r['rmse_nm']:>10.3f}"
    )
lines += ["",
          "Reading the table",
          "  * Random patterns with per-pixel slope < lam/4 should reconstruct",
          "    to sub-nm RMSE, same as the single Gaussian bump.",
          "  * If PSNR stays high across all seeds, the algorithm is not",
          "    exploiting a Gaussian-specific symmetry — it is genuinely",
          "    solving the inverse problem for arbitrary smooth deformations."]
txt = "\n".join(lines)
(OUT / "random_pattern.txt").write_text(txt, encoding="utf-8")
print("\n" + txt)

(OUT / "random_pattern.json").write_text(json.dumps(dict(
    config=dict(wavelength_nm=lam*1e9, mem_pitch_um=dx*1e6,
                distance_mm=DIST*1e3, mem=MEM, cmos=CMOS, grid=GRID,
                lr=LR, n_iter=N_ITER, phase_wrap_pixel_nm=PHASE_WRAP_NM),
    results=results,
), indent=2))
print(f"\nSaved -> {OUT/'random_pattern.png'}")
