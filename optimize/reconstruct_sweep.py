"""
reconstruct_sweep.py
--------------------
Gradient-descent reconstruction of membrane height h from CMOS intensity,
sweeping across key physical parameters:

  1. Propagation distance  : [1, 3, 5, 10] mm
  2. Deformation amplitude : [100, 200, 500, 1000] nm
  3. Deformation type      : single_bump, multi_bump, random

For each configuration the script:
  - generates ground-truth h and forward I_target
  - runs Adam optimisation (MSE + TV reg) to recover h from I_target
  - records loss curve, final PSNR, RMSE

Outputs:
  output/reconstruct_sweep.png         – big comparison grid
  output/reconstruct_loss_curves.png   – loss curves per config
  output/reconstruct_metrics.png       – bar charts of PSNR / RMSE
"""

import sys, os, itertools, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from sensor_model import HolographicSensor, gaussian_bump, make_h_multi_bump, make_h_random
from sensor_params import wavelength, mem_pitch
from sensor_utils import STYLE

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

# -----------------------------------------------------------------------
# Sweep parameters
# -----------------------------------------------------------------------
DISTANCES   = [1e-3, 3e-3, 5e-3, 10e-3]           # [m]
AMPLITUDES  = [100e-9, 200e-9, 500e-9, 1000e-9]    # [m]
DEF_TYPES   = ["single_bump", "multi_bump", "random"]

# Use smaller resolution for speed
MEM_RES  = 128
CMOS_RES = 256
GRID_RES = 384

N_ITER   = 500
LR       = 5e-10
LAM_TV   = 1e-20

# -----------------------------------------------------------------------
# Deformation generators (adapted for variable amplitude)
# -----------------------------------------------------------------------

def make_h(def_type: str, amplitude: float) -> torch.Tensor:
    N, dx = MEM_RES, mem_pitch
    if def_type == "single_bump":
        return gaussian_bump(N=N, dx=dx, amplitude=amplitude,
                             sigma=150e-6, device=device)
    elif def_type == "multi_bump":
        # Scale multi-bump so that peak ~ amplitude
        # NOTE: centers must be within membrane: ±(N/2)*dx = ±640um for N=128
        specs = [
            dict(amplitude=1.0,  sigma=80e-6,  center=(0.0, 0.0)),
            dict(amplitude=0.8,  sigma=100e-6, center=(300e-6, 200e-6)),
            dict(amplitude=0.6,  sigma=60e-6,  center=(-250e-6, -300e-6)),
            dict(amplitude=0.9,  sigma=70e-6,  center=(100e-6, -400e-6)),
            dict(amplitude=0.7,  sigma=120e-6, center=(-400e-6, 300e-6)),
        ]
        h = torch.zeros(N, N, dtype=torch.float32, device=device)
        for s in specs:
            h += gaussian_bump(N=N, dx=dx, device=device,
                               amplitude=s["amplitude"], sigma=s["sigma"],
                               center=s["center"])
        # normalise peak to desired amplitude
        h = h * (amplitude / h.abs().max().clamp(min=1e-30))
        return h
    elif def_type == "random":
        h = make_h_random(N=N, dx=dx, device=device, amplitude=amplitude)
        return h
    else:
        raise ValueError(f"Unknown deformation type: {def_type}")

# -----------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------

def rmse_nm(h_pred: torch.Tensor, h_true: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((h_pred - h_true)**2)).item()) * 1e9

def psnr_db(h_pred: torch.Tensor, h_true: torch.Tensor) -> float:
    mse = float(torch.mean((h_pred - h_true)**2).item())
    peak = float(h_true.abs().max().item())
    if mse < 1e-30:
        return 100.0
    return 10.0 * np.log10(peak**2 / mse)

# -----------------------------------------------------------------------
# Single reconstruction run
# -----------------------------------------------------------------------

def reconstruct(sensor, h_true, n_iter=N_ITER, lr=LR, lam_tv=LAM_TV):
    with torch.no_grad():
        I_target = sensor(h_true)

    h_pred = torch.zeros(MEM_RES, MEM_RES, dtype=torch.float32,
                         device=device, requires_grad=True)
    opt = torch.optim.Adam([h_pred], lr=lr)

    losses = []
    for i in range(n_iter):
        opt.zero_grad()
        I_pred = sensor(h_pred)
        mse = torch.mean((I_pred - I_target)**2)
        tv = (torch.mean(torch.abs(h_pred[1:, :] - h_pred[:-1, :])) +
              torch.mean(torch.abs(h_pred[:, 1:] - h_pred[:, :-1])))
        loss = mse + lam_tv * tv
        loss.backward()
        opt.step()
        losses.append(float(loss))

    return h_pred.detach(), losses, I_target

# -----------------------------------------------------------------------
# Main sweep
# -----------------------------------------------------------------------

# We'll do a focused sweep: fix amplitude=200nm, sweep distance x def_type
# then fix distance=5mm, sweep amplitude x def_type
# This gives manageable number of runs.

results = []

# --- Sweep 1: distance x def_type (amplitude fixed at 200nm) ---
print("\n=== Sweep 1: Distance x Deformation Type (amp=200nm) ===")
amp_fixed = 200e-9
for dist, def_type in itertools.product(DISTANCES, DEF_TYPES):
    label = f"d={dist*1e3:.0f}mm  {def_type}"
    print(f"\n  {label} ...", end="", flush=True)
    t0 = time.time()

    sensor = HolographicSensor(
        wavelength=wavelength, mem_res=MEM_RES, mem_pitch=mem_pitch,
        cmos_res=CMOS_RES, grid_res=GRID_RES, distance=dist, device=device,
    ).to(device)

    h_true = make_h(def_type, amp_fixed)
    h_pred, losses, I_target = reconstruct(sensor, h_true)

    r = rmse_nm(h_pred, h_true)
    p = psnr_db(h_pred, h_true)
    dt = time.time() - t0
    print(f"  RMSE={r:.2f}nm  PSNR={p:.1f}dB  ({dt:.1f}s)")

    results.append(dict(
        sweep="distance", distance=dist, amplitude=amp_fixed,
        def_type=def_type, label=label,
        h_true=h_true.cpu().numpy(), h_pred=h_pred.cpu().numpy(),
        losses=losses, rmse=r, psnr=p,
        I_target=I_target.cpu().numpy(),
    ))

# --- Sweep 2: amplitude x def_type (distance fixed at 5mm) ---
print("\n=== Sweep 2: Amplitude x Deformation Type (d=5mm) ===")
dist_fixed = 5e-3
for amp, def_type in itertools.product(AMPLITUDES, DEF_TYPES):
    label = f"amp={amp*1e9:.0f}nm  {def_type}"
    # Skip the 200nm cases already run in sweep 1
    if amp == amp_fixed:
        # reuse from sweep 1
        existing = [r for r in results
                    if r["distance"] == dist_fixed and r["def_type"] == def_type]
        if existing:
            r = existing[0].copy()
            r["sweep"] = "amplitude"
            r["label"] = label
            results.append(r)
            print(f"\n  {label} ... (reused from sweep 1)")
            continue

    print(f"\n  {label} ...", end="", flush=True)
    t0 = time.time()

    sensor = HolographicSensor(
        wavelength=wavelength, mem_res=MEM_RES, mem_pitch=mem_pitch,
        cmos_res=CMOS_RES, grid_res=GRID_RES, distance=dist_fixed, device=device,
    ).to(device)

    h_true = make_h(def_type, amp)
    h_pred, losses, I_target = reconstruct(sensor, h_true)

    r = rmse_nm(h_pred, h_true)
    p = psnr_db(h_pred, h_true)
    dt = time.time() - t0
    print(f"  RMSE={r:.2f}nm  PSNR={p:.1f}dB  ({dt:.1f}s)")

    results.append(dict(
        sweep="amplitude", distance=dist_fixed, amplitude=amp,
        def_type=def_type, label=label,
        h_true=h_true.cpu().numpy(), h_pred=h_pred.cpu().numpy(),
        losses=losses, rmse=r, psnr=p,
        I_target=I_target.cpu().numpy(),
    ))

print(f"\nTotal runs: {len(results)}")

# -----------------------------------------------------------------------
# Plot 1: Loss curves
# -----------------------------------------------------------------------
plt.rcParams.update(STYLE)

# Distance sweep loss curves
dist_results = [r for r in results if r["sweep"] == "distance"]
n_dist = len(DISTANCES)
n_def  = len(DEF_TYPES)

fig, axes = plt.subplots(n_dist, n_def, figsize=(5*n_def, 4*n_dist), squeeze=False)
fig.suptitle("Loss Curves — Distance × Deformation Type  (amp=200nm)",
             fontsize=13, color="#e6edf3")
fig.patch.set_facecolor("#0d1117")

for r in dist_results:
    di = DISTANCES.index(r["distance"])
    dj = DEF_TYPES.index(r["def_type"])
    ax = axes[di, dj]
    ax.set_facecolor("#0d1117")
    ax.plot(r["losses"], color="#58a6ff", lw=1.0)
    ax.set_yscale("log")
    ax.set_title(r["label"], fontsize=9, color="#e6edf3")
    ax.set_xlabel("iter", fontsize=7, color="#8b949e")
    ax.set_ylabel("loss", fontsize=7, color="#8b949e")
    ax.tick_params(colors="#8b949e", labelsize=6)
    ax.grid(color="#30363d", lw=0.4, ls=":")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

plt.tight_layout()
plt.savefig(OUT / "sweep_loss_distance.png", dpi=150, bbox_inches="tight",
            facecolor="#0d1117")
plt.close()
print(f"\nSaved -> {OUT / 'sweep_loss_distance.png'}")

# Amplitude sweep loss curves
amp_results = [r for r in results if r["sweep"] == "amplitude"]
n_amp = len(AMPLITUDES)

fig, axes = plt.subplots(n_amp, n_def, figsize=(5*n_def, 4*n_amp), squeeze=False)
fig.suptitle("Loss Curves — Amplitude × Deformation Type  (d=5mm)",
             fontsize=13, color="#e6edf3")
fig.patch.set_facecolor("#0d1117")

for r in amp_results:
    ai = AMPLITUDES.index(r["amplitude"])
    dj = DEF_TYPES.index(r["def_type"])
    ax = axes[ai, dj]
    ax.set_facecolor("#0d1117")
    ax.plot(r["losses"], color="#f78166", lw=1.0)
    ax.set_yscale("log")
    ax.set_title(r["label"], fontsize=9, color="#e6edf3")
    ax.set_xlabel("iter", fontsize=7, color="#8b949e")
    ax.set_ylabel("loss", fontsize=7, color="#8b949e")
    ax.tick_params(colors="#8b949e", labelsize=6)
    ax.grid(color="#30363d", lw=0.4, ls=":")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

plt.tight_layout()
plt.savefig(OUT / "sweep_loss_amplitude.png", dpi=150, bbox_inches="tight",
            facecolor="#0d1117")
plt.close()
print(f"Saved -> {OUT / 'sweep_loss_amplitude.png'}")

# -----------------------------------------------------------------------
# Plot 2: Reconstruction quality — h_true vs h_pred cross-sections
# -----------------------------------------------------------------------
# Show distance sweep (single_bump only) as a clear example
sb_dist = [r for r in dist_results if r["def_type"] == "single_bump"]

fig, axes = plt.subplots(2, len(sb_dist), figsize=(5*len(sb_dist), 8), squeeze=False)
fig.suptitle("Reconstruction: single_bump @ 200nm — varying distance",
             fontsize=13, color="#e6edf3")
fig.patch.set_facecolor("#0d1117")

m_um = (np.arange(MEM_RES) - (MEM_RES - 1) * 0.5) * mem_pitch * 1e6

for col, r in enumerate(sb_dist):
    ht = r["h_true"] * 1e9
    hp = r["h_pred"] * 1e9
    mid = MEM_RES // 2

    # Row 0: cross-section
    ax = axes[0, col]
    ax.set_facecolor("#0d1117")
    ax.plot(m_um, ht[mid, :], color="#8b949e", lw=1.5, ls=":", label="h_true")
    ax.plot(m_um, hp[mid, :], color="#58a6ff", lw=1.5, label="h_pred")
    ax.set_title(f"d={r['distance']*1e3:.0f}mm  RMSE={r['rmse']:.1f}nm  PSNR={r['psnr']:.1f}dB",
                 fontsize=8, color="#e6edf3")
    ax.set_xlabel("x [μm]", fontsize=7, color="#8b949e")
    ax.set_ylabel("h [nm]", fontsize=7, color="#8b949e")
    ax.legend(fontsize=7, labelcolor="#e6edf3", framealpha=0.3)
    ax.tick_params(colors="#8b949e", labelsize=6)
    ax.grid(color="#30363d", lw=0.4, ls=":")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

    # Row 1: 2D error map
    ax = axes[1, col]
    ax.set_facecolor("#0d1117")
    err = hp - ht
    ext = [m_um[0], m_um[-1], m_um[0], m_um[-1]]
    vabs = max(abs(err).max(), 1e-3)
    im = ax.imshow(err.T, extent=ext, origin="lower", cmap="RdBu",
                   vmin=-vabs, vmax=vabs, aspect="equal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Δh [nm]")
    ax.set_title("h_pred − h_true", fontsize=8, color="#e6edf3")
    ax.set_xlabel("y [μm]", fontsize=7, color="#8b949e")
    ax.set_ylabel("x [μm]", fontsize=7, color="#8b949e")
    ax.tick_params(colors="#8b949e", labelsize=6)
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

plt.tight_layout()
plt.savefig(OUT / "sweep_reconstruct_distance.png", dpi=150, bbox_inches="tight",
            facecolor="#0d1117")
plt.close()
print(f"Saved -> {OUT / 'sweep_reconstruct_distance.png'}")

# -----------------------------------------------------------------------
# Plot 3: Bar chart — PSNR & RMSE summary
# -----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Reconstruction Quality Summary", fontsize=13, color="#e6edf3")
fig.patch.set_facecolor("#0d1117")

colors_def = {"single_bump": "#58a6ff", "multi_bump": "#3fb950", "random": "#e3b341"}

# (0,0) PSNR vs distance
ax = axes[0, 0]
ax.set_facecolor("#0d1117")
for dt in DEF_TYPES:
    rs = [r for r in dist_results if r["def_type"] == dt]
    ds = [r["distance"]*1e3 for r in rs]
    ps = [r["psnr"] for r in rs]
    ax.plot(ds, ps, "o-", color=colors_def[dt], lw=1.5, ms=6, label=dt)
ax.set_xlabel("distance [mm]", fontsize=8, color="#8b949e")
ax.set_ylabel("PSNR [dB]", fontsize=8, color="#8b949e")
ax.set_title("PSNR vs Distance  (amp=200nm)", fontsize=9, color="#e6edf3")
ax.legend(fontsize=7, labelcolor="#e6edf3", framealpha=0.3)
ax.tick_params(colors="#8b949e", labelsize=7)
ax.grid(color="#30363d", lw=0.4, ls=":")
for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

# (0,1) RMSE vs distance
ax = axes[0, 1]
ax.set_facecolor("#0d1117")
for dt in DEF_TYPES:
    rs = [r for r in dist_results if r["def_type"] == dt]
    ds = [r["distance"]*1e3 for r in rs]
    rm = [r["rmse"] for r in rs]
    ax.plot(ds, rm, "o-", color=colors_def[dt], lw=1.5, ms=6, label=dt)
ax.set_xlabel("distance [mm]", fontsize=8, color="#8b949e")
ax.set_ylabel("RMSE [nm]", fontsize=8, color="#8b949e")
ax.set_title("RMSE vs Distance  (amp=200nm)", fontsize=9, color="#e6edf3")
ax.legend(fontsize=7, labelcolor="#e6edf3", framealpha=0.3)
ax.tick_params(colors="#8b949e", labelsize=7)
ax.grid(color="#30363d", lw=0.4, ls=":")
for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

# (1,0) PSNR vs amplitude
ax = axes[1, 0]
ax.set_facecolor("#0d1117")
for dt in DEF_TYPES:
    rs = [r for r in amp_results if r["def_type"] == dt]
    amps = [r["amplitude"]*1e9 for r in rs]
    ps = [r["psnr"] for r in rs]
    ax.plot(amps, ps, "s-", color=colors_def[dt], lw=1.5, ms=6, label=dt)
ax.set_xlabel("amplitude [nm]", fontsize=8, color="#8b949e")
ax.set_ylabel("PSNR [dB]", fontsize=8, color="#8b949e")
ax.set_title("PSNR vs Amplitude  (d=5mm)", fontsize=9, color="#e6edf3")
ax.legend(fontsize=7, labelcolor="#e6edf3", framealpha=0.3)
ax.tick_params(colors="#8b949e", labelsize=7)
ax.grid(color="#30363d", lw=0.4, ls=":")
for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

# (1,1) RMSE vs amplitude
ax = axes[1, 1]
ax.set_facecolor("#0d1117")
for dt in DEF_TYPES:
    rs = [r for r in amp_results if r["def_type"] == dt]
    amps = [r["amplitude"]*1e9 for r in rs]
    rm = [r["rmse"] for r in rs]
    ax.plot(amps, rm, "s-", color=colors_def[dt], lw=1.5, ms=6, label=dt)
ax.set_xlabel("amplitude [nm]", fontsize=8, color="#8b949e")
ax.set_ylabel("RMSE [nm]", fontsize=8, color="#8b949e")
ax.set_title("RMSE vs Amplitude  (d=5mm)", fontsize=9, color="#e6edf3")
ax.legend(fontsize=7, labelcolor="#e6edf3", framealpha=0.3)
ax.tick_params(colors="#8b949e", labelsize=7)
ax.grid(color="#30363d", lw=0.4, ls=":")
for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

plt.tight_layout()
plt.savefig(OUT / "sweep_metrics.png", dpi=150, bbox_inches="tight",
            facecolor="#0d1117")
plt.close()
print(f"Saved -> {OUT / 'sweep_metrics.png'}")

# -----------------------------------------------------------------------
# Plot 4: Big grid — all deformation types, distance sweep, h_true vs h_pred
# -----------------------------------------------------------------------
fig, axes = plt.subplots(n_def, n_dist * 2, figsize=(5 * n_dist, 4 * n_def),
                         squeeze=False)
fig.suptitle("Reconstruction Grid: h_true (odd cols) vs h_pred (even cols) — amp=200nm",
             fontsize=12, color="#e6edf3")
fig.patch.set_facecolor("#0d1117")

for r in dist_results:
    row = DEF_TYPES.index(r["def_type"])
    col_base = DISTANCES.index(r["distance"]) * 2
    ht = r["h_true"] * 1e9
    hp = r["h_pred"] * 1e9
    ext = [m_um[0], m_um[-1], m_um[0], m_um[-1]]
    vmax = max(abs(ht).max(), abs(hp).max(), 1)

    for j, (data, suffix) in enumerate([(ht, "true"), (hp, "pred")]):
        ax = axes[row, col_base + j]
        ax.set_facecolor("#0d1117")
        im = ax.imshow(data.T, extent=ext, origin="lower", cmap="RdBu",
                       vmin=-vmax, vmax=vmax, aspect="equal")
        d_mm = r["distance"] * 1e3
        ax.set_title(f"{r['def_type']}  d={d_mm:.0f}mm  h_{suffix}",
                     fontsize=7, color="#e6edf3")
        ax.tick_params(colors="#8b949e", labelsize=5)
        for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

plt.tight_layout()
plt.savefig(OUT / "sweep_grid.png", dpi=120, bbox_inches="tight",
            facecolor="#0d1117")
plt.close()
print(f"Saved -> {OUT / 'sweep_grid.png'}")

print("\n=== All sweep plots saved ===")
