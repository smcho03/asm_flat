"""
random_surface_eval.py
----------------------
Evaluate holographic reconstruction on random Gaussian surfaces.
Tests multiple random seeds to get statistical performance measures
(mean, std, min, max of PSNR and RMSE).

Sweep parameters:
  - amplitude: [50, 100, 150, 200] nm (sub-wavelength regime)
  - sigma_filter: [100, 200, 400] um (correlation length)
  - seeds: 10 per configuration
"""

import sys, os, time, functools
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Force flush on all prints
print = functools.partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).parent.parent))
from sensor_model import HolographicSensor, make_h_random
from sensor_params import wavelength, mem_pitch
from sensor_utils import STYLE

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

# -----------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------
MEM_RES  = 128
CMOS_RES = 256
GRID_RES = 384
DISTANCE = 5e-3
N_ITER   = 2500
LR       = 2e-8

AMPLITUDES    = [50e-9]
SIGMA_FILTERS = [150e-6]
N_SEEDS       = 1

# -----------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------

def rmse_nm(h_pred, h_true):
    return float(torch.sqrt(torch.mean((h_pred - h_true)**2)).item()) * 1e9

def psnr_db(h_pred, h_true):
    mse = float(torch.mean((h_pred - h_true)**2).item())
    peak = float(h_true.abs().max().item())
    if mse < 1e-30: return 100.0
    return 10.0 * np.log10(peak**2 / mse)

def ncc(h_pred, h_true):
    a = h_pred - h_pred.mean()
    b = h_true - h_true.mean()
    denom = torch.sqrt(torch.sum(a**2) * torch.sum(b**2))
    if denom < 1e-30: return 0.0
    return float((torch.sum(a * b) / denom).item())

# -----------------------------------------------------------------------
# Reconstruction
# -----------------------------------------------------------------------

sensor = HolographicSensor(
    wavelength=wavelength, mem_res=MEM_RES, mem_pitch=mem_pitch,
    cmos_res=CMOS_RES, grid_res=GRID_RES, distance=DISTANCE, device=device,
).to(device)

def reconstruct(h_true):
    with torch.no_grad():
        I_target = sensor(h_true)

    h_pred = torch.zeros(MEM_RES, MEM_RES, dtype=torch.float32,
                         device=device, requires_grad=True)
    opt = torch.optim.Adam([h_pred], lr=LR)

    for i in range(N_ITER):
        opt.zero_grad()
        I_pred = sensor(h_pred)
        loss = torch.mean((I_pred - I_target)**2)
        loss.backward()
        opt.step()

    return h_pred.detach()

# -----------------------------------------------------------------------
# Main evaluation
# -----------------------------------------------------------------------

print(f"\n{'='*70}")
print(f"  Random Gaussian Surface Reconstruction Evaluation")
print(f"  Distance={DISTANCE*1e3:.0f}mm, N_ITER={N_ITER}, LR={LR}")
print(f"  {N_SEEDS} random seeds per configuration")
print(f"{'='*70}")

all_results = []

for amp in AMPLITUDES:
    for sigma in SIGMA_FILTERS:
        amp_nm = amp * 1e9
        sigma_um = sigma * 1e6
        label = f"amp={amp_nm:.0f}nm sigma={sigma_um:.0f}um"
        print(f"\n--- {label} ---")

        psnrs, rmses, nccs = [], [], []

        for seed in range(N_SEEDS):
            t0 = time.time()
            h_true = make_h_random(N=MEM_RES, dx=mem_pitch, device=device,
                                   seed=seed, amplitude=amp,
                                   sigma_filter=sigma)
            h_pred = reconstruct(h_true)

            p = psnr_db(h_pred, h_true)
            r = rmse_nm(h_pred, h_true)
            n = ncc(h_pred, h_true)
            dt = time.time() - t0

            psnrs.append(p)
            rmses.append(r)
            nccs.append(n)
            print(f"  seed={seed:2d}  PSNR={p:5.1f}dB  RMSE={r:6.1f}nm  NCC={n:.3f}  ({dt:.1f}s)")

        result = dict(
            amp=amp, sigma=sigma, label=label,
            psnr_mean=np.mean(psnrs), psnr_std=np.std(psnrs),
            psnr_min=np.min(psnrs), psnr_max=np.max(psnrs),
            rmse_mean=np.mean(rmses), rmse_std=np.std(rmses),
            rmse_min=np.min(rmses), rmse_max=np.max(rmses),
            ncc_mean=np.mean(nccs), ncc_std=np.std(nccs),
            ncc_min=np.min(nccs), ncc_max=np.max(nccs),
        )
        all_results.append(result)

        print(f"  SUMMARY: PSNR={result['psnr_mean']:.1f}+/-{result['psnr_std']:.1f}dB  "
              f"RMSE={result['rmse_mean']:.1f}+/-{result['rmse_std']:.1f}nm  "
              f"NCC={result['ncc_mean']:.3f}+/-{result['ncc_std']:.3f}")

# -----------------------------------------------------------------------
# Summary table
# -----------------------------------------------------------------------

print(f"\n{'='*90}")
print(f"  STATISTICAL SUMMARY ({N_SEEDS} seeds per config)")
print(f"{'='*90}")
print(f"  {'Amplitude':>10s} | {'Sigma':>8s} | {'PSNR mean+/-std':>16s} | {'RMSE mean+/-std':>18s} | {'NCC mean+/-std':>16s}")
print(f"  {'-'*10:>10s}-+-{'-'*8:>8s}-+-{'-'*16:>16s}-+-{'-'*18:>18s}-+-{'-'*16:>16s}")

for r in all_results:
    amp_str = f"{r['amp']*1e9:.0f}nm"
    sig_str = f"{r['sigma']*1e6:.0f}um"
    psnr_str = f"{r['psnr_mean']:5.1f}+/-{r['psnr_std']:4.1f}"
    rmse_str = f"{r['rmse_mean']:6.1f}+/-{r['rmse_std']:5.1f}"
    ncc_str = f"{r['ncc_mean']:5.3f}+/-{r['ncc_std']:5.3f}"
    print(f"  {amp_str:>10s} | {sig_str:>8s} | {psnr_str:>16s} | {rmse_str:>18s} | {ncc_str:>16s}")

print(f"{'='*90}")

# -----------------------------------------------------------------------
# Save text results
# -----------------------------------------------------------------------

with open(OUT / "random_surface_results.txt", "w") as f:
    f.write("="*90 + "\n")
    f.write("  Random Gaussian Surface Reconstruction - Statistical Results\n")
    f.write(f"  Distance={DISTANCE*1e3:.0f}mm, N_ITER={N_ITER}, LR={LR}\n")
    f.write(f"  MEM_RES={MEM_RES}, CMOS_RES={CMOS_RES}, GRID_RES={GRID_RES}\n")
    f.write(f"  {N_SEEDS} random seeds per configuration\n")
    f.write("="*90 + "\n\n")

    f.write(f"  {'Amplitude':>10s} | {'Sigma':>8s} | {'PSNR (dB)':>20s} | {'RMSE (nm)':>22s} | {'NCC':>20s}\n")
    f.write(f"  {'':>10s} | {'':>8s} | {'mean +/- std':>20s} | {'mean +/- std':>22s} | {'mean +/- std':>20s}\n")
    f.write(f"  {'-'*10}-+-{'-'*8}-+-{'-'*20}-+-{'-'*22}-+-{'-'*20}\n")

    for r in all_results:
        amp_str = f"{r['amp']*1e9:.0f}nm"
        sig_str = f"{r['sigma']*1e6:.0f}um"
        psnr_str = f"{r['psnr_mean']:5.1f} +/- {r['psnr_std']:4.1f} [{r['psnr_min']:.1f}-{r['psnr_max']:.1f}]"
        rmse_str = f"{r['rmse_mean']:6.1f} +/- {r['rmse_std']:5.1f} [{r['rmse_min']:.1f}-{r['rmse_max']:.1f}]"
        ncc_str = f"{r['ncc_mean']:5.3f} +/- {r['ncc_std']:5.3f} [{r['ncc_min']:.3f}-{r['ncc_max']:.3f}]"
        f.write(f"  {amp_str:>10s} | {sig_str:>8s} | {psnr_str:>20s} | {rmse_str:>22s} | {ncc_str:>20s}\n")

    f.write("\n")

print(f"\nSaved -> {OUT / 'random_surface_results.txt'}")

# -----------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------

plt.rcParams.update(STYLE)

n_amp = len(AMPLITUDES)
n_sig = len(SIGMA_FILTERS)

# Fig 1: PSNR heatmap
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Random Surface Reconstruction Performance", fontsize=13, color="#e6edf3")
fig.patch.set_facecolor("#0d1117")

metrics = [
    ("PSNR [dB]", "psnr_mean", "RdYlGn"),
    ("RMSE [nm]", "rmse_mean", "RdYlGn_r"),
    ("NCC", "ncc_mean", "RdYlGn"),
]

for col, (title, key, cmap) in enumerate(metrics):
    ax = axes[col]
    ax.set_facecolor("#0d1117")

    data = np.zeros((n_amp, n_sig))
    for r in all_results:
        ai = AMPLITUDES.index(r["amp"])
        si = SIGMA_FILTERS.index(r["sigma"])
        data[ai, si] = r[key]

    im = ax.imshow(data, cmap=cmap, aspect="auto", origin="lower")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n_sig))
    ax.set_xticklabels([f"{s*1e6:.0f}" for s in SIGMA_FILTERS])
    ax.set_yticks(range(n_amp))
    ax.set_yticklabels([f"{a*1e9:.0f}" for a in AMPLITUDES])
    ax.set_xlabel("Sigma [um]", fontsize=8, color="#8b949e")
    ax.set_ylabel("Amplitude [nm]", fontsize=8, color="#8b949e")
    ax.set_title(title, fontsize=10, color="#e6edf3")
    ax.tick_params(colors="#8b949e", labelsize=7)

    # Annotate cells
    for i in range(n_amp):
        for j in range(n_sig):
            ax.text(j, i, f"{data[i,j]:.1f}", ha="center", va="center",
                    fontsize=7, color="black", fontweight="bold")

plt.tight_layout()
plt.savefig(OUT / "random_surface_heatmap.png", dpi=150, bbox_inches="tight",
            facecolor="#0d1117")
plt.close()
print(f"Saved -> {OUT / 'random_surface_heatmap.png'}")

# Fig 2: Box plots by amplitude
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Random Surface Performance Distribution (by amplitude)",
             fontsize=13, color="#e6edf3")
fig.patch.set_facecolor("#0d1117")

for col, (title, key) in enumerate([("PSNR [dB]", "psnr"), ("RMSE [nm]", "rmse"), ("NCC", "ncc")]):
    ax = axes[col]
    ax.set_facecolor("#0d1117")

    # Group by amplitude (aggregate over sigma)
    for ai, amp in enumerate(AMPLITUDES):
        amp_results = [r for r in all_results if r["amp"] == amp]
        means = [r[f"{key}_mean"] for r in amp_results]
        ax.bar(ai, np.mean(means), color="#58a6ff", alpha=0.7, width=0.6)
        ax.errorbar(ai, np.mean(means), yerr=np.std(means),
                    color="#e6edf3", capsize=4, capthick=1.5)

    ax.set_xticks(range(n_amp))
    ax.set_xticklabels([f"{a*1e9:.0f}" for a in AMPLITUDES])
    ax.set_xlabel("Amplitude [nm]", fontsize=8, color="#8b949e")
    ax.set_ylabel(title, fontsize=8, color="#8b949e")
    ax.set_title(title, fontsize=10, color="#e6edf3")
    ax.tick_params(colors="#8b949e", labelsize=7)
    ax.grid(color="#30363d", lw=0.4, ls=":", axis="y")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

plt.tight_layout()
plt.savefig(OUT / "random_surface_bars.png", dpi=150, bbox_inches="tight",
            facecolor="#0d1117")
plt.close()
print(f"Saved -> {OUT / 'random_surface_bars.png'}")

print("\nDone.")
