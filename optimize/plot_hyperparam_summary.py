"""
plot_hyperparam_summary.py
--------------------------
Comprehensive hyperparameter summary:
  - Bar charts: PSNR + RMSE side by side for each hyperparameter
  - Numeric tables printed to console and saved as text

Uses exact numbers from completed experiment runs.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from sensor_utils import STYLE

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# ======================================================================
#  Exact numbers from completed runs
# ======================================================================

data = {
    # --- Learning rate (Adam, MSE, no reg, zeros, 2000 iter) ---
    "Learning Rate": {
        "labels": ["1e-11", "5e-11", "1e-10", "5e-10", "1e-9", "5e-9", "1e-8", "1.5e-8", "2e-8", "2.5e-8", "3e-8"],
        "psnr":   [13.2,    12.1,    12.8,    19.9,    21.4,   22.1,   22.5,   22.8,     23.6,   10.8,     9.0],
        "rmse":   [43.9,    49.4,    45.8,    20.2,    17.1,   15.6,   14.9,   14.5,     13.3,   57.5,     70.7],
        "best_idx": 8,
        "note": "Adam, MSE, no reg, zeros init, 2000 iter  [2.5e-8+ diverges]",
    },
    # --- Optimizer (best lr=5e-9, MSE, no reg, zeros, 2000 iter) ---
    "Optimizer": {
        "labels": ["Adam\nlr=5e-9", "L-BFGS\nlr=1e-11", "L-BFGS\nlr=1e-10", "L-BFGS\nlr=1e-9"],
        "psnr":   [22.1,            13.6,                13.6,                13.6],
        "rmse":   [15.6,            41.5,                41.5,                41.5],
        "best_idx": 0,
        "note": "lr=5e-9, MSE, no reg, zeros, 2000 iter",
    },
    # --- Loss function (Adam, lr=5e-9, no reg, zeros, 2000 iter) ---
    "Loss Function": {
        "labels": ["MSE", "L1", "log-MSE"],
        "psnr":   [22.1,  19.4, 16.6],
        "rmse":   [15.6,  21.4, 29.4],
        "best_idx": 0,
        "note": "Adam, lr=5e-9, no reg, zeros, 2000 iter",
    },
    # --- Regularization (Adam, lr=5e-9, MSE, zeros, 2000 iter) ---
    "Regularization": {
        "labels": ["none", "TV\n(1e4)", "Laplacian\n(1e16)", "TV+Lap"],
        "psnr":   [22.1,   21.6,        13.8,                 13.8],
        "rmse":   [15.6,   16.5,        40.7,                 40.7],
        "best_idx": 0,
        "note": "Adam, lr=5e-9, MSE, zeros, 2000 iter",
    },
    # --- Initialization (Adam, lr=5e-9, MSE, no reg, 2000 iter) ---
    "Initialization": {
        "labels": ["zeros", "small_random\n(10% amp)", "flat_perturb\n(1% amp)"],
        "psnr":   [22.1,    20.8,                       21.0],
        "rmse":   [15.6,    18.2,                        17.9],
        "best_idx": 0,
        "note": "Adam, lr=5e-9, MSE, no reg, 2000 iter",
    },
    # --- Iteration count (best combo: Adam, lr=5e-9, MSE, no reg, zeros) ---
    "Iteration Count": {
        "labels": ["500", "2000", "5000"],
        "psnr":   [16.7,  22.1,   22.3],
        "rmse":   [29.4,  15.6,   15.3],
        "best_idx": 2,
        "note": "Adam, lr=5e-9, MSE, no reg, zeros  [d=5mm, single_bump 200nm]",
    },
    # --- Distance (amp=200nm, single_bump) ---
    "Distance": {
        "labels": ["1mm", "2mm", "3mm", "4mm", "5mm", "6mm", "7mm", "8mm", "10mm", "12mm", "15mm", "20mm"],
        "psnr":   [15.4,  15.4,  20.0,  21.5,  23.6,  24.9,  22.4,  20.6,  19.6,  17.7,  17.4,  20.9],
        "rmse":   [34.1,  33.8,  20.0,  16.9,  13.3,  11.4,  15.2,  18.6,  20.9,  26.1,  27.0,  18.1],
        "best_idx": 5,
        "note": "single_bump, amp=200nm, Adam lr=2e-8, 2000 iter",
    },
    # --- Amplitude (d=5mm, single_bump) ---
    "Amplitude (single_bump)": {
        "labels": ["100 nm\n(<lam/4)", "200 nm\n(~lam/4)", "500 nm\n(wrapping)", "1000 nm\n(severe)"],
        "psnr":   [17.5,             16.7,               14.5,                 13.9],
        "rmse":   [13.3,             29.4,               93.7,                 202.3],
        "best_idx": 0,
        "note": "single_bump, d=5mm, Adam lr=5e-10, 500 iter",
    },
}

# ======================================================================
#  Helper: double bar chart (PSNR + RMSE)
# ======================================================================

def _style(ax):
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e", labelsize=7)
    ax.grid(color="#30363d", lw=0.4, ls=":", axis="y")
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")

def double_bar(ax_psnr, ax_rmse, entry, title):
    labels    = entry["labels"]
    psnrs     = entry["psnr"]
    rmses     = entry["rmse"]
    best_idx  = entry["best_idx"]
    n = len(labels)
    x = np.arange(n)

    # PSNR
    colors_p = ["#58a6ff" if i == best_idx else "#30363d" for i in range(n)]
    bars = ax_psnr.bar(x, psnrs, color=colors_p, edgecolor="#8b949e", linewidth=0.5)
    for bar, val in zip(bars, psnrs):
        ax_psnr.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.3,
                     f"{val:.1f}", ha="center", va="bottom",
                     fontsize=7, color="#e6edf3")
    ax_psnr.set_xticks(x)
    ax_psnr.set_xticklabels(labels, fontsize=6.5, color="#8b949e")
    ax_psnr.set_ylabel("PSNR [dB]", fontsize=8, color="#8b949e")
    ax_psnr.set_title(title, fontsize=9, color="#e6edf3", pad=6)
    _style(ax_psnr)

    # RMSE
    colors_r = ["#f78166" if i == best_idx else "#30363d" for i in range(n)]
    bars = ax_rmse.bar(x, rmses, color=colors_r, edgecolor="#8b949e", linewidth=0.5)
    for bar, val in zip(bars, rmses):
        ax_rmse.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.5,
                     f"{val:.1f}", ha="center", va="bottom",
                     fontsize=7, color="#e6edf3")
    ax_rmse.set_xticks(x)
    ax_rmse.set_xticklabels(labels, fontsize=6.5, color="#8b949e")
    ax_rmse.set_ylabel("RMSE [nm]", fontsize=8, color="#8b949e")
    ax_rmse.set_title(title, fontsize=9, color="#e6edf3", pad=6)
    _style(ax_rmse)

# ======================================================================
#  Figure: one row per hyperparameter (PSNR left, RMSE right)
# ======================================================================

plt.rcParams.update(STYLE)

n_params = len(data)
fig, axes = plt.subplots(n_params, 2, figsize=(14, 3.5 * n_params))
fig.suptitle("Holographic Reconstruction - Hyperparameter Summary\n"
             "Test case: single_bump  |  lambda=632.8nm  |  d=5mm (unless noted)",
             fontsize=12, color="#e6edf3", y=1.005)
fig.patch.set_facecolor("#0d1117")

for row, (key, entry) in enumerate(data.items()):
    double_bar(axes[row, 0], axes[row, 1], entry, key)
    # small note text
    axes[row, 0].set_xlabel(entry["note"], fontsize=6, color="#484f58")
    axes[row, 1].set_xlabel(entry["note"], fontsize=6, color="#484f58")

    # highlight lambda/4 line on amplitude plot
    if key == "Amplitude (single_bump)":
        axes[row, 0].axvline(1.5, color="#e3b341", lw=0.8, ls="--", alpha=0.6)
        axes[row, 1].axvline(1.5, color="#e3b341", lw=0.8, ls="--", alpha=0.6)
        axes[row, 0].text(1.55, axes[row, 0].get_ylim()[1]*0.9,
                          "lam/4 = 158nm", fontsize=6, color="#e3b341")

plt.tight_layout()
out_path = OUT / "hyperparam_psnr_rmse_bars.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"Saved -> {out_path}")

# ======================================================================
#  Individual figures per hyperparameter
# ======================================================================

INDIV = OUT / "hyperparam_individual"
INDIV.mkdir(exist_ok=True)

slug_map = {
    "Learning Rate":          "01_learning_rate",
    "Optimizer":              "02_optimizer",
    "Loss Function":          "03_loss_function",
    "Regularization":         "04_regularization",
    "Initialization":         "05_initialization",
    "Iteration Count":        "06_iteration_count",
    "Distance":               "07_distance",
    "Amplitude (single_bump)":"08_amplitude",
}

for key, entry in data.items():
    fig, (ax_p, ax_r) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"Hyperparameter: {key}\n{entry['note']}",
                 fontsize=10, color="#e6edf3")
    fig.patch.set_facecolor("#0d1117")

    double_bar(ax_p, ax_r, entry, key)
    ax_p.set_xlabel(entry["note"], fontsize=6, color="#484f58")
    ax_r.set_xlabel(entry["note"], fontsize=6, color="#484f58")

    if key == "Amplitude (single_bump)":
        for ax in (ax_p, ax_r):
            ax.axvline(1.5, color="#e3b341", lw=0.8, ls="--", alpha=0.6)
            ax.text(1.55, ax.get_ylim()[1] * 0.9,
                    "lam/4 = 158nm", fontsize=6, color="#e3b341")

    plt.tight_layout()
    slug = slug_map.get(key, key.lower().replace(" ", "_"))
    p = INDIV / f"{slug}.png"
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  Saved -> {p}")

# ======================================================================
#  Numeric tables
# ======================================================================

SEP = "=" * 72

def print_table(name, entry):
    labels = entry["labels"]
    psnrs  = entry["psnr"]
    rmses  = entry["rmse"]
    best   = entry["best_idx"]
    note   = entry["note"]

    clean_labels = [l.replace("\n", " ") for l in labels]
    col_w = max(max(len(l) for l in clean_labels), 14)

    header = f"  {'':>{col_w}}  {'PSNR [dB]':>10}  {'RMSE [nm]':>10}"
    sep    = f"  {'-'*col_w}  {'-'*10}  {'-'*10}"

    lines = [SEP, f"  {name}", f"  ({note})", SEP, header, sep]
    for i, (l, p, r) in enumerate(zip(clean_labels, psnrs, rmses)):
        marker = " <-- best" if i == best else ""
        lines.append(f"  {l:>{col_w}}  {p:>10.1f}  {r:>10.1f}{marker}")
    lines.append("")
    return "\n".join(lines)

output_lines = [
    SEP,
    "  HOLOGRAPHIC RECONSTRUCTION - HYPERPARAMETER NUMERICAL RESULTS",
    "  Test: single_bump 200nm, lambda=632.8nm, MEM_RES=128, CMOS_RES=256",
    SEP,
    "",
]
for name, entry in data.items():
    output_lines.append(print_table(name, entry))

table_str = "\n".join(output_lines)
print(table_str.encode("ascii", errors="replace").decode("ascii"))

txt_path = OUT / "hyperparam_tables.txt"
with open(txt_path, "w", encoding="utf-8") as f:
    f.write(table_str)
print(f"Saved -> {txt_path}")
