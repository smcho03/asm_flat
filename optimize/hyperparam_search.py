"""
hyperparam_search.py
--------------------
Systematic hyperparameter search for holographic reconstruction.

Swept parameters:
  1. Learning rate     : [1e-11, 5e-11, 1e-10, 5e-10, 1e-9, 5e-9]
  2. Optimizer         : Adam, L-BFGS
  3. Regularization    : none, TV, Laplacian, TV+Lap
  4. Loss function     : MSE, L1, log-MSE
  5. Initialization    : zeros, Gaussian prior, back-propagation estimate
  6. Distance          : [1, 3, 5, 10] mm
  7. Iteration count   : run until convergence (up to 3000)

Uses single_bump (200nm, sigma=150um) as the canonical test case.

Output:
  optimize/output/hyperparam_*.png
"""

import sys, os, time, itertools
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from sensor_model import HolographicSensor, gaussian_bump
from sensor_params import wavelength, mem_pitch
from sensor_utils import STYLE

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

MEM_RES  = 128
CMOS_RES = 256
GRID_RES = 384
DISTANCE = 5e-3
AMP      = 200e-9
SIGMA    = 150e-6

# -----------------------------------------------------------------------
# Ground truth
# -----------------------------------------------------------------------
sensor = HolographicSensor(
    wavelength=wavelength, mem_res=MEM_RES, mem_pitch=mem_pitch,
    cmos_res=CMOS_RES, grid_res=GRID_RES, distance=DISTANCE, device=device,
).to(device)

h_true = gaussian_bump(N=MEM_RES, dx=mem_pitch, amplitude=AMP,
                       sigma=SIGMA, device=device)
with torch.no_grad():
    I_target = sensor(h_true)

print(f"h_true peak: {h_true.max()*1e9:.1f} nm")
print(f"I_target range: [{I_target.min():.4f}, {I_target.max():.4f}]")

# -----------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------
def rmse_nm(a, b):
    return float(torch.sqrt(torch.mean((a - b)**2))) * 1e9

def psnr_db(a, b):
    mse = float(torch.mean((a - b)**2))
    peak = float(b.abs().max())
    if mse < 1e-30: return 100.0
    return 10.0 * np.log10(peak**2 / mse)

# -----------------------------------------------------------------------
# Initialization strategies
# -----------------------------------------------------------------------
def init_zeros():
    return torch.zeros(MEM_RES, MEM_RES, dtype=torch.float32,
                       device=device, requires_grad=True)

def init_small_random():
    """Band-limited random init — no GT knowledge, but breaks symmetry."""
    torch.manual_seed(0)
    noise = torch.randn(MEM_RES, MEM_RES, dtype=torch.float32, device=device)
    # Low-pass filter to membrane-scale features
    f = torch.fft.fftfreq(MEM_RES, d=mem_pitch)
    Fx, Fy = torch.meshgrid(f, f, indexing="ij")
    sigma_f = 1.0 / (200e-6)  # cutoff ~ 200um features
    lpf = torch.exp(-0.5 * (Fx**2 + Fy**2) / sigma_f**2)
    noise_c = torch.fft.fft2(noise) * lpf
    smooth = torch.fft.ifft2(noise_c).real
    # Scale to ~10% of expected amplitude range
    smooth = smooth * (AMP * 0.1 / smooth.abs().max().clamp(min=1e-30))
    return smooth.clone().detach().requires_grad_(True)

def init_flat_perturb():
    """Zeros + tiny uniform noise — breaks symmetry without GT bias."""
    torch.manual_seed(1)
    h = torch.randn(MEM_RES, MEM_RES, dtype=torch.float32, device=device) * (AMP * 0.01)
    return h.clone().detach().requires_grad_(True)

INITS = {
    "zeros": init_zeros,
    "small_random": init_small_random,
    "flat_perturb": init_flat_perturb,
}

# -----------------------------------------------------------------------
# Loss functions
# -----------------------------------------------------------------------
def loss_mse(I_pred, I_tgt):
    return torch.mean((I_pred - I_tgt)**2)

def loss_l1(I_pred, I_tgt):
    return torch.mean(torch.abs(I_pred - I_tgt))

def loss_log_mse(I_pred, I_tgt):
    """Log-space MSE — weights low-intensity regions more equally."""
    eps = 1e-8
    return torch.mean((torch.log(I_pred + eps) - torch.log(I_tgt + eps))**2)

LOSSES = {
    "MSE": loss_mse,
    "L1": loss_l1,
    "log_MSE": loss_log_mse,
}

# -----------------------------------------------------------------------
# Regularizations
# -----------------------------------------------------------------------
def reg_none(h):
    return torch.tensor(0.0, device=device)

def reg_tv(h, lam=1e4):
    return lam * (torch.mean(torch.abs(h[1:, :] - h[:-1, :])) +
                  torch.mean(torch.abs(h[:, 1:] - h[:, :-1])))

def reg_lap(h, lam=1e16):
    lap = (h[2:, 1:-1] + h[:-2, 1:-1] +
           h[1:-1, 2:] + h[1:-1, :-2] - 4.0 * h[1:-1, 1:-1])
    return lam * torch.mean(lap**2)

def reg_tv_lap(h, lam_tv=1e4, lam_lap=1e16):
    return reg_tv(h, lam_tv) + reg_lap(h, lam_lap)

REGS = {
    "none": reg_none,
    "TV": reg_tv,
    "Laplacian": reg_lap,
    "TV+Lap": reg_tv_lap,
}

# -----------------------------------------------------------------------
# Single reconstruction run
# -----------------------------------------------------------------------
def run_recon(init_fn, loss_fn, reg_fn, lr, optimizer_type="Adam",
              n_iter=2000, log_every=500):
    h_pred = init_fn()
    if optimizer_type == "Adam":
        opt = torch.optim.Adam([h_pred], lr=lr)
    elif optimizer_type == "LBFGS":
        opt = torch.optim.LBFGS([h_pred], lr=lr, max_iter=20,
                                 history_size=10, line_search_fn="strong_wolfe")
    else:
        raise ValueError(optimizer_type)

    losses = []
    psnrs = []

    for i in range(n_iter):
        if optimizer_type == "LBFGS":
            def closure():
                opt.zero_grad()
                I_pred = sensor(h_pred)
                data_loss = loss_fn(I_pred, I_target)
                r = reg_fn(h_pred)
                total = data_loss + r
                total.backward()
                return total
            loss_val = opt.step(closure)
            losses.append(float(loss_val))
        else:
            opt.zero_grad()
            I_pred = sensor(h_pred)
            data_loss = loss_fn(I_pred, I_target)
            r = reg_fn(h_pred)
            total = data_loss + r
            total.backward()
            opt.step()
            losses.append(float(total.detach()))

        if (i + 1) % log_every == 0:
            p = psnr_db(h_pred.detach(), h_true)
            psnrs.append((i + 1, p))

    final_psnr = psnr_db(h_pred.detach(), h_true)
    final_rmse = rmse_nm(h_pred.detach(), h_true)
    return h_pred.detach(), losses, final_psnr, final_rmse

# =======================================================================
# Experiment 1: Learning Rate sweep (Adam, MSE, no reg)
# =======================================================================
print("\n=== Experiment 1: Learning Rate ===")
LRS = [1e-11, 5e-11, 1e-10, 5e-10, 1e-9, 5e-9]
lr_results = []
for lr in LRS:
    t0 = time.time()
    h, losses, p, r = run_recon(init_zeros, loss_mse, reg_none, lr,
                                 n_iter=2000)
    dt = time.time() - t0
    print(f"  lr={lr:.0e}  PSNR={p:.1f}dB  RMSE={r:.1f}nm  ({dt:.1f}s)")
    lr_results.append(dict(lr=lr, losses=losses, psnr=p, rmse=r, h=h.cpu().numpy()))

# =======================================================================
# Experiment 2: Optimizer comparison (best lr from exp1)
# =======================================================================
print("\n=== Experiment 2: Optimizer ===")
best_lr_adam = max(lr_results, key=lambda x: x["psnr"])["lr"]
print(f"  Best Adam lr: {best_lr_adam:.0e}")

opt_results = {}
# Adam with best lr
t0 = time.time()
h, losses, p, r = run_recon(init_zeros, loss_mse, reg_none, best_lr_adam,
                             optimizer_type="Adam", n_iter=2000)
dt = time.time() - t0
print(f"  Adam lr={best_lr_adam:.0e}  PSNR={p:.1f}dB  RMSE={r:.1f}nm  ({dt:.1f}s)")
opt_results["Adam"] = dict(losses=losses, psnr=p, rmse=r, h=h.cpu().numpy())

# L-BFGS (needs different lr scale)
for lbfgs_lr in [1e-11, 1e-10, 1e-9]:
    t0 = time.time()
    try:
        h, losses, p, r = run_recon(init_zeros, loss_mse, reg_none, lbfgs_lr,
                                     optimizer_type="LBFGS", n_iter=200)
        dt = time.time() - t0
        print(f"  L-BFGS lr={lbfgs_lr:.0e}  PSNR={p:.1f}dB  RMSE={r:.1f}nm  ({dt:.1f}s)")
        opt_results[f"LBFGS_lr{lbfgs_lr:.0e}"] = dict(losses=losses, psnr=p, rmse=r, h=h.cpu().numpy())
    except Exception as e:
        print(f"  L-BFGS lr={lbfgs_lr:.0e}  FAILED: {e}")

# =======================================================================
# Experiment 3: Loss function
# =======================================================================
print("\n=== Experiment 3: Loss Function ===")
loss_results = {}
for name, fn in LOSSES.items():
    t0 = time.time()
    h, losses, p, r = run_recon(init_zeros, fn, reg_none, best_lr_adam,
                                 n_iter=2000)
    dt = time.time() - t0
    print(f"  {name:10s}  PSNR={p:.1f}dB  RMSE={r:.1f}nm  ({dt:.1f}s)")
    loss_results[name] = dict(losses=losses, psnr=p, rmse=r, h=h.cpu().numpy())

# =======================================================================
# Experiment 4: Regularization
# =======================================================================
print("\n=== Experiment 4: Regularization ===")
reg_results = {}
for name, fn in REGS.items():
    t0 = time.time()
    h, losses, p, r = run_recon(init_zeros, loss_mse, fn, best_lr_adam,
                                 n_iter=2000)
    dt = time.time() - t0
    print(f"  {name:10s}  PSNR={p:.1f}dB  RMSE={r:.1f}nm  ({dt:.1f}s)")
    reg_results[name] = dict(losses=losses, psnr=p, rmse=r, h=h.cpu().numpy())

# =======================================================================
# Experiment 5: Initialization
# =======================================================================
print("\n=== Experiment 5: Initialization ===")
init_results = {}
for name, fn in INITS.items():
    t0 = time.time()
    h, losses, p, r = run_recon(fn, loss_mse, reg_none, best_lr_adam,
                                 n_iter=2000)
    dt = time.time() - t0
    print(f"  {name:18s}  PSNR={p:.1f}dB  RMSE={r:.1f}nm  ({dt:.1f}s)")
    init_results[name] = dict(losses=losses, psnr=p, rmse=r, h=h.cpu().numpy())

# =======================================================================
# Experiment 6: Best combination (from above) with more iterations
# =======================================================================
print("\n=== Experiment 6: Best combination with 5000 iterations ===")

# Find best from each experiment
best_loss = max(loss_results, key=lambda k: loss_results[k]["psnr"])
best_reg = max(reg_results, key=lambda k: reg_results[k]["psnr"])
best_init = max(init_results, key=lambda k: init_results[k]["psnr"])

print(f"  Best lr: {best_lr_adam:.0e}")
print(f"  Best loss: {best_loss}")
print(f"  Best reg: {best_reg}")
print(f"  Best init: {best_init}")

t0 = time.time()
h_best, losses_best, p_best, r_best = run_recon(
    INITS[best_init], LOSSES[best_loss], REGS[best_reg], best_lr_adam,
    n_iter=5000, log_every=1000)
dt = time.time() - t0
print(f"  FINAL:  PSNR={p_best:.1f}dB  RMSE={r_best:.1f}nm  ({dt:.1f}s)")

# =======================================================================
# Visualization
# =======================================================================
plt.rcParams.update(STYLE)
m_um = (np.arange(MEM_RES) - (MEM_RES - 1) * 0.5) * mem_pitch * 1e6
mid = MEM_RES // 2
h_true_nm = h_true.cpu().numpy() * 1e9

def _style(ax):
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e", labelsize=7)
    ax.grid(color="#30363d", lw=0.4, ls=":")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

# --- Fig 1: LR sweep ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Experiment 1: Learning Rate Sweep (Adam, MSE, no reg, 2000 iters)",
             fontsize=12, color="#e6edf3")
fig.patch.set_facecolor("#0d1117")

# Loss curves
ax = axes[0]
for r in lr_results:
    ax.plot(r["losses"], lw=1.0, label=f"lr={r['lr']:.0e}")
ax.set_yscale("log")
ax.set_xlabel("iteration", fontsize=8, color="#8b949e")
ax.set_ylabel("loss", fontsize=8, color="#8b949e")
ax.set_title("Loss curves", fontsize=9, color="#e6edf3")
ax.legend(fontsize=6, labelcolor="#e6edf3", framealpha=0.3)
_style(ax)

# PSNR bar
ax = axes[1]
lrs_str = [f"{r['lr']:.0e}" for r in lr_results]
psnrs = [r["psnr"] for r in lr_results]
colors = ["#58a6ff" if p == max(psnrs) else "#8b949e" for p in psnrs]
bars = ax.bar(lrs_str, psnrs, color=colors, edgecolor="#30363d")
for bar, val in zip(bars, psnrs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{val:.1f}", ha="center", fontsize=7, color="#e6edf3")
ax.set_xlabel("learning rate", fontsize=8, color="#8b949e")
ax.set_ylabel("PSNR [dB]", fontsize=8, color="#8b949e")
ax.set_title("Final PSNR", fontsize=9, color="#e6edf3")
_style(ax)

# Cross-section of best vs worst
ax = axes[2]
best_lr_r = max(lr_results, key=lambda x: x["psnr"])
worst_lr_r = min(lr_results, key=lambda x: x["psnr"])
ax.plot(m_um, h_true_nm[mid, :], color="#8b949e", lw=2, ls=":", label="GT")
ax.plot(m_um, best_lr_r["h"][mid, :]*1e9, color="#58a6ff", lw=1.5,
        label=f"best lr={best_lr_r['lr']:.0e}")
ax.plot(m_um, worst_lr_r["h"][mid, :]*1e9, color="#f85149", lw=1.0, ls="--",
        label=f"worst lr={worst_lr_r['lr']:.0e}")
ax.set_xlabel("x [μm]", fontsize=8, color="#8b949e")
ax.set_ylabel("h [nm]", fontsize=8, color="#8b949e")
ax.set_title("Cross-section", fontsize=9, color="#e6edf3")
ax.legend(fontsize=7, labelcolor="#e6edf3", framealpha=0.3)
_style(ax)

plt.tight_layout()
plt.savefig(OUT / "hyperparam_lr.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"\nSaved -> {OUT / 'hyperparam_lr.png'}")

# --- Fig 2: All experiments summary ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Hyperparameter Search Summary — single_bump 200nm, d=5mm",
             fontsize=13, color="#e6edf3")
fig.patch.set_facecolor("#0d1117")

def bar_chart(ax, data_dict, title, key="psnr", ylabel="PSNR [dB]"):
    names = list(data_dict.keys())
    vals = [data_dict[k][key] for k in names]
    best_idx = vals.index(max(vals)) if key == "psnr" else vals.index(min(vals))
    colors = ["#58a6ff" if i == best_idx else "#8b949e" for i in range(len(vals))]
    bars = ax.bar(range(len(names)), vals, color=colors, edgecolor="#30363d")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=6, color="#8b949e", rotation=30, ha="right")
    ax.set_ylabel(ylabel, fontsize=8, color="#8b949e")
    ax.set_title(title, fontsize=9, color="#e6edf3")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{val:.1f}", ha="center", fontsize=6, color="#e6edf3")
    _style(ax)

# (0,0) LR - PSNR
lr_dict = {f"{r['lr']:.0e}": r for r in lr_results}
bar_chart(axes[0, 0], lr_dict, "Learning Rate", "psnr", "PSNR [dB]")

# (0,1) Optimizer - PSNR
bar_chart(axes[0, 1], opt_results, "Optimizer", "psnr", "PSNR [dB]")

# (0,2) Loss function - PSNR
bar_chart(axes[0, 2], loss_results, "Loss Function", "psnr", "PSNR [dB]")

# (1,0) Regularization - PSNR
bar_chart(axes[1, 0], reg_results, "Regularization", "psnr", "PSNR [dB]")

# (1,1) Initialization - PSNR
bar_chart(axes[1, 1], init_results, "Initialization", "psnr", "PSNR [dB]")

# (1,2) Final best combination cross-section
ax = axes[1, 2]
ax.plot(m_um, h_true_nm[mid, :], color="#8b949e", lw=2, ls=":", label="GT")
ax.plot(m_um, h_best.cpu().numpy()[mid, :]*1e9, color="#58a6ff", lw=1.5,
        label=f"Best combo\nPSNR={p_best:.1f}dB")
# Also show the baseline (zeros, MSE, no reg, best lr, 2000 iter)
baseline = lr_results[[r["lr"] for r in lr_results].index(best_lr_adam)]
ax.plot(m_um, baseline["h"][mid, :]*1e9, color="#e3b341", lw=1.0, ls="--",
        label=f"Baseline\nPSNR={baseline['psnr']:.1f}dB")
ax.set_xlabel("x [μm]", fontsize=8, color="#8b949e")
ax.set_ylabel("h [nm]", fontsize=8, color="#8b949e")
ax.set_title(f"Best: {best_init}+{best_loss}+{best_reg} (5000 iter)",
             fontsize=8, color="#e6edf3")
ax.legend(fontsize=6, labelcolor="#e6edf3", framealpha=0.3)
_style(ax)

plt.tight_layout()
plt.savefig(OUT / "hyperparam_summary.png", dpi=150, bbox_inches="tight",
            facecolor="#0d1117")
plt.close()
print(f"Saved -> {OUT / 'hyperparam_summary.png'}")

# --- Fig 3: Loss curves overlay ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Loss Curve Comparison", fontsize=12, color="#e6edf3")
fig.patch.set_facecolor("#0d1117")

# Loss fn comparison
ax = axes[0]
for name, r in loss_results.items():
    ax.plot(r["losses"], lw=1.0, label=f"{name} (PSNR={r['psnr']:.1f})")
ax.set_yscale("log")
ax.set_xlabel("iteration", fontsize=8, color="#8b949e")
ax.set_ylabel("loss", fontsize=8, color="#8b949e")
ax.set_title("Loss functions", fontsize=9, color="#e6edf3")
ax.legend(fontsize=7, labelcolor="#e6edf3", framealpha=0.3)
_style(ax)

# Init comparison
ax = axes[1]
for name, r in init_results.items():
    ax.plot(r["losses"], lw=1.0, label=f"{name} (PSNR={r['psnr']:.1f})")
ax.set_yscale("log")
ax.set_xlabel("iteration", fontsize=8, color="#8b949e")
ax.set_ylabel("loss", fontsize=8, color="#8b949e")
ax.set_title("Initialization strategies", fontsize=9, color="#e6edf3")
ax.legend(fontsize=7, labelcolor="#e6edf3", framealpha=0.3)
_style(ax)

plt.tight_layout()
plt.savefig(OUT / "hyperparam_loss_curves.png", dpi=150, bbox_inches="tight",
            facecolor="#0d1117")
plt.close()
print(f"Saved -> {OUT / 'hyperparam_loss_curves.png'}")

# --- Print final summary ---
print("\n" + "="*60)
print("  HYPERPARAMETER SEARCH SUMMARY")
print("="*60)
print(f"  Best learning rate : {best_lr_adam:.0e}")
print(f"  Best loss function : {best_loss}")
print(f"  Best regularization: {best_reg}")
print(f"  Best initialization: {best_init}")
print(f"  Best PSNR (5k iter): {p_best:.1f} dB")
print(f"  Best RMSE (5k iter): {r_best:.1f} nm")
print("="*60)
