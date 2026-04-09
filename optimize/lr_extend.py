"""
lr_extend.py
------------
Extend learning rate search beyond 5e-9.
Adam, MSE, no reg, zeros init, 2000 iter
Test case: single_bump 200nm sigma=150um, d=5mm
"""

import sys, time, functools
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

print = functools.partial(print, flush=True)

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

sensor = HolographicSensor(
    wavelength=wavelength, mem_res=MEM_RES, mem_pitch=mem_pitch,
    cmos_res=CMOS_RES, grid_res=GRID_RES, distance=DISTANCE, device=device,
).to(device)

h_true = gaussian_bump(N=MEM_RES, dx=mem_pitch, amplitude=AMP,
                       sigma=SIGMA, device=device)
with torch.no_grad():
    I_target = sensor(h_true)

def rmse_nm(a, b):
    return float(torch.sqrt(torch.mean((a - b)**2))) * 1e9

def psnr_db(a, b):
    mse = float(torch.mean((a - b)**2))
    peak = float(b.abs().max())
    if mse < 1e-30: return 100.0
    return 10.0 * np.log10(peak**2 / mse)

def run(lr, n_iter=2000):
    h_pred = torch.zeros(MEM_RES, MEM_RES, dtype=torch.float32,
                         device=device, requires_grad=True)
    opt = torch.optim.Adam([h_pred], lr=lr)
    losses = []
    for i in range(n_iter):
        opt.zero_grad()
        I_pred = sensor(h_pred)
        loss = torch.mean((I_pred - I_target)**2)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        if not np.isfinite(losses[-1]):
            print(f"    -> diverged at iter {i}")
            break
    return h_pred.detach(), losses

# Previously tested (for reference plot)
prev = [
    (1e-11, 13.2, 43.9),
    (5e-11, 12.1, 49.4),
    (1e-10, 12.8, 45.8),
    (5e-10, 19.9, 20.2),
    (1e-9,  21.4, 17.1),
    (5e-9,  22.1, 15.6),
]

# Extended range
NEW_LRS = [1e-8, 1.5e-8, 2e-8, 2.5e-8, 3e-8, 4e-8, 5e-8, 1e-7, 5e-7, 1e-6]

print("\n=== Extended LR search (beyond 5e-9) ===")
new_results = []
for lr in NEW_LRS:
    t0 = time.time()
    h, losses = run(lr)
    p = psnr_db(h, h_true)
    r = rmse_nm(h, h_true)
    dt = time.time() - t0
    status = "DIVERGED" if not np.isfinite(losses[-1]) else f"PSNR={p:.1f}dB  RMSE={r:.1f}nm"
    print(f"  lr={lr:.0e}  {status}  ({dt:.1f}s)")
    new_results.append(dict(lr=lr, psnr=p, rmse=r, losses=losses, h=h.cpu().numpy()))

# -----------------------------------------------------------------------
# Full LR table (prev + new)
# -----------------------------------------------------------------------
print("\n=== Full LR sweep results ===")
print(f"  {'LR':>8}  {'PSNR [dB]':>10}  {'RMSE [nm]':>10}")
print(f"  {'-'*8}  {'-'*10}  {'-'*10}")
for lr, p, r in prev:
    print(f"  {lr:>8.0e}  {p:>10.1f}  {r:>10.1f}")
for res in new_results:
    p = res['psnr']
    r = res['rmse']
    flag = "  <- DIVERGED" if not np.isfinite(p) else ""
    print(f"  {res['lr']:>8.0e}  {p:>10.1f}  {r:>10.1f}{flag}")

# -----------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------
plt.rcParams.update(STYLE)

all_lrs   = [x[0] for x in prev] + [r["lr"] for r in new_results]
all_psnrs = [x[1] for x in prev] + [r["psnr"] for r in new_results]
all_rmses = [x[2] for x in prev] + [r["rmse"] for r in new_results]

# Filter out non-finite
valid = [(lr, p, r) for lr, p, r in zip(all_lrs, all_psnrs, all_rmses)
         if np.isfinite(p) and np.isfinite(r)]
v_lrs, v_psnrs, v_rmses = zip(*valid) if valid else ([], [], [])

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Extended Learning Rate Search — Adam, MSE, no reg, zeros, 2000 iter",
             fontsize=12, color="#e6edf3")
fig.patch.set_facecolor("#0d1117")

def _style(ax):
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e", labelsize=7)
    ax.grid(color="#30363d", lw=0.4, ls=":")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

# PSNR vs LR
ax = axes[0]
ax.semilogx(v_lrs, v_psnrs, "o-", color="#58a6ff", lw=1.5, markersize=5)
best_lr = v_lrs[v_psnrs.index(max(v_psnrs))]
ax.axvline(best_lr, color="#3fb950", lw=0.8, ls="--", alpha=0.7)
ax.axvline(5e-9, color="#8b949e", lw=0.8, ls=":", alpha=0.7)
ax.text(5e-9*1.2, min(v_psnrs)+0.5, "prev best\n5e-9", fontsize=6, color="#8b949e")
ax.set_xlabel("Learning Rate", fontsize=8, color="#8b949e")
ax.set_ylabel("PSNR [dB]", fontsize=8, color="#8b949e")
ax.set_title("PSNR vs LR", fontsize=9, color="#e6edf3")
_style(ax)

# RMSE vs LR
ax = axes[1]
ax.semilogx(v_lrs, v_rmses, "o-", color="#f78166", lw=1.5, markersize=5)
ax.axvline(best_lr, color="#3fb950", lw=0.8, ls="--", alpha=0.7)
ax.set_xlabel("Learning Rate", fontsize=8, color="#8b949e")
ax.set_ylabel("RMSE [nm]", fontsize=8, color="#8b949e")
ax.set_title("RMSE vs LR", fontsize=9, color="#e6edf3")
_style(ax)

# Loss curves for new LRs
ax = axes[2]
colors = ["#58a6ff", "#3fb950", "#e3b341", "#f78166", "#bc8cff"]
for res, col in zip(new_results, colors):
    losses = res["losses"]
    if any(np.isfinite(l) for l in losses):
        ax.plot(losses, color=col, lw=1.0, label=f"lr={res['lr']:.0e}")
ax.set_yscale("log")
ax.set_xlabel("iteration", fontsize=8, color="#8b949e")
ax.set_ylabel("loss", fontsize=8, color="#8b949e")
ax.set_title("Loss curves (new LRs)", fontsize=9, color="#e6edf3")
ax.legend(fontsize=7, labelcolor="#e6edf3", framealpha=0.3)
_style(ax)

plt.tight_layout()
out_path = OUT / "lr_extended.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"\nSaved -> {out_path}")
