"""
lr_sweep_big.py
---------------
Wide log-scale learning-rate sweep for the inverse problem.

Uses the scaled parameterization  h = h_var * H_SCALE  so that Adam behaves
normally and lr ~ 1e-3 is the natural starting point (consistent with
standard ML training practice).  Sweeps 8 decades on a log scale.

Output:
  output/lr_sweep_big/
    lr_sweep.png             - qualitative: PSNR/RMSE vs lr + loss curves
    lr_sweep_results.txt     - quantitative table
    lr_sweep_results.json    - machine-readable copy
"""

import sys, time, json, functools
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

print = functools.partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).parent.parent))
from sensor_model  import HolographicSensor, gaussian_bump
from sensor_params import wavelength, mem_pitch
from sensor_utils  import STYLE

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "output" / "lr_sweep_big"
OUT.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
MEM_RES  = 128
CMOS_RES = 256
GRID_RES = 384
DISTANCE = 5e-3
AMP      = 200e-9
SIGMA    = 150e-6
H_SCALE  = 500e-9
N_ITER   = 2000

# Wide log-scale sweep. Big steps across the full dynamic range.
LRS = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1e0]

# -----------------------------------------------------------------------
# Forward model + ground truth
# -----------------------------------------------------------------------
sensor = HolographicSensor(
    wavelength=wavelength, mem_res=MEM_RES, mem_pitch=mem_pitch,
    cmos_res=CMOS_RES, grid_res=GRID_RES, distance=DISTANCE, device=device,
).to(device)

h_true = gaussian_bump(N=MEM_RES, dx=mem_pitch, amplitude=AMP,
                       sigma=SIGMA, device=device)
with torch.no_grad():
    I_target = sensor(h_true)

def rmse_nm(a, b): return float(torch.sqrt(torch.mean((a - b)**2))) * 1e9
def psnr_db(a, b):
    mse = float(torch.mean((a - b)**2))
    peak = float(b.abs().max())
    if mse < 1e-30: return 100.0
    return 10.0 * np.log10(peak**2 / mse)

def run(lr, n_iter=N_ITER):
    h_var = torch.zeros(MEM_RES, MEM_RES, dtype=torch.float32,
                        device=device, requires_grad=True)
    opt = torch.optim.Adam([h_var], lr=lr)
    losses = []
    diverged = False
    for i in range(n_iter):
        opt.zero_grad()
        I_pred = sensor(h_var * H_SCALE)
        loss = torch.mean((I_pred - I_target)**2)
        loss.backward()
        opt.step()
        lv = float(loss.detach())
        losses.append(lv)
        if not np.isfinite(lv):
            diverged = True
            break
    h_final = (h_var.detach() * H_SCALE)
    return h_final, losses, diverged

# -----------------------------------------------------------------------
# Run sweep
# -----------------------------------------------------------------------
print(f"\nWide log-scale LR sweep  |  h = h_var * {H_SCALE*1e9:.0f}nm  |  {N_ITER} iters\n")
print(f"  {'lr':>8}  {'PSNR [dB]':>10}  {'RMSE [nm]':>10}  {'final loss':>12}  {'time [s]':>8}  status")
print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*8}  {'-'*8}")

results = []
for lr in LRS:
    t0 = time.time()
    h, losses, diverged = run(lr)
    p = psnr_db(h, h_true) if not diverged else float("nan")
    r = rmse_nm(h, h_true) if not diverged else float("nan")
    dt = time.time() - t0
    final_loss = losses[-1] if losses else float("nan")
    status = "DIVERGED" if diverged else "ok"
    print(f"  {lr:>8.0e}  {p:>10.2f}  {r:>10.3f}  {final_loss:>12.3e}  {dt:>8.2f}  {status}")
    results.append(dict(lr=lr, psnr=p, rmse=r, final_loss=final_loss,
                        wall_s=dt, status=status, losses=losses,
                        h=h.cpu().numpy()))

# -----------------------------------------------------------------------
# Quantitative output files
# -----------------------------------------------------------------------
valid = [r for r in results if r["status"] == "ok" and np.isfinite(r["psnr"])]
best  = max(valid, key=lambda x: x["psnr"]) if valid else None

lines = []
lines.append("lr_sweep_big.py — Wide log-scale Learning Rate sweep")
lines.append("=" * 60)
lines.append(f"Parameterization  : h = h_var * {H_SCALE*1e9:.0f} nm")
lines.append(f"Test case         : Gaussian bump, A={AMP*1e9:.0f}nm, sigma={SIGMA*1e6:.0f}um")
lines.append(f"Distance          : {DISTANCE*1e3:.1f} mm")
lines.append(f"Grid              : mem={MEM_RES}, cmos={CMOS_RES}, grid={GRID_RES}")
lines.append(f"Iterations        : {N_ITER}   Optimizer: Adam   Loss: MSE   Init: zeros")
lines.append("")
lines.append(f"  {'lr':>8}  {'PSNR [dB]':>10}  {'RMSE [nm]':>10}  {'final loss':>12}  {'status':>8}")
lines.append(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*8}")
for r in results:
    lines.append(f"  {r['lr']:>8.0e}  {r['psnr']:>10.2f}  {r['rmse']:>10.3f}"
                 f"  {r['final_loss']:>12.3e}  {r['status']:>8}")
lines.append("")
if best is not None:
    lines.append(f"BEST lr = {best['lr']:.0e}   PSNR={best['psnr']:.2f} dB   RMSE={best['rmse']:.3f} nm")
txt = "\n".join(lines)
(OUT / "lr_sweep_results.txt").write_text(txt, encoding="utf-8")
print("\n" + txt)

(OUT / "lr_sweep_results.json").write_text(json.dumps(
    [dict(lr=r["lr"], psnr=r["psnr"], rmse=r["rmse"],
          final_loss=r["final_loss"], status=r["status"], wall_s=r["wall_s"])
     for r in results], indent=2))

# -----------------------------------------------------------------------
# Qualitative plot
# -----------------------------------------------------------------------
plt.rcParams.update(STYLE)

v_lrs   = [r["lr"]   for r in valid]
v_psnrs = [r["psnr"] for r in valid]
v_rmses = [r["rmse"] for r in valid]

fig, axes = plt.subplots(1, 3, figsize=(20, 5.2))
fig.suptitle(
    f"Wide log-scale LR sweep  |  h = h_var * {H_SCALE*1e9:.0f}nm  |  Adam, MSE, {N_ITER} iters",
    fontsize=12, color="#e6edf3")
fig.patch.set_facecolor("#0d1117")

def _style(ax):
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e", labelsize=7)
    ax.grid(color="#30363d", lw=0.4, ls=":")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

ax = axes[0]
if v_lrs:
    ax.semilogx(v_lrs, v_psnrs, "o-", color="#58a6ff", lw=1.5, markersize=6)
    best_lr = v_lrs[v_psnrs.index(max(v_psnrs))]
    ax.axvline(best_lr, color="#3fb950", lw=0.8, ls="--", alpha=0.7,
               label=f"best {best_lr:.0e}")
    ax.legend(fontsize=7, labelcolor="#e6edf3", framealpha=0.3)
ax.set_xlabel("Learning Rate", fontsize=9, color="#8b949e")
ax.set_ylabel("PSNR [dB]", fontsize=9, color="#8b949e")
ax.set_title("PSNR vs LR", fontsize=10, color="#e6edf3")
_style(ax)

ax = axes[1]
if v_lrs:
    ax.loglog(v_lrs, v_rmses, "o-", color="#f78166", lw=1.5, markersize=6)
ax.set_xlabel("Learning Rate", fontsize=9, color="#8b949e")
ax.set_ylabel("RMSE [nm]", fontsize=9, color="#8b949e")
ax.set_title("RMSE vs LR", fontsize=10, color="#e6edf3")
_style(ax)

ax = axes[2]
cmap = plt.get_cmap("viridis")
for idx, r in enumerate(results):
    c = cmap(idx / max(len(results) - 1, 1))
    label = f"lr={r['lr']:.0e}" + ("" if r["status"] == "ok" else " (div)")
    if any(np.isfinite(l) for l in r["losses"]):
        ax.plot(r["losses"], color=c, lw=1.0, label=label)
ax.set_yscale("log")
ax.set_xlabel("iteration", fontsize=9, color="#8b949e")
ax.set_ylabel("loss", fontsize=9, color="#8b949e")
ax.set_title("Loss curves", fontsize=10, color="#e6edf3")
ax.legend(fontsize=6, labelcolor="#e6edf3", framealpha=0.3, ncol=2)
_style(ax)

plt.tight_layout()
out_path = OUT / "lr_sweep.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"\nSaved -> {out_path}")
