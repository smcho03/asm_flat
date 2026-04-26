"""
tv_regularizer_sweep.py
-----------------------
Test whether a total-variation (TV) smoothness prior resolves the
multi-basin ambiguity exposed by random_pattern_reconstruction.py.

We sweep the TV weight lambda on a single challenging pattern (seed r2,
A=100 nm, sigma=150 um) where raw=0.5 init gave only 23.9 dB despite
the per-pixel slope being 50x below the phase-wrap limit. If any
lambda lifts this case cleanly (> 35 dB), we apply the same lambda to
all 5 patterns in a follow-up run.

Loss:
    total = MSE(|U|^2, I_target)  +  lambda * TV(h)
where TV(h) = mean(|h[i+1,j]-h[i,j]| + |h[i,j+1]-h[i,j]|).

Output:
  output/tv_regularizer_sweep/
    sweep.png, sweep.txt, sweep.json
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
OUT  = ROOT / "output" / "tv_regularizer_sweep"
OUT.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

MEM, CMOS, GRID, DIST = 128, 256, 384, 5e-3
A_NM, SIGMA_SMOOTH_UM, SEED = 100, 150, 2    # r2: raw=0.5 gave 23.9 dB
A_M = A_NM * 1e-9
H_SCALE = 2.0 * A_M
LR = 3e-3
N_ITER = 15000

lam_opt = wavelength
dx = mem_pitch

sensor = HolographicSensor(
    wavelength=lam_opt, mem_res=MEM, mem_pitch=dx,
    cmos_res=CMOS, grid_res=GRID, distance=DIST, device=device,
).to(device)


def random_height(seed, A_m, sigma_smooth_um):
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((MEM, MEM)).astype(np.float32)
    sigma_px = sigma_smooth_um * 1e-6 / dx
    k_half = int(max(3 * sigma_px, 3))
    kx = np.arange(-k_half, k_half + 1)
    kernel_1d = np.exp(-(kx ** 2) / (2 * sigma_px ** 2))
    kernel_1d /= kernel_1d.sum()
    kernel_2d = np.outer(kernel_1d, kernel_1d).astype(np.float32)
    nt = torch.from_numpy(noise).unsqueeze(0).unsqueeze(0)
    kt = torch.from_numpy(kernel_2d).unsqueeze(0).unsqueeze(0)
    pad = k_half
    smooth = F.conv2d(F.pad(nt, (pad, pad, pad, pad), mode="reflect"),
                      kt).squeeze().numpy()
    smooth -= smooth.min()
    peak = float(smooth.max())
    smooth = smooth * (A_m / peak) if peak > 1e-30 else np.zeros_like(smooth)
    return torch.tensor(smooth, dtype=torch.float32, device=device)


def tv(h):
    dx_g = (h[1:, :] - h[:-1, :]).abs().mean()
    dy_g = (h[:, 1:] - h[:, :-1]).abs().mean()
    return dx_g + dy_g


def psnr_db(a, b):
    mse = float(torch.mean((a - b) ** 2))
    peak = float(b.abs().max())
    if mse < 1e-30: return 200.0
    if peak < 1e-30: return 0.0
    return 10.0 * np.log10(peak ** 2 / mse)


h_true = random_height(SEED, A_M, SIGMA_SMOOTH_UM)
with torch.no_grad():
    I_tgt = sensor(h_true)
    tv_true = float(tv(h_true))
print(f"GT: peak={float(h_true.max())*1e9:.1f} nm, "
      f"mean={float(h_true.mean())*1e9:.1f} nm, TV={tv_true:.3e}")

# lambda sweep — lambda=0 reproduces the no-TV baseline (~24 dB)
lambdas = [0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]

def run_with_tv(lam_tv):
    raw = torch.full((MEM, MEM), 0.5, device=device,
                     dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([raw], lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_ITER)
    for _ in range(N_ITER):
        opt.zero_grad()
        h = (raw * raw) * H_SCALE
        data = torch.mean((sensor(h) - I_tgt) ** 2)
        reg  = lam_tv * tv(h)
        (data + reg).backward()
        opt.step(); sched.step()
    with torch.no_grad():
        h_final = (raw * raw) * H_SCALE
        data = float(torch.mean((sensor(h_final) - I_tgt) ** 2))
        reg  = float(tv(h_final))
    return h_final, data, reg


print(f"\n{'lambda':>10} {'PSNR':>8} {'RMSE[nm]':>10} "
      f"{'data loss':>11} {'TV(h)':>10} {'t[s]':>6}")
results = []
keep = {}
for lam_tv in lambdas:
    t0 = time.time()
    h_rec, data_loss, tv_val = run_with_tv(lam_tv)
    p = psnr_db(h_rec, h_true)
    rmse = float(torch.sqrt(torch.mean((h_rec - h_true) ** 2))) * 1e9
    dt = time.time() - t0
    print(f"{lam_tv:>10.3e} {p:>8.2f} {rmse:>10.3f} "
          f"{data_loss:>11.3e} {tv_val:>10.3e} {dt:>6.0f}")
    results.append(dict(lam=lam_tv, psnr=p, rmse_nm=rmse,
                        data_loss=data_loss, tv=tv_val, t=dt))
    keep[lam_tv] = h_rec.cpu().numpy() * 1e9

# ---------------- plot ----------------
plt.rcParams.update(STYLE)
fig = plt.figure(figsize=(16, 8))
gs = fig.add_gridspec(2, 4)
fig.suptitle(f"TV regularizer sweep  (r2: A={A_NM} nm, σ={SIGMA_SMOOTH_UM} um)",
             fontsize=13, color="#e6edf3")
fig.patch.set_facecolor("#0d1117")

def _style(ax):
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e", labelsize=8)
    ax.grid(color="#30363d", lw=0.4, ls=":")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

# PSNR vs lambda
ax = fig.add_subplot(gs[0, :2])
lam_axis = np.array([r["lam"] if r["lam"] > 0 else 1e-6 for r in results])
ax.plot(lam_axis, [r["psnr"] for r in results], "o-",
        color="#58a6ff", lw=1.6)
ax.set_xscale("log")
ax.set_xlabel("λ_TV", fontsize=10, color="#8b949e")
ax.set_ylabel("PSNR [dB]", fontsize=10, color="#8b949e")
ax.set_title("(A) PSNR vs TV weight", fontsize=10, color="#e6edf3")
_style(ax)

ax = fig.add_subplot(gs[0, 2:])
ax.plot(lam_axis, [r["data_loss"] for r in results], "o-",
        color="#3fb950", lw=1.4, label="MSE(|U|²-I)")
ax.plot(lam_axis, [r["tv"] for r in results], "s-",
        color="#f78166", lw=1.4, label="TV(h)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("λ_TV", fontsize=10, color="#8b949e")
ax.legend(fontsize=8, labelcolor="#e6edf3", framealpha=0.3)
ax.set_title("(B) Data vs regularizer term", fontsize=10, color="#e6edf3")
_style(ax)

# bottom: h_true vs 4 reconstructions (lam=0, best, high-lam, over-smoothed)
extent_um = (MEM * dx / 2) * 1e6
h_true_np = h_true.cpu().numpy() * 1e9
vmin, vmax = 0.0, float(h_true_np.max())
panels = [("GT", h_true_np)]
# append 3 lambda values
for lam_tv in [lambdas[0], lambdas[len(lambdas)//2], lambdas[-1]]:
    panels.append((f"λ={lam_tv:.0e}", keep[lam_tv]))

for j, (name, h_nm) in enumerate(panels):
    ax = fig.add_subplot(gs[1, j])
    im = ax.imshow(h_nm, cmap="magma", vmin=vmin, vmax=vmax,
                   extent=[-extent_um, extent_um, -extent_um, extent_um])
    if name == "GT":
        title = "GT"
    else:
        # find matching result
        lam_val = float(name.split("=")[1])
        # handle scientific notation edge case
        r = min(results, key=lambda rr: abs(rr["lam"] - lam_val)
                if rr["lam"] > 0 else abs(1e-6 - lam_val))
        title = f"{name}\nPSNR={r['psnr']:.1f} dB"
    ax.set_title(title, fontsize=9, color="#e6edf3")
    _style(ax)

plt.tight_layout()
plt.savefig(OUT / "sweep.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()

lines = ["tv_regularizer_sweep.py — can TV prior break random-pattern ambiguity?",
         "=" * 66,
         f"  pattern  = seed=r2, A={A_NM} nm, sigma_smooth={SIGMA_SMOOTH_UM} um",
         f"  GT peak  = {float(h_true.max())*1e9:.1f} nm",
         f"  GT mean  = {float(h_true.mean())*1e9:.1f} nm",
         f"  GT TV    = {tv_true:.3e}",
         f"  algorithm= squared (h>=0) + Adam {N_ITER} iter, cosine decay, raw_init=0.5",
         "",
         f"  {'lambda':>10}  {'PSNR':>8}  {'RMSE[nm]':>10}  "
         f"{'data loss':>12}  {'TV(h)':>10}"]
for r in results:
    lines.append(f"  {r['lam']:>10.3e}  {r['psnr']:>8.2f}  {r['rmse_nm']:>10.3f}  "
                 f"{r['data_loss']:>12.3e}  {r['tv']:>10.3e}")
lines += ["",
          "Interpretation",
          "  * lambda=0 is the no-TV baseline (expected ~24 dB from",
          "    random_pattern_reconstruction).",
          "  * If any lambda pushes PSNR > 35 dB while data loss stays",
          "    comparable, TV resolves the ambiguity and we should",
          "    adopt that lambda globally.",
          "  * If even the best lambda stays at ~24 dB, the multi-basin",
          "    ambiguity is not fixable by a simple smoothness prior and",
          "    we need stronger constraints (e.g. multi-distance)."]
txt = "\n".join(lines)
(OUT / "sweep.txt").write_text(txt, encoding="utf-8")
print("\n" + txt)

(OUT / "sweep.json").write_text(json.dumps(dict(
    config=dict(A_nm=A_NM, sigma_smooth_um=SIGMA_SMOOTH_UM, seed=SEED,
                mem_pitch_um=dx*1e6, distance_mm=DIST*1e3,
                n_iter=N_ITER, lr=LR, lambdas=lambdas,
                gt_peak_nm=float(h_true.max())*1e9,
                gt_mean_nm=float(h_true.mean())*1e9,
                gt_tv=tv_true),
    results=results,
), indent=2))
print(f"\nSaved -> {OUT/'sweep.png'}")
