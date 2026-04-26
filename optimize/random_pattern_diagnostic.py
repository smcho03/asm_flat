"""
random_pattern_diagnostic.py
----------------------------
Why does the corrected algorithm (squared + Adam-15k) hit only ~13 dB
on smooth random patterns when it reached ~50 dB on a single Gaussian
with identical per-pixel slope?

Hypotheses tested on a single reference case (seed=1, A=100 nm,
sigma_smooth=150 um):

  (a) baseline          — raw=0.2 uniform, Adam 15k        (current default)
  (b) longer            — raw=0.2 uniform, Adam 50k        (rule out under-conv)
  (c) mean-init         — raw=sqrt(mean(h_true)/H)         (better DC start)
  (d) multi-start       — 5 random inits, keep best        (multi-modal landscape?)
  (e) GT-warm           — raw=sqrt(h_true/H) + fine-tune   (upper bound)
  (f) lbfgs polish      — (a) warm-start -> LBFGS          (last-mile refinement)

If (e) reaches >>30 dB and everything else plateaus, algorithm has a
local-minima problem (need better init / global search). If (e) also
plateaus, the inverse problem is genuinely ill-posed for random h.

Output:
  output/random_pattern_diagnostic/
    diag.png, diag.txt, diag.json
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
OUT  = ROOT / "output" / "random_pattern_diagnostic"
OUT.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

MEM, CMOS, GRID, DIST = 128, 256, 384, 5e-3
A_NM, SIGMA_SMOOTH_UM, SEED = 100, 150, 1
A_M = A_NM * 1e-9
H_SCALE = 2.0 * A_M
LR = 3e-3

lam = wavelength
dx  = mem_pitch
PHASE_WRAP_NM = lam / 4.0 * 1e9

sensor = HolographicSensor(
    wavelength=lam, mem_res=MEM, mem_pitch=dx,
    cmos_res=CMOS, grid_res=GRID, distance=DIST, device=device,
).to(device)


def make_random_height(seed, A_m, sigma_smooth_um):
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
    smooth = smooth * (A_m / peak) if peak > 1e-30 else np.zeros_like(smooth)
    return torch.tensor(smooth, dtype=torch.float32, device=device)


h_true = make_random_height(SEED, A_M, SIGMA_SMOOTH_UM)
with torch.no_grad():
    I_tgt = sensor(h_true)


def psnr_db(a, b):
    mse = float(torch.mean((a - b) ** 2))
    peak = float(b.abs().max())
    if mse < 1e-30: return 200.0
    if peak < 1e-30: return 0.0
    return 10.0 * np.log10(peak ** 2 / mse)


def adam_run(raw_init, n_iter):
    raw = raw_init.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([raw], lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iter)
    for _ in range(n_iter):
        opt.zero_grad()
        h = (raw * raw) * H_SCALE
        loss = torch.mean((sensor(h) - I_tgt) ** 2)
        loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        h_final = (raw * raw) * H_SCALE
        l = float(torch.mean((sensor(h_final) - I_tgt) ** 2))
    return raw.detach(), h_final, l


def lbfgs_polish(raw_init, max_iter=400):
    raw = raw_init.clone().detach().requires_grad_(True)
    opt = torch.optim.LBFGS([raw], lr=1.0, max_iter=max_iter,
                            tolerance_grad=1e-14, tolerance_change=1e-14,
                            line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        h = (raw * raw) * H_SCALE
        loss = torch.mean((sensor(h) - I_tgt) ** 2)
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        h_final = (raw * raw) * H_SCALE
        l = float(torch.mean((sensor(h_final) - I_tgt) ** 2))
    return h_final, l


results = []
keep = {}

# (a) baseline
t0 = time.time()
raw0 = torch.full((MEM, MEM), 0.2, device=device, dtype=torch.float32)
raw_a, h_a, la = adam_run(raw0, 15000)
pa = psnr_db(h_a, h_true); ra = float(torch.sqrt(torch.mean((h_a-h_true)**2)))*1e9
print(f"(a) baseline        PSNR={pa:.2f} dB  RMSE={ra:.2f} nm  ({time.time()-t0:.0f}s)")
results.append(dict(name="baseline", psnr=pa, rmse_nm=ra, loss=la, t=time.time()-t0))
keep["baseline"] = h_a.cpu().numpy()*1e9

# (b) longer
t0 = time.time()
raw_b, h_b, lb = adam_run(raw0, 50000)
pb = psnr_db(h_b, h_true); rb = float(torch.sqrt(torch.mean((h_b-h_true)**2)))*1e9
print(f"(b) longer 50k      PSNR={pb:.2f} dB  RMSE={rb:.2f} nm  ({time.time()-t0:.0f}s)")
results.append(dict(name="longer50k", psnr=pb, rmse_nm=rb, loss=lb, t=time.time()-t0))
keep["longer50k"] = h_b.cpu().numpy()*1e9

# (c) mean-init (start near mean of ground truth)
t0 = time.time()
mean_h = float(h_true.mean())
raw_init_val = float(np.sqrt(max(mean_h, 1e-30) / H_SCALE))
raw_c_init = torch.full((MEM, MEM), raw_init_val, device=device, dtype=torch.float32)
raw_c, h_c, lc = adam_run(raw_c_init, 15000)
pc = psnr_db(h_c, h_true); rc = float(torch.sqrt(torch.mean((h_c-h_true)**2)))*1e9
print(f"(c) mean-init       PSNR={pc:.2f} dB  RMSE={rc:.2f} nm  ({time.time()-t0:.0f}s)"
      f"   [raw_init={raw_init_val:.3f}, mean_h={mean_h*1e9:.1f} nm]")
results.append(dict(name="mean_init", psnr=pc, rmse_nm=rc, loss=lc, t=time.time()-t0,
                    raw_init=raw_init_val, mean_h_nm=mean_h*1e9))
keep["mean_init"] = h_c.cpu().numpy()*1e9

# (d) multi-start
t0 = time.time()
best = None
for ms_seed in range(5):
    rng = torch.Generator(device=device).manual_seed(1000 + ms_seed)
    raw_init_d = 0.2 + 0.1 * torch.randn(MEM, MEM, device=device, generator=rng)
    _, h_d, ld = adam_run(raw_init_d, 15000)
    pd_ = psnr_db(h_d, h_true)
    if best is None or pd_ > best[0]:
        best = (pd_, h_d, ld, ms_seed)
pd_best, h_d_best, ld, ms_best_seed = best
rd = float(torch.sqrt(torch.mean((h_d_best-h_true)**2)))*1e9
print(f"(d) multi-start x5  best PSNR={pd_best:.2f} dB  RMSE={rd:.2f} nm  "
      f"(seed {ms_best_seed})  ({time.time()-t0:.0f}s)")
results.append(dict(name="multi_start_x5", psnr=pd_best, rmse_nm=rd, loss=ld,
                    t=time.time()-t0, best_seed=ms_best_seed))
keep["multi_start"] = h_d_best.cpu().numpy()*1e9

# (e) GT-warm start
t0 = time.time()
raw_e_init = torch.sqrt(torch.clamp(h_true / H_SCALE, min=1e-20))
raw_e, h_e, le = adam_run(raw_e_init, 15000)
pe = psnr_db(h_e, h_true); re = float(torch.sqrt(torch.mean((h_e-h_true)**2)))*1e9
print(f"(e) GT-warm         PSNR={pe:.2f} dB  RMSE={re:.2f} nm  ({time.time()-t0:.0f}s)")
results.append(dict(name="gt_warm", psnr=pe, rmse_nm=re, loss=le, t=time.time()-t0))
keep["gt_warm"] = h_e.cpu().numpy()*1e9

# (f) lbfgs polish on baseline
t0 = time.time()
h_f, lf = lbfgs_polish(raw_a)
pf = psnr_db(h_f, h_true); rf = float(torch.sqrt(torch.mean((h_f-h_true)**2)))*1e9
print(f"(f) baseline+LBFGS  PSNR={pf:.2f} dB  RMSE={rf:.2f} nm  ({time.time()-t0:.0f}s)")
results.append(dict(name="baseline_lbfgs", psnr=pf, rmse_nm=rf, loss=lf, t=time.time()-t0))
keep["lbfgs"] = h_f.cpu().numpy()*1e9

# ---------------- plot ----------------
plt.rcParams.update(STYLE)
h_true_np = h_true.cpu().numpy() * 1e9
panels = [("GT", h_true_np), ("baseline", keep["baseline"]),
          ("longer50k", keep["longer50k"]), ("mean_init", keep["mean_init"]),
          ("multi_start", keep["multi_start"]), ("gt_warm", keep["gt_warm"]),
          ("lbfgs", keep["lbfgs"])]
fig = plt.figure(figsize=(4*len(panels), 8))
gs = fig.add_gridspec(2, len(panels))
fig.patch.set_facecolor("#0d1117")
fig.suptitle(f"Random pattern diagnostic  (A={A_NM} nm, σ={SIGMA_SMOOTH_UM} um, seed={SEED})",
             fontsize=13, color="#e6edf3")

def _style(ax):
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e", labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

vmin, vmax = 0.0, float(h_true_np.max())
extent_um = (MEM * dx / 2) * 1e6
for j, (name, h_nm) in enumerate(panels):
    ax = fig.add_subplot(gs[0, j])
    im = ax.imshow(h_nm, cmap="magma", vmin=vmin, vmax=vmax,
                   extent=[-extent_um, extent_um, -extent_um, extent_um])
    if name == "GT":
        title = "GT"
    else:
        r = next(rr for rr in results if rr["name"].startswith(name.split("_")[0]) or rr["name"]==name)
        # match by keep-key name
        mapping = {"baseline":"baseline","longer50k":"longer50k","mean_init":"mean_init",
                   "multi_start":"multi_start_x5","gt_warm":"gt_warm","lbfgs":"baseline_lbfgs"}
        r = next(rr for rr in results if rr["name"]==mapping[name])
        title = f"{name}\nPSNR={r['psnr']:.1f} dB"
    ax.set_title(title, fontsize=9, color="#e6edf3")
    _style(ax)

    ax = fig.add_subplot(gs[1, j])
    if name == "GT":
        ax.axis("off")
    else:
        err = h_nm - h_true_np
        emax = float(max(abs(err.min()), abs(err.max())))
        im = ax.imshow(err, cmap="RdBu_r", vmin=-emax, vmax=emax,
                       extent=[-extent_um, extent_um, -extent_um, extent_um])
        ax.set_title(f"err max={emax:.1f} nm", fontsize=9, color="#e6edf3")
        _style(ax)

plt.tight_layout()
plt.savefig(OUT / "diag.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()

lines = ["random_pattern_diagnostic.py — why random fails",
         "=" * 60,
         f"  case: A={A_NM} nm, sigma_smooth={SIGMA_SMOOTH_UM} um, seed={SEED}",
         f"  mean h = {float(h_true.mean())*1e9:.1f} nm, peak h = {float(h_true.max())*1e9:.1f} nm",
         f"  phase-wrap pixel limit = {PHASE_WRAP_NM:.2f} nm",
         "",
         f"  {'strategy':>18}  {'PSNR':>8}  {'RMSE[nm]':>10}  {'loss':>10}  {'t[s]':>6}"]
for r in results:
    lines.append(f"  {r['name']:>18s}  {r['psnr']:>8.2f}  {r['rmse_nm']:>10.3f}  "
                 f"{r['loss']:>10.3e}  {r['t']:>6.0f}")
lines += ["",
          "Interpretation",
          "  * If (e) GT-warm achieves >>30 dB while all others plateau near",
          "    10-15 dB, the algorithm has a bad local-minima problem and",
          "    needs a better initial guess or global search.",
          "  * If (e) also plateaus, the inverse problem is genuinely ill-posed",
          "    for generic random patterns with dx=10 um — only strong priors",
          "    or multiple measurements can help."]
txt = "\n".join(lines)
(OUT / "diag.txt").write_text(txt, encoding="utf-8")
print("\n" + txt)

(OUT / "diag.json").write_text(json.dumps(dict(
    config=dict(A_nm=A_NM, sigma_smooth_um=SIGMA_SMOOTH_UM, seed=SEED,
                mem_pitch_um=dx*1e6, distance_mm=DIST*1e3,
                n_iter_default=15000, lr=LR,
                mean_h_nm=float(h_true.mean())*1e9,
                peak_h_nm=float(h_true.max())*1e9),
    results=results,
), indent=2))
print(f"\nSaved -> {OUT/'diag.png'}")
