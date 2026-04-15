"""
large_deformation_sweep.py
--------------------------
Sweep deformation amplitude across many decades (1 nm -> 10 um) and assess:
  - forward intensity behaviour (linear / phase-wrapped / saturated)
  - reconstruction accuracy via MSE minimization
  - effective useful range (PSNR >= 20 dB)

The meeting emphasised the need for LARGE deformation scales: single-
amplitude sanity results can miss the phase-wrap transition that dominates
real-world tactile data.

Output:
  output/large_deformation_sweep/
    amplitude_sweep.png
    amplitude_sweep.txt
    amplitude_sweep.json
"""

import sys, time, json, functools
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print = functools.partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).parent.parent))
from sensor_model  import HolographicSensor, gaussian_bump
from sensor_params import wavelength, mem_pitch
from sensor_utils  import STYLE

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "output" / "large_deformation_sweep"
OUT.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

MEM   = 128
CMOS  = 256
GRID  = 384
DIST  = 5e-3
SIGMA = 150e-6

sensor = HolographicSensor(
    wavelength=wavelength, mem_res=MEM, mem_pitch=mem_pitch,
    cmos_res=CMOS, grid_res=GRID, distance=DIST, device=device,
).to(device)

amps_nm = np.array([1, 3, 10, 30, 79, 158, 300, 500, 1000, 2000, 5000, 10000])
amps_m  = amps_nm * 1e-9

def rmse_nm(a, b): return float(torch.sqrt(torch.mean((a - b)**2))) * 1e9
def psnr_db(a, b):
    mse_v = float(torch.mean((a - b)**2))
    peak  = float(b.abs().max())
    if mse_v < 1e-30: return 100.0
    if peak < 1e-30:  return 0.0
    return 10.0 * np.log10(peak**2 / mse_v)

def reconstruct(h_true, lr=1e-3, n_iter=1500, h_scale=None):
    if h_scale is None:
        h_scale = max(float(h_true.abs().max()) * 2.0, 100e-9)
    with torch.no_grad():
        I_tgt = sensor(h_true)
    h_var = torch.zeros(MEM, MEM, dtype=torch.float32, device=device,
                        requires_grad=True)
    opt = torch.optim.Adam([h_var], lr=lr)
    losses = []
    for _ in range(n_iter):
        opt.zero_grad()
        I = sensor(h_var * h_scale)
        loss = torch.mean((I - I_tgt)**2)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return (h_var.detach() * h_scale), losses, h_scale

results = []
print(f"\n{'A [nm]':>8}  {'scale':>10}  {'PSNR':>8}  {'RMSE[nm]':>10}  {'loss':>10}  {'I rms':>8}")
for A in amps_m:
    h_true = gaussian_bump(N=MEM, dx=mem_pitch, amplitude=float(A),
                           sigma=SIGMA, device=device)
    t0 = time.time()
    h_rec, losses, hs = reconstruct(h_true)
    dt = time.time() - t0
    p = psnr_db(h_rec, h_true)
    r = rmse_nm(h_rec, h_true)
    with torch.no_grad():
        I_tgt = sensor(h_true)
    I_rms = float((I_tgt - 1.0).std())
    print(f"{A*1e9:>8.1f}  {hs*1e9:>8.0f}nm  {p:>8.2f}  {r:>10.3f}  "
          f"{losses[-1]:>10.2e}  {I_rms:>8.3f}  ({dt:.1f}s)")
    results.append(dict(A_nm=float(A*1e9), h_scale_nm=float(hs*1e9),
                        psnr=p, rmse=r, final_loss=float(losses[-1]),
                        I_rms=float(I_rms), h_true=h_true.cpu().numpy(),
                        h_rec=h_rec.cpu().numpy(), losses=losses,
                        I_tgt=I_tgt.cpu().numpy()))

# -----------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------
plt.rcParams.update(STYLE)
fig, axes = plt.subplots(2, 3, figsize=(20, 11))
fig.suptitle("Large-deformation sweep  1 nm -> 10 um",
             fontsize=13, color="#e6edf3")
fig.patch.set_facecolor("#0d1117")

def _style(ax):
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e", labelsize=7)
    ax.grid(color="#30363d", lw=0.4, ls=":")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

A_arr   = np.array([r["A_nm"] for r in results])
psnrs   = np.array([r["psnr"] for r in results])
rmses   = np.array([r["rmse"] for r in results])
I_rmss  = np.array([r["I_rms"] for r in results])

# (a) PSNR
ax = axes[0,0]
ax.semilogx(A_arr, psnrs, "o-", color="#58a6ff", lw=1.5, markersize=6)
ax.axvline(wavelength*0.25*1e9, color="#f85149", ls="--", lw=0.8,
           label=f"lam/4 = {wavelength*0.25*1e9:.0f} nm")
ax.axhline(20.0, color="#3fb950", ls=":", lw=0.8, label="20 dB")
ax.set_xlabel("amplitude A [nm]", fontsize=9, color="#8b949e")
ax.set_ylabel("PSNR [dB]", fontsize=9, color="#8b949e")
ax.set_title("Reconstruction PSNR vs amplitude", fontsize=10, color="#e6edf3")
ax.legend(fontsize=7, labelcolor="#e6edf3", framealpha=0.3)
_style(ax)

# (b) RMSE (relative)
ax = axes[0,1]
ax.loglog(A_arr, rmses, "o-", color="#f78166", lw=1.5, markersize=6)
ax.loglog(A_arr, A_arr, color="#8b949e", ls=":", lw=0.8,
          label="100% (no recon)")
ax.loglog(A_arr, 0.1*A_arr, color="#3fb950", ls=":", lw=0.8,
          label="10%")
ax.set_xlabel("amplitude A [nm]", fontsize=9, color="#8b949e")
ax.set_ylabel("RMSE [nm]", fontsize=9, color="#8b949e")
ax.set_title("Absolute RMSE vs amplitude", fontsize=10, color="#e6edf3")
ax.legend(fontsize=7, labelcolor="#e6edf3", framealpha=0.3)
_style(ax)

# (c) Forward-model intensity fluctuation vs A
ax = axes[0,2]
ax.loglog(A_arr, I_rmss, "o-", color="#e3b341", lw=1.5, markersize=6)
ax.set_xlabel("amplitude A [nm]", fontsize=9, color="#8b949e")
ax.set_ylabel("std(I - 1)", fontsize=9, color="#8b949e")
ax.set_title("Forward intensity RMS vs amplitude", fontsize=10, color="#e6edf3")
_style(ax)

# (d,e,f) Representative reconstructions
m_um = (np.arange(MEM) - (MEM-1)*0.5) * mem_pitch * 1e6
picks = [0, len(results)//2, len(results)-1]
for i, pk in enumerate(picks):
    ax = axes[1, i]
    r = results[pk]
    mid = MEM // 2
    ax.plot(m_um, r["h_true"][mid, :]*1e9, color="#58a6ff", lw=1.5, label="h_true")
    ax.plot(m_um, r["h_rec"][mid, :]*1e9, color="#f78166", lw=1.2, ls="--",
            label="h_rec")
    ax.set_xlabel("x [um]", fontsize=9, color="#8b949e")
    ax.set_ylabel("h [nm]", fontsize=9, color="#8b949e")
    ax.set_title(f"A={r['A_nm']:.0f} nm, PSNR={r['psnr']:.1f} dB",
                 fontsize=10, color="#e6edf3")
    ax.legend(fontsize=7, labelcolor="#e6edf3", framealpha=0.3)
    _style(ax)

plt.tight_layout()
plt.savefig(OUT / "amplitude_sweep.png", dpi=150, bbox_inches="tight",
            facecolor="#0d1117")
plt.close()

# Text summary
lines = [
    "large_deformation_sweep.py — Amplitude sweep results",
    "====================================================",
    f"Gaussian bump sigma = {SIGMA*1e6:.0f} um   distance = {DIST*1e3:.0f} mm",
    f"Reconstruction      : Adam, lr=1e-3, 1500 iters, scaled param",
    f"Useful (PSNR>=20dB) range: "
    f"{min((r['A_nm'] for r in results if r['psnr']>=20), default='n/a')} - "
    f"{max((r['A_nm'] for r in results if r['psnr']>=20), default='n/a')} nm",
    "",
    f"{'A [nm]':>8}  {'PSNR':>8}  {'RMSE[nm]':>10}  {'final loss':>12}",
]
for r in results:
    lines.append(f"{r['A_nm']:>8.1f}  {r['psnr']:>8.2f}  "
                 f"{r['rmse']:>10.3f}  {r['final_loss']:>12.3e}")
txt = "\n".join(lines)
(OUT / "amplitude_sweep.txt").write_text(txt, encoding="utf-8")
print("\n" + txt)
(OUT / "amplitude_sweep.json").write_text(json.dumps(
    [{k: r[k] for k in ("A_nm", "h_scale_nm", "psnr", "rmse",
                         "final_loss", "I_rms")} for r in results],
    indent=2))
print(f"\nSaved -> {OUT/'amplitude_sweep.png'}")
