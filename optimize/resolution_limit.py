"""
resolution_limit.py
-------------------
Quantify lateral resolution of the holographic tactile sensor.

Theoretical bound (Abbe):
    delta_xy ~ lam / (2 NA)  with NA = sin(theta_max)
    sin(theta_max) = (L/2) / sqrt(d^2 + (L/2)^2),  L = grid_res * mem_pitch

Numerical experiment:
  Two Gaussian bumps separated by varying distance; reconstruct via Adam
  on the scaled parameterization, then test the Rayleigh criterion on the
  recovered cross-section:
    * find the two highest local maxima
    * the valley depth between them must be <= 0.81 * mean(peak heights)
      (Rayleigh-like, used in microscopy)

Bump shape: A=300 nm, sigma=40 um — small enough to probe lateral resolution
without running into phase-wrap regime, large enough that the recon is
well-conditioned.

Output:
  output/resolution_limit/
    resolution_summary.png
    resolution_metrics.txt
    resolution_metrics.json
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
from sensor_model  import HolographicSensor, gaussian_bump  # noqa
from sensor_params import wavelength, mem_pitch
from sensor_utils  import STYLE

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "output" / "resolution_limit"
OUT.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

lam = wavelength
dx  = mem_pitch

# -----------------------------------------------------------------------
# Theoretical
# -----------------------------------------------------------------------
def abbe_limit(d, L):
    sin_max = (L / 2.0) / np.sqrt(d**2 + (L / 2.0)**2)
    return lam / (2.0 * sin_max)

# -----------------------------------------------------------------------
# Numerical config
# -----------------------------------------------------------------------
MEM   = 128
CMOS  = 256
GRID  = 384
DIST  = 5e-3
AMP   = 300e-9
SIGMA = 40e-6
H_SCALE = 400e-9
LR    = 3e-3                # close to best from lr_sweep_big
N_ITER = 1500

sensor = HolographicSensor(
    wavelength=lam, mem_res=MEM, mem_pitch=dx,
    cmos_res=CMOS, grid_res=GRID, distance=DIST, device=device,
).to(device)

def two_gaussian(sep_m):
    coords = (torch.arange(MEM, dtype=torch.float32, device=device) - (MEM-1)*0.5) * dx
    X, Y = torch.meshgrid(coords, coords, indexing="ij")
    c = sep_m / 2.0
    g1 = torch.exp(-((X-c)**2 + Y**2) / (2*SIGMA**2))
    g2 = torch.exp(-((X+c)**2 + Y**2) / (2*SIGMA**2))
    return AMP * (g1 + g2)

def reconstruct(h_true, n_iter=N_ITER, lr=LR):
    with torch.no_grad():
        I_tgt = sensor(h_true)
    h_var = torch.zeros(MEM, MEM, dtype=torch.float32, device=device,
                        requires_grad=True)
    opt = torch.optim.Adam([h_var], lr=lr)
    for _ in range(n_iter):
        opt.zero_grad()
        I = sensor(h_var * H_SCALE)
        loss = torch.mean((I - I_tgt)**2)
        loss.backward()
        opt.step()
    return (h_var.detach() * H_SCALE)

# -----------------------------------------------------------------------
# Robust Rayleigh check
# -----------------------------------------------------------------------
def rayleigh_check(line_nm, sep_m):
    """
    line_nm   : 1D recovered cross-section (nm), length N
    sep_m     : separation between bump centres (m)
    Returns: (resolved, peak_left, peak_right, valley, dip_ratio)
    """
    N = len(line_nm)
    mid = N // 2
    sep_pix_half = max(int(round((sep_m/2) / dx)), 1)
    # Search windows for left and right peaks
    L_lo = max(mid - 2*sep_pix_half, 0)
    L_hi = mid - max(sep_pix_half//2, 1)
    R_lo = mid + max(sep_pix_half//2, 1)
    R_hi = min(mid + 2*sep_pix_half, N - 1)
    if L_hi <= L_lo or R_hi <= R_lo:
        return False, np.nan, np.nan, np.nan, np.nan
    L_peak_idx = L_lo + int(np.argmax(line_nm[L_lo:L_hi]))
    R_peak_idx = R_lo + int(np.argmax(line_nm[R_lo:R_hi]))
    L_peak = line_nm[L_peak_idx]
    R_peak = line_nm[R_peak_idx]
    # Valley between the two peaks
    if R_peak_idx <= L_peak_idx:
        return False, L_peak, R_peak, np.nan, np.nan
    valley = line_nm[L_peak_idx:R_peak_idx+1].min()
    mean_peak = 0.5 * (L_peak + R_peak)
    if mean_peak <= 0:
        return False, L_peak, R_peak, valley, np.nan
    dip_ratio = (mean_peak - valley) / mean_peak  # 0 (merged) -> 1 (deep dip)
    resolved = (dip_ratio >= 0.19) and (L_peak > 0) and (R_peak > 0)
    return bool(resolved), float(L_peak), float(R_peak), float(valley), float(dip_ratio)

# -----------------------------------------------------------------------
# Sweep
# -----------------------------------------------------------------------
separations_um = [60, 80, 100, 120, 150, 200, 300, 500, 800]
seps_m = [s * 1e-6 for s in separations_um]

results = []
print(f"\nAbbe (d={DIST*1e3}mm, L={GRID*dx*1e3:.2f}mm) = "
      f"{abbe_limit(DIST, GRID*dx)*1e6:.2f} um")
print(f"Membrane Nyquist                              = {2*dx*1e6:.0f} um\n")
print(f"{'sep [um]':>8}  {'L_peak':>8}  {'R_peak':>8}  {'valley':>8}  "
      f"{'dip%':>6}  {'res':>4}  {'time[s]':>7}")
for s in seps_m:
    t0 = time.time()
    h_true = two_gaussian(s)
    h_rec = reconstruct(h_true)
    h_np  = h_rec.cpu().numpy() * 1e9
    line  = h_np[MEM//2, :]
    ok, L, R, V, dip = rayleigh_check(line, s)
    dt = time.time() - t0
    print(f"{s*1e6:>8.0f}  {L:>8.1f}  {R:>8.1f}  {V:>8.1f}  "
          f"{dip*100 if np.isfinite(dip) else float('nan'):>6.1f}  "
          f"{str(ok):>4}  {dt:>7.1f}")
    results.append(dict(sep_um=s*1e6, L_peak=L, R_peak=R, valley=V,
                        dip_ratio=dip, resolved=ok, h_rec=h_np))

resolved_ums = sorted([r["sep_um"] for r in results if r["resolved"]])
numerical_res = resolved_ums[0] if resolved_ums else float("nan")

# -----------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------
plt.rcParams.update(STYLE)
fig, axes = plt.subplots(2, 3, figsize=(20, 11))
fig.suptitle("Resolution limit — Rayleigh-criterion two-bump test",
             fontsize=13, color="#e6edf3")
fig.patch.set_facecolor("#0d1117")

def _style(ax):
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e", labelsize=7)
    ax.grid(color="#30363d", lw=0.4, ls=":")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

# (A) Theoretical Abbe vs distance for several aperture sizes
ax = axes[0,0]
ds_mm = np.linspace(0.5, 30, 300)
for L_mm in [3.84, 5.12, 7.68, 15.36]:
    ax.plot(ds_mm, abbe_limit(ds_mm*1e-3, L_mm*1e-3)*1e6, lw=1.2,
            label=f"L = {L_mm} mm")
ax.axhline(2*dx*1e6, color="#f85149", ls="--", lw=0.8,
           label=f"Nyquist 2 dx = {2*dx*1e6:.0f} um")
ax.axvline(DIST*1e3, color="#8b949e", ls=":", lw=0.6)
ax.set_xlabel("distance d [mm]", fontsize=9, color="#8b949e")
ax.set_ylabel("lateral resolution [um]", fontsize=9, color="#8b949e")
ax.set_title("(A) Abbe limit  lam/(2 NA)", fontsize=10, color="#e6edf3")
ax.legend(fontsize=7, labelcolor="#e6edf3", framealpha=0.3)
ax.set_yscale("log")
_style(ax)

# (B) dip ratio vs separation
ax = axes[0,1]
seps = [r["sep_um"] for r in results]
dips = [r["dip_ratio"]*100 if np.isfinite(r["dip_ratio"]) else 0
        for r in results]
colors = ["#3fb950" if r["resolved"] else "#f85149" for r in results]
ax.bar(range(len(seps)), dips, color=colors, edgecolor="#30363d")
ax.axhline(19.0, color="#e3b341", ls="--", lw=0.8, label="Rayleigh thresh 19%")
ax.set_xticks(range(len(seps)))
ax.set_xticklabels([f"{int(s)}" for s in seps], fontsize=7, color="#8b949e")
ax.set_xlabel("separation [um]", fontsize=9, color="#8b949e")
ax.set_ylabel("(peak - valley) / peak  [%]", fontsize=9, color="#8b949e")
ax.set_title("(B) Dip ratio per separation (green=resolved)",
             fontsize=10, color="#e6edf3")
ax.legend(fontsize=7, labelcolor="#e6edf3", framealpha=0.3)
_style(ax)

# (C) cross-sections, color-coded
ax = axes[0,2]
m_um = (np.arange(MEM) - (MEM-1)*0.5) * dx * 1e6
cmap = plt.get_cmap("viridis")
for i, r in enumerate(results):
    c = cmap(i / max(len(results)-1, 1))
    mid = MEM // 2
    ax.plot(m_um, r["h_rec"][mid, :], color=c, lw=1.0,
            label=f"{r['sep_um']:.0f} um {'OK' if r['resolved'] else 'merged'}")
ax.set_xlim(-600, 600)
ax.set_xlabel("x [um]", fontsize=9, color="#8b949e")
ax.set_ylabel("h [nm]", fontsize=9, color="#8b949e")
ax.set_title("(C) Recovered cross-sections", fontsize=10, color="#e6edf3")
ax.legend(fontsize=6, labelcolor="#e6edf3", framealpha=0.3, ncol=2)
_style(ax)

# (D,E,F) Three representative recoveries: one merged, one borderline, one clear
def pick(target_state):
    for r in results:
        if r["resolved"] == target_state:
            return r
    return results[0]
selected = []
# pick one clearly merged (smallest sep), borderline (smallest resolved), clear (largest)
selected.append(min(results, key=lambda r: r["sep_um"]))
if resolved_ums:
    selected.append(next(r for r in results if r["sep_um"] == resolved_ums[0]))
else:
    selected.append(results[len(results)//2])
selected.append(max(results, key=lambda r: r["sep_um"]))

for idx, r in enumerate(selected):
    ax = axes[1, idx]
    im = ax.imshow(r["h_rec"].T, cmap="RdBu",
                   extent=[m_um.min(), m_um.max(), m_um.min(), m_um.max()],
                   origin="lower", aspect="equal",
                   vmin=-AMP*1e9, vmax=AMP*1e9)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("h [nm]", color="#8b949e", fontsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="#8b949e")
    label = "RESOLVED" if r["resolved"] else "merged"
    ax.set_title(f"sep={r['sep_um']:.0f} um  dip={r['dip_ratio']*100:.0f}%  ({label})",
                 fontsize=9, color="#e6edf3")
    _style(ax)

plt.tight_layout()
plt.savefig(OUT / "resolution_summary.png", dpi=150, bbox_inches="tight",
            facecolor="#0d1117")
plt.close()

# -----------------------------------------------------------------------
# Quantitative
# -----------------------------------------------------------------------
txt_lines = [
    "resolution_limit.py — Lateral resolution",
    "========================================",
    f"Configuration",
    f"  wavelength       = {lam*1e9:.1f} nm",
    f"  distance         = {DIST*1e3:.1f} mm",
    f"  grid physical L  = {GRID*dx*1e3:.2f} mm",
    f"  bump amplitude   = {AMP*1e9:.0f} nm   sigma = {SIGMA*1e6:.0f} um",
    f"  Abbe theoretical = {abbe_limit(DIST, GRID*dx)*1e6:.2f} um",
    f"  Membrane Nyquist = {2*dx*1e6:.1f} um",
    "",
    "Two-bump separation sweep, Rayleigh dip threshold 19%",
    f"  {'sep [um]':>8}  {'L_pk[nm]':>9}  {'R_pk[nm]':>9}  "
    f"{'val[nm]':>9}  {'dip%':>6}  resolved",
]
for r in results:
    txt_lines.append(
        f"  {r['sep_um']:>8.0f}  {r['L_peak']:>9.1f}  {r['R_peak']:>9.1f}  "
        f"{r['valley']:>9.1f}  "
        f"{(r['dip_ratio']*100 if np.isfinite(r['dip_ratio']) else float('nan')):>6.1f}  "
        f"{r['resolved']}"
    )
txt_lines.append("")
txt_lines.append(f"Smallest numerically resolved : {numerical_res} um")
txt_lines.append(f"Bound (Nyquist)              : {2*dx*1e6:.1f} um")
txt_lines.append(f"Bound (Abbe)                 : {abbe_limit(DIST, GRID*dx)*1e6:.2f} um")
txt = "\n".join(txt_lines)
(OUT / "resolution_metrics.txt").write_text(txt, encoding="utf-8")
print("\n" + txt)
def _f(x):
    try:
        return float(x)
    except Exception:
        return None
(OUT / "resolution_metrics.json").write_text(json.dumps(dict(
    abbe_um=float(abbe_limit(DIST, GRID*dx)*1e6),
    nyquist_um=float(2*dx*1e6),
    numerical_resolved_um=_f(numerical_res),
    sweep=[{k: (bool(r[k]) if k == "resolved" else _f(r[k]))
            for k in ("sep_um", "L_peak", "R_peak", "valley",
                       "dip_ratio", "resolved")} for r in results],
), indent=2))

print(f"\nSaved -> {OUT/'resolution_summary.png'}")
