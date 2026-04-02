import os
import numpy as np
import torch, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.restoration import unwrap_phase
sys.path.insert(0, str(Path(__file__).parent))
from sensor_model import HolographicSensor, make_h_random
from sensor_params import wavelength, mem_res, mem_pitch, cmos_res, grid_res, distance
from sensor_utils import STYLE

os.makedirs("output", exist_ok=True)

device = "cpu"
sensor = HolographicSensor(
    wavelength=wavelength, mem_res=mem_res, mem_pitch=mem_pitch,
    cmos_res=cmos_res, grid_res=grid_res, distance=distance, device=device,
).to(device)

lam = wavelength

h = make_h_random(N=mem_res, dx=mem_pitch, device=device)

# float64로 직접 계산 (sensor 모델 우회)
def propagate_f64(sensor, h_f32):
    """float64로 전파 계산. sensor의 H를 float64로 캐스팅해서 사용."""
    import torch.nn.functional as F_
    ms, me = sensor._ms, sensor._me
    gr = sensor.grid_res
    pad = (ms, gr - me, ms, gr - me)

    h64 = h_f32.double()
    A64 = torch.ones_like(h64)
    h_grid = F_.pad(h64, pad)
    A_grid = F_.pad(A64, pad)

    phi = (4.0 * np.pi / sensor.lam) * h_grid
    U0  = A_grid * torch.exp(1j * phi)
    H64 = sensor.H.cdouble()
    Ud  = torch.fft.ifft2(torch.fft.fft2(U0) * H64)
    return Ud

# lambda/2 덧셈을 float64에서 수행 (float32로 다시 변환하지 않음)
h_shifted = h.double() + float(lam / 2)

with torch.no_grad():
    Ud_orig  = propagate_f64(sensor, h)
    Ud_shift = propagate_f64(sensor, h_shifted)
    cs, ce   = sensor._s, sensor._e
    I_orig   = Ud_orig[cs:ce, cs:ce].abs().pow(2)
    I_shift  = Ud_shift[cs:ce, cs:ce].abs().pow(2)
    Uc_orig  = Ud_orig[cs:ce, cs:ce]
    Uc_shift = Ud_shift[cs:ce, cs:ce]

diff_abs = (Ud_orig - Ud_shift).abs()
I_diff   = (I_orig - I_shift).abs()

print(f"complex field max diff  : {diff_abs.max().item():.2e}")
print(f"complex field mean diff : {diff_abs.mean().item():.2e}")
print(f"intensity max diff      : {I_diff.max().item():.2e}")
print(f"intensity mean diff     : {I_diff.mean().item():.2e}")

# --- numpy 변환 ---
mem_coords  = (np.arange(mem_res)  - (mem_res  - 1) * 0.5) * mem_pitch
cmos_coords = (np.arange(cmos_res) - (cmos_res - 1) * 0.5) * mem_pitch
m_um = mem_coords  * 1e6
c_um = cmos_coords * 1e6
ext_m = [m_um.min(), m_um.max(), m_um.min(), m_um.max()]
ext_c = [c_um.min(), c_um.max(), c_um.min(), c_um.max()]

h_nm       = h.cpu().float().numpy() * 1e9
hs_nm      = h_shifted.cpu().float().numpy() * 1e9
I_o_np     = I_orig.cpu().numpy()
I_s_np     = I_shift.cpu().numpy()
I_diff_np  = I_diff.cpu().numpy()
phi_o      = unwrap_phase(np.angle(Uc_orig.cpu().numpy()).astype(np.float64)).astype(np.float32)
phi_s      = unwrap_phase(np.angle(Uc_shift.cpu().numpy()).astype(np.float64)).astype(np.float32)
phi_diff   = np.angle(Uc_orig.cpu().numpy() * np.conj(Uc_shift.cpu().numpy())).astype(np.float32)

# --- 그림 ---
plt.rcParams.update(STYLE)
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
fig.suptitle(
    f"Phase periodicity check:  h  vs  h + lambda/2\n"
    f"(lambda/2 = {lam/2*1e9:.1f} nm,  field diff should be ~0)",
    fontsize=11, color="#e6edf3",
)
fig.patch.set_facecolor("#0d1117")

def _imshow(ax, data, ext, cmap, vmin=None, vmax=None, label="", title=""):
    im = ax.imshow(data.T, extent=ext, origin="lower", aspect="auto",
                   cmap=cmap, vmin=vmin, vmax=vmax)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(label, color="#8b949e", fontsize=7)
    cb.ax.yaxis.set_tick_params(labelsize=6, color="#8b949e")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="#8b949e")
    ax.set_title(title, fontsize=9)
    ax.set_facecolor("#0d1117")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

# row 0: h maps
_imshow(axes[0,0], h_nm,  ext_m, "RdBu", label="h [nm]",  title="h  (original)")
_imshow(axes[0,1], hs_nm, ext_m, "RdBu", label="h [nm]",  title="h + lambda/2")
_imshow(axes[0,2], hs_nm - h_nm, ext_m, "RdBu", label="dh [nm]", title="diff h  (should be uniform lambda/2)")

# row 1: CMOS intensity
vmax_I = max(I_o_np.max(), I_s_np.max())
_imshow(axes[1,0], I_o_np,    ext_c, "inferno", vmin=0, vmax=vmax_I, label="I [a.u.]", title="Intensity  (h)")
_imshow(axes[1,1], I_s_np,    ext_c, "inferno", vmin=0, vmax=vmax_I, label="I [a.u.]", title="Intensity  (h + lambda/2)")
_imshow(axes[1,2], I_diff_np, ext_c, "RdBu",    label="|dI| [a.u.]", title="|delta I|  (should be ~0)")

# row 2: CMOS phase
phi_max = max(abs(phi_o.min()), abs(phi_o.max()), abs(phi_s.min()), abs(phi_s.max()))
_imshow(axes[2,0], phi_o,    ext_c, "viridis", vmin=-phi_max, vmax=phi_max, label="phi [rad]", title="Phase  (h, unwrapped)")
_imshow(axes[2,1], phi_s,    ext_c, "viridis", vmin=-phi_max, vmax=phi_max, label="phi [rad]", title="Phase  (h + lambda/2, unwrapped)")
abs_dphi = max(abs(phi_diff.min()), abs(phi_diff.max()), 1e-12)
_imshow(axes[2,2], phi_diff, ext_c, "RdBu", vmin=-abs_dphi, vmax=abs_dphi, label="dphi [rad]", title="delta_phi  (should be ~0)")

plt.tight_layout()
out = "output/check_phase_period.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"Saved -> {out}")
