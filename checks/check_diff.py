import torch, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from sensor_model import HolographicSensor, gaussian_bump
from sensor_params import wavelength, mem_res, mem_pitch, cmos_res, grid_res, distance

device = "cpu"
sensor = HolographicSensor(
    wavelength=wavelength, mem_res=mem_res, mem_pitch=mem_pitch,
    cmos_res=cmos_res, grid_res=grid_res, distance=distance, device=device,
).to(device)

h0 = gaussian_bump(N=mem_res, dx=mem_pitch, amplitude=0,          sigma=100e-6, device=device)
h1 = gaussian_bump(N=mem_res, dx=mem_pitch, amplitude=wavelength, sigma=100e-6, device=device)

with torch.no_grad():
    I0 = sensor(h0)
    I1 = sensor(h1)

diff = (I0 - I1).abs()
print(f"max diff : {diff.max().item():.2e}")
print(f"mean diff: {diff.mean().item():.2e}")
print(f"I0 max   : {I0.max().item():.8f}")
print(f"I1 max   : {I1.max().item():.8f}")
