"""
sensor_params.py
----------------
Shared optical system parameters for the holographic tactile sensor.
Import these by name in each sanity script.
"""

wavelength = 632.8e-9   # illumination wavelength [m]  (He-Ne laser)
mem_res    = 512        # membrane pixel count (square)  -> 5.12 mm physical
mem_pitch  = 10e-6      # membrane pixel pitch [m]
cmos_res   = 1024       # CMOS pixel count (square)      -> 10.24 mm physical
grid_res   = 1536       # simulation grid (zero-padding) -> 15.36 mm
distance   = 5e-3       # propagation distance membrane -> CMOS [m]

# Physical sizes (for reference):
#   membrane : mem_res  * mem_pitch = 5.12  mm
#   CMOS     : cmos_res * mem_pitch = 10.24 mm  (2x membrane -> boundary visible)
#   grid     : grid_res * mem_pitch = 15.36 mm  (zero-padded ASM)
