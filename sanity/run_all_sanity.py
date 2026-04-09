"""
run_all_sanity.py
-----------------
All sanity checks -> sanity_results/{MMDD_HHMMSS}/

Usage:
  python run_all_sanity.py
"""
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime

import torch

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent))          # sanity/ (for sibling imports)
sys.path.insert(0, str(Path(__file__).parent.parent))  # root   (for sensor_model etc.)

timestamp = datetime.now().strftime("%m%d_%H%M%S")
out_dir   = Path(__file__).parent / 'sanity_results' / timestamp
out_dir.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"


def section(msg: str) -> None:
    print(f"\n{'='*54}\n  {msg}\n{'='*54}", flush=True)


def run_safe(name: str, func) -> None:
    print(f"\n[{name}] running ...", flush=True)
    t0 = time.time()
    try:
        func(out_dir, device=device)
        print(f"[{name}] done  ({time.time()-t0:.1f}s)", flush=True)
    except Exception:
        print(f"[{name}] ERROR:", flush=True)
        traceback.print_exc()


section("Holographic Tactile Sensor - Sanity Checks")
print(f"  device : {device}", flush=True)
print(f"  output : {out_dir}", flush=True)

from sanity_01_flat_mirror          import run as run_01
from sanity_02_symmetry             import run as run_02
from sanity_03_height_sweep         import run as run_03
from sanity_04_deformation_patterns import run as run_04
from sanity_05_animation            import run as run_05

t_total = time.time()
run_safe('01 flat_mirror',          run_01)
run_safe('02 symmetry',             run_02)
run_safe('03 height_sweep',         run_03)
run_safe('04 deformation_patterns', run_04)
run_safe('05 animation',            run_05)

section("Complete")
files = sorted(out_dir.glob('*.png'))
print(f"  total elapsed : {time.time()-t_total:.1f}s", flush=True)
print(f"  saved files   ({len(files)}):", flush=True)
for f in files:
    print(f"    {f.name}", flush=True)
