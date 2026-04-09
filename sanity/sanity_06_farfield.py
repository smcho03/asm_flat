"""
sanity_06_farfield.py
---------------------
Far-field / Fraunhofer limit:  |FT(A)|² 검증
h=0 고정, 진폭 A 변조 사용.

Physics: I(CMOS) → |FT(A)|²  as d → ∞

Cases:
  rect    : A = 전체 membrane aperture (5.12mm rect)
               → FT = sinc²  (null = λd/a, d=50m 에서 null=6.2mm ≈ CMOS 반폭)
  delta   : A = 중앙 단일 픽셀  (a=10μm → Fraunhofer d>>0.16mm, 이미 near-field=far-field)
               → FT = uniform (constant)  at all d ≥ 5mm
  Gaussian: A = Gaussian (σ=1mm)  → Fraunhofer d>>1.6m
               → FT = Gaussian (σ_f = λd/2πσ)  at d=50m: σ_f=5mm (CMOS 반폭과 유사)

Layout per image: 3 rows × 5 cols
  row 0 : ref  A=full aperture (rect, h=0)  — col 0: A map, cols 1-4: I at d1…d4
  row 1 : test A=specific source            — col 0: A map, cols 1-4: I at d1…d4
  row 2 : diff = I_test − I_ref             — col 0: ΔA,   cols 1-4: ΔI at d1…d4

Distances: 5mm, 1m, 10m, 50m
각 패널 독립 정규화 (per-panel vmax).
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from sensor_model  import HolographicSensor
from sensor_utils  import STYLE
from sensor_params import wavelength, mem_res, mem_pitch, cmos_res, grid_res


DISTANCES   = [5e-3, 1.0, 10.0, 50.0]
DIST_LABELS = ["d = 5 mm", "d = 1 m", "d = 10 m", "d = 50 m"]


def _make_sensors(device):
    return [
        HolographicSensor(
            wavelength=wavelength, mem_res=mem_res, mem_pitch=mem_pitch,
            cmos_res=cmos_res, grid_res=grid_res, distance=d, device=device,
        ).to(device)
        for d in DISTANCES
    ]


def _plot_case(
    name, A_ref_np, A_test_np,
    I_refs, I_tests,
    mem_coords, cmos_coords,
    out_path, suptitle, color, expect_str,
):
    m_um  = mem_coords  * 1e6
    c_um  = cmos_coords * 1e6
    ext_m = [m_um.min(), m_um.max(), m_um.min(), m_um.max()]
    ext_c = [c_um.min(), c_um.max(), c_um.min(), c_um.max()]

    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(3, 5, figsize=(24, 13))
    fig.suptitle(suptitle, fontsize=10, color=color)
    fig.patch.set_facecolor("#0d1117")

    def _style(ax):
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="#8b949e", labelsize=6)
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")
        ax.grid(color="#30363d", linewidth=0.3, linestyle=":")

    def _cb(ax, im, label):
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(label, color="#8b949e", fontsize=6)
        cb.ax.yaxis.set_tick_params(color="#8b949e", labelsize=5)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="#8b949e")

    dA_np     = A_test_np - A_ref_np
    A_maps    = [A_ref_np, A_test_np, dA_np]
    A_titles  = ["A = rect aperture (ref)", f"A = {name}", "dA = test - ref"]
    row_labels = ["I_ref (rect)", f"I_test ({name})", "dI"]

    # ---- col 0: amplitude maps ----------------------------------------
    for row in range(3):
        ax  = axes[row, 0]
        Am  = A_maps[row]
        if row == 2:
            vabs = max(float(np.abs(Am).max()), 1e-10)
            norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
            im = ax.imshow(Am.T, extent=ext_m, origin="lower", aspect="auto",
                           cmap="RdBu", norm=norm)
            _cb(ax, im, "dA")
        else:
            vmax = max(float(Am.max()), 1e-10)
            im = ax.imshow(Am.T, extent=ext_m, origin="lower", aspect="auto",
                           cmap="viridis", vmin=0, vmax=vmax)
            _cb(ax, im, "A [a.u.]")
        ax.set_title(A_titles[row], fontsize=8,
                     color=color if row == 1 else "#e6edf3")
        ax.set_xlabel("y [um]", fontsize=7, color="#8b949e")
        ax.set_ylabel(row_labels[row], fontsize=8, color=color)
        _style(ax)

    # ---- cols 1-4: per-panel independent normalisation ----------------
    for col_idx, (d_label, d_val, I_ref, I_test) in enumerate(
            zip(DIST_LABELS, DISTANCES, I_refs, I_tests)):

        null_mm = wavelength * d_val / (mem_res * mem_pitch) * 1e3
        suffix  = f"\nnull={null_mm:.1f} mm"

        dI     = I_test - I_ref
        dI_abs = max(float(np.abs(dI).max()), 1e-10)

        # row 0: I_ref
        ax  = axes[0, col_idx + 1]
        vm  = max(float(I_ref.max()), 1e-10)
        im  = ax.imshow(I_ref.T, extent=ext_c, origin="lower", aspect="auto",
                        cmap="inferno", vmin=0, vmax=vm)
        _cb(ax, im, "I [a.u.]")
        ax.set_title(f"ref  {d_label}{suffix}", fontsize=7, color="#e6edf3")
        ax.set_xlabel("y' [um]", fontsize=7, color="#8b949e")
        ax.set_ylabel("x' [um]", fontsize=7, color="#8b949e")
        _style(ax)

        # row 1: I_test  (independent vmax)
        ax  = axes[1, col_idx + 1]
        vm  = max(float(I_test.max()), 1e-10)
        im  = ax.imshow(I_test.T, extent=ext_c, origin="lower", aspect="auto",
                        cmap="inferno", vmin=0, vmax=vm)
        _cb(ax, im, "I [a.u.]")
        ax.set_title(f"test  {d_label}{suffix}", fontsize=7, color=color)
        ax.set_xlabel("y' [um]", fontsize=7, color="#8b949e")
        ax.set_ylabel("x' [um]", fontsize=7, color="#8b949e")
        _style(ax)

        # row 2: dI
        ax   = axes[2, col_idx + 1]
        norm = TwoSlopeNorm(vmin=-dI_abs, vcenter=0, vmax=dI_abs)
        im   = ax.imshow(dI.T, extent=ext_c, origin="lower", aspect="auto",
                         cmap="RdBu", norm=norm)
        _cb(ax, im, "dI [a.u.]")
        ax.set_title(f"diff  {d_label}{suffix}", fontsize=7, color="#e6edf3")
        ax.set_xlabel("y' [um]", fontsize=7, color="#8b949e")
        ax.set_ylabel("x' [um]", fontsize=7, color="#8b949e")
        _style(ax)

    fig.text(0.5, 0.005,
             f"Per-panel independent normalisation.  h=0 for all.  "
             f"Fraunhofer: d >> a2/lam = {(mem_res*mem_pitch)**2/wavelength:.0f} m.  "
             f"Expected: {expect_str}",
             ha="center", fontsize=8, color="#8b949e")

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  saved -> {out_path.name}", flush=True)


def run(out_dir: Path, device: str = "cpu") -> None:
    print("[06] Far-field FT verification  (A modulation, h=0) ...", flush=True)

    N  = mem_res
    dx = mem_pitch
    M  = cmos_res

    sensors = _make_sensors(device)

    mem_coords  = (np.arange(N) - (N - 1) * 0.5) * dx
    cmos_coords = (np.arange(M) - (M - 1) * 0.5) * dx

    coords = torch.tensor(mem_coords, dtype=torch.float32, device=device)
    X, Y   = torch.meshgrid(coords, coords, indexing="ij")
    h_zero = torch.zeros(N, N, device=device)

    # ref: full aperture (rect)
    A_ref = torch.ones(N, N, device=device)

    # delta: single center pixel
    A_delta = torch.zeros(N, N, device=device)
    A_delta[N // 2, N // 2] = 1.0

    # Gaussian amplitude  sigma=1mm so that at d=50m: sigma_f = lambda*d/(2pi*sigma) = 5.0mm
    sigma_g = 1e-3
    A_gauss = torch.exp(-(X**2 + Y**2) / (2.0 * sigma_g**2))

    # rect sub-aperture: central 1/4 of membrane (1.28mm) → Fraunhofer d>>2.6m
    # null at d=50m: lambda*d/a = 632.8e-9*50/1.28e-3 = 24.7mm >> CMOS → main lobe only
    # Use smaller sub-rect: 64px = 640um → null at d=50m = 49.5mm >> CMOS
    # Actually for rect FT, use A_ref (full 5.12mm) — its sinc null at d=50m = 6.2mm
    # → just outside CMOS, so the main lobe fills the CMOS (qualitatively shows FT)

    sigma_g_str = f"sigma={sigma_g*1e3:.0f}mm"

    CASES = [
        dict(name="delta", A_test=A_delta, color="#3fb950",
             expect="uniform (constant)  [single pixel: far-field at d>0.16mm]"),
        dict(name="Gaussian", A_test=A_gauss, color="#e3b341",
             expect=f"Gaussian (sigma_f=lam*d/2pi*sigma)  [{sigma_g_str}: at d=50m sigma_f=5mm]"),
    ]

    A_ref_np = A_ref.cpu().numpy()

    # Also compute I_ref once per distance
    I_refs_base = []
    with torch.no_grad():
        for s in sensors:
            I_refs_base.append(s(h_zero, A=A_ref).cpu().numpy())

    # rect image: just show I_ref (rect aperture) evolving with distance → sinc²
    # Use A_test = A_ref (same), so test row = ref row, diff = 0
    # Instead make A_test = slightly modified rect to show diff is 0 → clean sinc²
    # Show rect aperture evolving alone (1 row, 4 dist) in a simpler plot
    print("  [rect] computing ...", flush=True)
    plt.rcParams.update(STYLE)
    fig_r, axes_r = plt.subplots(1, 5, figsize=(24, 5))
    fig_r.suptitle(
        f"Sanity 06 [rect]  —  Full rect aperture (A=1, h=0)  ->  FT = sinc2\n"
        f"aperture={N*dx*1e3:.1f}mm  Fraunhofer: d>>{(N*dx)**2/wavelength:.0f}m  "
        f"sinc null = lam*d/a",
        fontsize=10, color="#58a6ff",
    )
    fig_r.patch.set_facecolor("#0d1117")

    def _style(ax):
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="#8b949e", labelsize=6)
        for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
        ax.grid(color="#30363d", linewidth=0.3, linestyle=":")

    def _cb2(ax, im, label):
        cb = fig_r.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(label, color="#8b949e", fontsize=6)
        cb.ax.yaxis.set_tick_params(color="#8b949e", labelsize=5)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="#8b949e")

    m_um  = mem_coords * 1e6
    c_um  = cmos_coords * 1e6
    ext_m = [m_um.min(), m_um.max(), m_um.min(), m_um.max()]
    ext_c = [c_um.min(), c_um.max(), c_um.min(), c_um.max()]

    # col 0: A map
    ax = axes_r[0]
    im = ax.imshow(A_ref_np.T, extent=ext_m, origin="lower", aspect="auto",
                   cmap="viridis", vmin=0, vmax=1)
    _cb2(ax, im, "A [a.u.]")
    ax.set_title("A(x,y) = rect aperture", fontsize=8, color="#58a6ff")
    ax.set_xlabel("y [um]", fontsize=7, color="#8b949e")
    ax.set_ylabel("x [um]", fontsize=7, color="#8b949e")
    _style(ax)

    for col_idx, (d_label, d_val, I_r) in enumerate(
            zip(DIST_LABELS, DISTANCES, I_refs_base)):
        ax  = axes_r[col_idx + 1]
        null_mm = wavelength * d_val / (N * dx) * 1e3
        vm  = max(float(I_r.max()), 1e-10)
        im  = ax.imshow(I_r.T, extent=ext_c, origin="lower", aspect="auto",
                        cmap="inferno", vmin=0, vmax=vm)
        _cb2(ax, im, "I [a.u.]")
        ax.set_title(f"{d_label}\nnull={null_mm:.1f}mm", fontsize=7, color="#e6edf3")
        ax.set_xlabel("y' [um]", fontsize=7, color="#8b949e")
        ax.set_ylabel("x' [um]", fontsize=7, color="#8b949e")
        _style(ax)

    plt.tight_layout()
    out_rect = out_dir / "sanity_06_rect.png"
    plt.savefig(out_rect, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  saved -> {out_rect.name}", flush=True)

    # delta and Gaussian: 3-row layout
    for case in CASES:
        name   = case["name"]
        A_test = case["A_test"]
        print(f"  [{name}] computing ...", flush=True)

        I_tests = []
        with torch.no_grad():
            for s in sensors:
                I_tests.append(s(h_zero, A=A_test).cpu().numpy())

        _plot_case(
            name        = name,
            A_ref_np    = A_ref_np,
            A_test_np   = A_test.cpu().numpy(),
            I_refs      = I_refs_base,
            I_tests     = I_tests,
            mem_coords  = mem_coords,
            cmos_coords = cmos_coords,
            out_path    = out_dir / f"sanity_06_{name}.png",
            suptitle    = (
                f"Sanity 06  [{name}]  —  Far-field FT: A={name}, h=0\n"
                f"{case['expect']}\n"
                f"lam={wavelength*1e9:.1f}nm  aperture={N*dx*1e3:.1f}mm  "
                f"Per-panel independent normalisation"
            ),
            color      = case["color"],
            expect_str = case["expect"],
        )


if __name__ == "__main__":
    out_dir = Path(__file__).parent / "sanity_results"
    out_dir.mkdir(exist_ok=True)
    run(out_dir, device="cuda" if torch.cuda.is_available() else "cpu")
