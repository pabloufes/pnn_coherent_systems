"""Run the fiber-model validation: RF power fading (SCM CW link).

Equivalent to SCM_FechamentoCW.m, SCM_FechamentoCW3dB.m, and SCM_FechamentoL.m.
Generates plots and saves them to artifacts/figures/.
"""

import os
import sys
import tempfile
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add the project root to the import path.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

output_base = os.environ.get("SIM_OUTPUT_DIR", os.path.join(BASE_DIR, "artifacts"))
FIG_DIR = os.path.join(output_base, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

from src.dsp.fiber import fibra_monomodo, FiberParams
from src.dsp.im_dd import (
    DFBParams,
    simular_scm_fechamento_cw,
    calcular_rf_power_fading_teorico,
)


def run_tests():
    print("=" * 70)
    print("Starting fiber model validation: SCM CW link (RF power fading)")
    print("=" * 70)

    l_km = 100.0
    lamb0 = 1563e-9
    c_light = 3e8
    fiber_params = FiberParams(alpha_db=0.2, d_1550=16.0, slope_s=0.09, n2=3.2e-20, a_eff=70e-12, c=c_light)
    dfb_params = DFBParams(i_bias=100.0, i_th=30.0, k_slope=0.999, i_factor=50.0)

    d_disp_1563 = fiber_params.d_1550 + fiber_params.slope_s * (lamb0 * 1e9 - 1550.0)

    # 1. Simulation 1: L = 100 km (0.2 to 25 GHz).
    print("\n1. Simulating RF power fading for L = 100 km...")
    freqs_ghz = np.arange(0.2, 25.1, 0.1)
    freqs_hz = freqs_ghz * 1e9

    resp_sim_db, _ = simular_scm_fechamento_cw(
        freqs_hz=freqs_hz, length_km=l_km, wavelength_m=lamb0,
        regime=0, np_points=512, num_cycles=10,
        dfb_params=dfb_params, fiber_params=fiber_params
    )
    resp_th_db, notches = calcular_rf_power_fading_teorico(
        freqs_hz=freqs_hz, length_km=l_km, wavelength_m=lamb0,
        d_disp_ps_nm_km=d_disp_1563, c=c_light
    )

    plt.figure(figsize=(11, 6))
    plt.plot(freqs_ghz, resp_sim_db, "b-", linewidth=2, label="Simulation (`fibra_monomodo`)")
    plt.plot(freqs_ghz, resp_th_db, "r--", linewidth=1.5, alpha=0.8, label="Theory $|\\cos(\\pi D \\lambda^2 L f^2 / c)|$")
    for fn in notches:
        fg = fn / 1e9
        plt.axvline(x=fg, color="gray", linestyle=":", alpha=0.6)
        plt.text(fg, -48, f"{fg:.1f}G", rotation=90, verticalalignment="bottom", fontsize=9)
    plt.title(f"SCM CW Link — RF Power Fading in Fiber (L = {l_km:.0f} km, $\\lambda = 1563$ nm)", fontsize=12)
    plt.xlabel("RF Subcarrier Frequency [GHz]", fontsize=11)
    plt.ylabel("Normalized Frequency Response [dB]", fontsize=11)
    plt.ylim([-55, 5])
    plt.xlim([0, 25])
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    p1 = os.path.join(FIG_DIR, "scm_fechamento_cw_100km.png")
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"-> Saved: {p1}")

    # 2. Simulation 2: vary L (25, 50, and 100 km).
    print("\n2. Simulating RF power fading for different distances (25, 50, 100 km)...")
    lengths_km = [25.0, 50.0, 100.0]
    colors = {25.0: "green", 50.0: "darkorange", 100.0: "blue"}

    plt.figure(figsize=(11, 6))
    for l_val in lengths_km:
        resp_l, _ = simular_scm_fechamento_cw(
            freqs_hz=freqs_hz, length_km=l_val, wavelength_m=lamb0,
            regime=0, np_points=512, dfb_params=dfb_params, fiber_params=fiber_params
        )
        plt.plot(freqs_ghz, resp_l, color=colors[l_val], linewidth=1.8, label=f"L = {l_val:.0f} km")

    plt.title("Effect of Fiber Length L on RF Power Fading (IM-DD)", fontsize=12)
    plt.xlabel("RF Subcarrier Frequency [GHz]", fontsize=11)
    plt.ylabel("Normalized Frequency Response [dB]", fontsize=11)
    plt.ylim([-55, 5])
    plt.xlim([0, 25])
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    p2 = os.path.join(FIG_DIR, "scm_fechamento_cw_distancias.png")
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"-> Saved: {p2}")

    # 3. Simulation 3: linear vs. split-step (SSFM).
    print("\n3. Validating Split-Step Fourier Method (regime 1) against linear propagation (regime 0)...")
    freqs_ssfm = np.arange(0.5, 20.1, 0.25) * 1e9
    params_lin_ssfm = FiberParams(d_1550=16.0, slope_s=0.09, n2=0.0)

    resp_ssfm_db, _ = simular_scm_fechamento_cw(
        freqs_hz=freqs_ssfm, length_km=l_km, wavelength_m=lamb0,
        regime=1, np_points=512, dfb_params=dfb_params, fiber_params=params_lin_ssfm, num_ssfm_steps=20
    )
    resp_lin_db, _ = simular_scm_fechamento_cw(
        freqs_hz=freqs_ssfm, length_km=l_km, wavelength_m=lamb0,
        regime=0, np_points=512, dfb_params=dfb_params, fiber_params=params_lin_ssfm
    )

    plt.figure(figsize=(10, 5.5))
    plt.plot(freqs_ssfm/1e9, resp_lin_db, "b-", linewidth=2.5, label="Regime 0: Linear Transfer Function")
    plt.plot(freqs_ssfm/1e9, resp_ssfm_db, "r--", linewidth=1.8, label="Regime 1: Split-Step Fourier Method (SSFM)")
    plt.title("SSFM Validation vs. Linear Model (L = 100 km)", fontsize=12)
    plt.xlabel("RF Subcarrier Frequency [GHz]", fontsize=11)
    plt.ylabel("Normalized Frequency Response [dB]", fontsize=11)
    plt.ylim([-55, 5])
    plt.xlim([0, 20])
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    p3 = os.path.join(FIG_DIR, "scm_fechamento_cw_ssfm_vs_linear.png")
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"-> Saved: {p3}")

    # 4. Simulation 4: 3 dB bandwidth vs. L (SCM_FechamentoCW3dB.m).
    print("\n4. Calculating 3 dB bandwidth vs. fiber length L...")
    lengths_test = np.arange(10, 105, 10)
    bw_sim = []
    bw_th = []
    freqs_fine = np.linspace(0.1e9, 20e9, 400)

    for l_val in lengths_test:
        resp_l, _ = simular_scm_fechamento_cw(
            freqs_hz=freqs_fine, length_km=l_val, wavelength_m=lamb0,
            regime=0, np_points=512, dfb_params=dfb_params, fiber_params=fiber_params
        )
        idx_3db = np.where(resp_l <= -3.0)[0]
        bw_sim.append(freqs_fine[idx_3db[0]] / 1e9 if len(idx_3db) > 0 else np.nan)
        bw_th.append(np.sqrt(c_light / (4 * (d_disp_1563 * 1e-6) * (lamb0**2) * (l_val * 1e3))) / 1e9)

    plt.figure(figsize=(9, 5))
    plt.plot(lengths_test, bw_sim, "bo-", linewidth=2, markersize=6, label="Simulation (`fibra_monomodo`)")
    plt.plot(lengths_test, bw_th, "r--", linewidth=1.8, label=r"Theory $BW_{3dB} = \sqrt{c / (4 D \lambda^2 L)}$")
    plt.title("3 dB Bandwidth vs. Fiber Length L (IM-DD)", fontsize=12)
    plt.xlabel("Fiber Length L [km]", fontsize=11)
    plt.ylabel("3 dB Bandwidth [GHz]", fontsize=11)
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    p4 = os.path.join(FIG_DIR, "scm_largura_banda_3db.png")
    plt.savefig(p4, dpi=300)
    plt.close()
    print(f"-> Saved: {p4}")

    print("\nAll four CW link simulations completed successfully!")


if __name__ == "__main__":
    run_tests()
