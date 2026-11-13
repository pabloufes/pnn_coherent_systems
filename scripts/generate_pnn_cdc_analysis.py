"""Generate BER/EVM sweeps for conventional and joint PNN-CDC equalization.

The distance sweep is linear; the launch-power sweep uses SSFM with Kerr.
Results are written under artifacts/reports and artifacts/figures.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

#from notebooks.ook_mzm_pushpull_ber_distancia import MAX_DISTANCE_KM

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dsp.edfa import EDFAIdeal
from src.dsp.fiber import FiberParams, fibra_monomodo
from src.dsp.joint_pnn_cdc_equalizer import cd_compensation_joint_pnn
from src.dsp.noise_models import generate_correlated_phases
from src.dsp.receiver import coherent_detection, cd_compensation, sample_to_symbols, demap_qpsk
from src.dsp.transmitter import (
    ElectricalFilterParams, IQMZMParams, generate_qpsk_symbols, modulate_optical_signal,
)
from src.models.pnn import PhotonicNeuralNetwork
from src.utils.metrics import calculate_ber, calculate_evm
from scripts.optimize_pnn_pso import particle_swarm_optimize


# Reproducible study configuration. Adjust PSO parameters for convergence studies.
NUM_TRAIN = 2**10
NUM_TEST = 2**12
SPS = 8
SYMBOL_RATE = 10e9
WAVELENGTH = 1550e-9
DT = 1.0 / (SPS * SYMBOL_RATE)
SPAN_KM = 50.0
MAX_DISTANCE_KM = 1000.0
DISTANCES_KM = np.arange(50.0, MAX_DISTANCE_KM + SPAN_KM, SPAN_KM)
POWER_DBM = np.arange(-20.0, 20.0 + 2.0, 2.0)
LINEWIDTHS_HZ = (0.0, 100e3, 1e6)
RHO = 0.0
PNN_TAPS = 8
PNN_DELAY_PS = 25.0
PNN_NORMALIZE = True
PSO_SWARM = 30
PSO_ITERATIONS = 60
REG_REL = 1e-3
SEED = 20261104
PLOT_FLOOR = 1.0 / (2 * NUM_TEST)

FIBER = FiberParams(alpha_db=0.2, d_1550=16.0, slope_s=0.09,
                    n2=3.2e-20, a_eff=70e-12)
IQ = IQMZMParams(electrical_response=False)
FILTER = ElectricalFilterParams(
    tipo="cosseno_levantado", largura_banda_hz=2 * SYMBOL_RATE, ordem=7
)
ARTIFACTS = ROOT / "artifacts" / str(MAX_DISTANCE_KM)
FIGURES = ARTIFACTS / "figures"
REPORTS = ARTIFACTS / "reports"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)
CONSTELLATION_NPZ = REPORTS / "pnn_cdc_constellation_points.npz"
CONSTELLATION_INDEX = REPORTS / "pnn_cdc_constellation_points.json"

estimate_phase = False
phase_rotation_estimate = False
OBJECTIVES = ("BER", "EVM")
FAMILIES = ("CDC", "joint_CDC")
STRATEGIES = (
    "No PNN / no CDC", "No PNN / CDC", "PNN / no CDC",
    "PNN + conventional CDC (joint training)",
    "PNN + joint CDC (joint training)",
)


def make_tx(symbols: np.ndarray, launch_dbm: float):
    return modulate_optical_signal(
        symbols, sps=SPS, launch_power_dbm=launch_dbm,
        symbol_rate=SYMBOL_RATE, modulator="iq_mzm", iq_params=IQ,
        filter_params=FILTER, plot_filter=False, n_plot=0,
    )


def propagate_spans(field: np.ndarray, t: np.ndarray, distance_km: float,
                    include_kerr: bool) -> np.ndarray:
    spans_float = distance_km / SPAN_KM
    spans = int(round(spans_float))
    if spans < 1 or not np.isclose(spans_float, spans):
        raise ValueError("Distance must be a positive multiple of 50 km.")
    out = np.asarray(field, dtype=complex).copy()
    edfa = EDFAIdeal(gain_db=FIBER.alpha_db * SPAN_KM)
    for _ in range(spans):
        out, _, _ = fibra_monomodo(
            t, out, wavelength=WAVELENGTH, t0=1.0 / SYMBOL_RATE,
            length_km=SPAN_KM, regime=1 if include_kerr else 0,
            params=FIBER, num_steps=5,
        )
        out = edfa.amplify(out)
    return out


def align_evm(tx_symbols: np.ndarray, rx_symbols: np.ndarray) -> float:
    # Same RMS EVM definition used by calculate_evm (power normalization and constant phase).
    return calculate_evm(rx_symbols,correct_phase_rotation=phase_rotation_estimate)


def estimate_constant_phase(tx_symbols: np.ndarray, rx_symbols: np.ndarray) -> float:
    n = min(len(tx_symbols), len(rx_symbols))
    tx = np.asarray(tx_symbols[:n], dtype=complex)
    rx = np.asarray(rx_symbols[:n], dtype=complex)
    tx = tx / (np.sqrt(np.mean(np.abs(tx) ** 2)) + 1e-30)
    rx = rx / (np.sqrt(np.mean(np.abs(rx) ** 2)) + 1e-30)
    return float(np.angle(np.mean(rx * np.conj(tx))))


def evaluate(field: np.ndarray, phase_lo: np.ndarray, tx_symbols: np.ndarray,
             tx_bits: np.ndarray, model: PhotonicNeuralNetwork | None,
             equalizer: str, distance_km: float, reg_rel: float = REG_REL,
             phase_offset: float | None = None,
             estimate_phase_from_known_symbols: bool = False):
    if model is None:
        optical = field
    else:
        _, optical = model.forward(field, dt_s=DT, return_optical_field=True)
    bb = coherent_detection(optical, phase_lo=phase_lo)
    if equalizer == "CDC":
        bb = cd_compensation(bb, DT, distance_km, dispersion_d=FIBER.d_1550)
    elif equalizer == "joint_CDC":
        h = model.frequency_response(np.fft.fftfreq(len(bb), d=DT))
        eps = reg_rel * np.mean(np.abs(h) ** 2)
        bb = cd_compensation_joint_pnn(
            bb, DT, distance_km, model, dispersion_d=FIBER.d_1550,
            wavelength=WAVELENGTH, reg_factor=eps,
        )
    elif equalizer != "none":
        raise ValueError(f"Unknown equalizer: {equalizer}")
    symbols_raw = sample_to_symbols(bb, sps=SPS)
    n = min(len(tx_symbols), len(symbols_raw))
    symbols_raw = symbols_raw[:n]
    if estimate_phase_from_known_symbols:
        phase_offset = estimate_constant_phase(tx_symbols[:n], symbols_raw)
    if phase_offset is None:
        phase_offset = 0.0
    symbols = symbols_raw * np.exp(-1j * phase_offset)
    bits_rx = demap_qpsk(symbols)
    bit_count = min(len(tx_bits), len(bits_rx))
    return {
        "ber": calculate_ber(tx_bits[:bit_count], bits_rx[:bit_count]),
        "evm_percent": align_evm(tx_symbols[:n], symbols_raw),
        "symbols": symbols[:n],
        "symbols_raw": symbols_raw,
        "phase_offset_rad": float(phase_offset),
        "bits_compared": int(bit_count),
    }


def train_pnn(field: np.ndarray, phase_lo: np.ndarray, tx_symbols: np.ndarray,
              tx_bits: np.ndarray, distance_km: float, objective_name: str,
              equalizer: str, seed: int):
    model = PhotonicNeuralNetwork(
        num_channels=PNN_TAPS, base_delay_ps=PNN_DELAY_PS,
        normalize_power=PNN_NORMALIZE, responsivity=1.0,
    )
    bounds = [(0.0, 1.0)] * PNN_TAPS + [(-np.pi, np.pi)] * PNN_TAPS

    def objective(position):
        model.set_weights(position[:PNN_TAPS], position[PNN_TAPS:])
        result = evaluate(
            field, phase_lo, tx_symbols, tx_bits, model, equalizer,
            distance_km, estimate_phase_from_known_symbols=estimate_phase,
        )
        if objective_name == "BER":
            return result["ber"]
        return result["evm_percent"]

    result = particle_swarm_optimize(
        objective, bounds, swarm_size=PSO_SWARM,
        max_iterations=PSO_ITERATIONS,
        periodic_dimensions=range(PNN_TAPS, 2 * PNN_TAPS), seed=seed,
    )
    model.set_weights(result["amplitude_weights"], result["phase_weights"])
    result["model"] = model
    return result


def phase_pair(n: int, linewidth_hz: float, seed: int):
    return generate_correlated_phases(
        n, DT, linewidth_hz, correlation_rho=RHO, seed=seed
    )


def run_point(sweep_name: str, x_value: float, linewidth_hz: float,
              tx_train_base: np.ndarray, t_train: np.ndarray,
              tx_test_base: np.ndarray, t_test: np.ndarray,
              train_symbols: np.ndarray, train_bits: np.ndarray,
              test_symbols: np.ndarray, test_bits: np.ndarray,
              train_phases, test_phases, include_kerr: bool, point_index: int,
              constellation_records: list):
    distance_km = x_value if sweep_name == "distance" else MAX_DISTANCE_KM
    if sweep_name == "distance":
        launch_dbm = 0.0
    else:
        launch_dbm = x_value
    phi_tx_train, phi_lo_train = train_phases
    phi_tx_test, phi_lo_test = test_phases
    field_train = propagate_spans(
        tx_train_base * np.exp(1j * phi_tx_train), t_train,
        distance_km, include_kerr,
    )
    field_test = propagate_spans(
        tx_test_base * np.exp(1j * phi_tx_test), t_test,
        distance_km, include_kerr,
    )

    fixed = {}
    for condition, eq in (("No PNN / no CDC", "none"), ("No PNN / CDC", "CDC")):
        training_reference = evaluate(
            field_train, phi_lo_train, train_symbols, train_bits, None, eq,
            distance_km, estimate_phase_from_known_symbols=estimate_phase,
        )
        fixed[condition] = evaluate(
            field_test, phi_lo_test, test_symbols, test_bits, None, eq,
            distance_km, phase_offset=training_reference["phase_offset_rad"],
        )
    records = []
    constellations = {}
    for objective_index, objective_name in enumerate(OBJECTIVES):
        seed_base = SEED + point_index * 100 + int(linewidth_hz / 1e3) + objective_index * 17
        model_results = {}
        for model_index, (key, eq) in enumerate((
            ("PNN / no CDC", "none"),
            ("PNN + conventional CDC (joint training)", "CDC"),
            ("PNN + joint CDC (joint training)", "joint_CDC"),
        )):
            pso = train_pnn(
                field_train, phi_lo_train, train_symbols, train_bits,
                distance_km, objective_name, eq,
                seed_base + model_index * 1009,
            )
            model_results[key] = (pso, eq)

        family_map = {
            "conventional": ("PNN / no CDC", "PNN + conventional CDC (joint training)"),
            "joint": ("PNN / no CDC", "PNN + joint CDC (joint training)"),
        }
        for family, conditions in family_map.items():
            for condition in conditions:
                pso, trained_eq = model_results[condition]
                training_reference = evaluate(
                    field_train, phi_lo_train, train_symbols, train_bits,
                    pso["model"], trained_eq, distance_km,
                    estimate_phase_from_known_symbols=estimate_phase,
                )
                # PNN-only output is evaluated without DSP; trained joint outputs use their DSP.
                eval_eq = "none" if condition == "PNN / no CDC" else trained_eq
                measured = evaluate(
                    field_test, phi_lo_test, test_symbols, test_bits,
                    pso["model"], eval_eq, distance_km,
                    phase_offset=training_reference["phase_offset_rad"],
                )
                records.append({
                    "sweep": sweep_name, "x": float(x_value),
                    "distance_km": float(distance_km),
                    "launch_power_dbm": float(launch_dbm),
                    "linewidth_hz": float(linewidth_hz),
                    "training_objective": objective_name,
                    "equalizer_family": family,
                    "strategy": condition,
                    "ber": measured["ber"], "evm_percent": measured["evm_percent"],
                    "bits_compared": measured["bits_compared"],
                    "best_training_objective": float(pso["best_fitness"]),
                    "pso_evaluations": int(pso["evaluations"]),
                    "amplitude_weights": pso["amplitude_weights"].tolist(),
                    "phase_weights": pso["phase_weights"].tolist(),
                    "pnn_delay_ps": PNN_DELAY_PS,
                    "include_kerr": bool(include_kerr),
                })
                constellations[(objective_name, family, condition)] = measured["symbols"]
                target = MAX_DISTANCE_KM/2.0 if sweep_name == "distance" else 0.0
                if np.isclose(x_value, target):
                    constellation_records.append({
                        "sweep": sweep_name, "x": float(x_value),
                        "linewidth_hz": float(linewidth_hz),
                        "objective": objective_name, "family": family,
                        "strategy": condition,
                        "rx": measured["symbols"].copy(),
                        "tx": test_symbols[:len(measured["symbols"])].copy(),
                    })
        for condition, result in fixed.items():
            for family in FAMILIES:
                records.append({
                    "sweep": sweep_name, "x": float(x_value),
                    "distance_km": float(distance_km),
                    "launch_power_dbm": float(launch_dbm),
                    "linewidth_hz": float(linewidth_hz),
                    "training_objective": objective_name,
                    "equalizer_family": "conventional" if family == "CDC" else "joint",
                    "strategy": condition, "ber": result["ber"],
                    "evm_percent": result["evm_percent"],
                    "bits_compared": result["bits_compared"],
                    "best_training_objective": np.nan, "pso_evaluations": 0,
                    "amplitude_weights": None, "phase_weights": None,
                    "pnn_delay_ps": PNN_DELAY_PS,
                    "include_kerr": bool(include_kerr),
                })
                target = MAX_DISTANCE_KM/2.0 if sweep_name == "distance" else 0.0
                if np.isclose(x_value, target):
                    constellation_records.append({
                        "sweep": sweep_name, "x": float(x_value),
                        "linewidth_hz": float(linewidth_hz),
                        "objective": objective_name,
                        "family": "conventional" if family == "CDC" else "joint",
                        "strategy": condition,
                        "rx": result["symbols"].copy(),
                        "tx": test_symbols[:len(result["symbols"])].copy(),
                    })
    return records, constellations


def plot_curves(frame: pd.DataFrame, sweep: str):
    x_col = "distance_km" if sweep == "distance" else "launch_power_dbm"
    x_label = "Fiber distance (km)" if sweep == "distance" else "Launch power (dBm)"
    for objective_name in OBJECTIVES:
        for family in ("conventional", "joint"):
            subset = frame[(frame.sweep == sweep) &
                           (frame.training_objective == objective_name) &
                           (frame.equalizer_family == family)]
            fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
            for ax, linewidth in zip(axes, LINEWIDTHS_HZ):
                group = subset[subset.linewidth_hz == linewidth]
                plotted_strategies = STRATEGIES[:3] + (
                    STRATEGIES[3] if family == "conventional" else STRATEGIES[4],
                )
                for strategy in plotted_strategies:
                    g = group[group.strategy == strategy].sort_values(x_col)
                    ax.semilogy(g[x_col], np.maximum(g.ber, PLOT_FLOOR),
                                marker="o", ms=3, label=strategy)
                ax.set_title(f"TX/LO linewidth: {linewidth / 1e3:g} kHz")
                ax.set_xlabel(x_label); ax.grid(True, which="both", alpha=.35)
            axes[0].set_ylabel("BER")
            axes[-1].legend(fontsize=8, loc="best")
            fig.suptitle(f"BER vs {x_label.lower()} — {objective_name} objective — {family} CDC")
            fig.tight_layout()
            fig.savefig(FIGURES / f"{sweep}_ber_{objective_name.lower()}_{family}_cdc.png",
                        dpi=220, bbox_inches="tight")
            plt.close(fig)

            fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
            for ax, linewidth in zip(axes, LINEWIDTHS_HZ):
                group = subset[subset.linewidth_hz == linewidth]
                plotted_strategies = STRATEGIES[:3] + (
                    STRATEGIES[3] if family == "conventional" else STRATEGIES[4],
                )
                for strategy in plotted_strategies:
                    g = group[group.strategy == strategy].sort_values(x_col)
                    ax.plot(g[x_col], g.evm_percent, marker="o", ms=3, label=strategy)
                ax.set_title(f"TX/LO linewidth: {linewidth / 1e3:g} kHz")
                ax.set_xlabel(x_label); ax.grid(True, alpha=.35)
            axes[0].set_ylabel("RMS EVM (%)")
            axes[-1].legend(fontsize=8, loc="best")
            fig.suptitle(f"EVM vs {x_label.lower()} — {objective_name} objective — {family} CDC")
            fig.tight_layout()
            fig.savefig(FIGURES / f"{sweep}_evm_{objective_name.lower()}_{family}_cdc.png",
                        dpi=220, bbox_inches="tight")
            plt.close(fig)


def save_constellation_figures(records: list):
    labels = STRATEGIES[:3]
    for sweep, objective_name, family in (
        (s, o, f) for s in ("distance", "power")
        for o in OBJECTIVES for f in ("conventional", "joint")
    ):
        subset = [r for r in records if r["sweep"] == sweep and
                  r["objective"] == objective_name and r["family"] == family]
        if not subset:
            continue
        pnn_strategy = STRATEGIES[3] if family == "conventional" else STRATEGIES[4]
        strategy_labels = labels + (pnn_strategy,)
        fig, axes = plt.subplots(3, 4, figsize=(16, 11))
        for row, linewidth in enumerate(LINEWIDTHS_HZ):
            for col, strategy in enumerate(strategy_labels):
                ax = axes[row, col]
                item = next((r for r in subset if r["linewidth_hz"] == linewidth and
                             r["strategy"] == strategy), None)
                if item is None:
                    ax.set_axis_off()
                    continue
                # Plot received samples and ideal QPSK reference separately.
                rx = item["rx"]
                tx = item["tx"]
                tx_norm = tx / (np.sqrt(np.mean(np.abs(tx) ** 2)) + 1e-30)
                rx_norm = rx / (np.sqrt(np.mean(np.abs(rx) ** 2)) + 1e-30)
                ax.scatter(tx_norm.real, tx_norm.imag, s=8, marker="x", alpha=.45,
                           color="black", label="TX reference")
                ax.scatter(rx_norm.real, rx_norm.imag, s=7, alpha=.35,
                           color="tab:blue", label="RX")
                ax.set_title(f"{linewidth/1e3:g} kHz — {strategy}", fontsize=8)
                ax.set_xlabel("I (normalized)"); ax.set_ylabel("Q (normalized)")
                ax.grid(True, alpha=.3); ax.set_aspect("equal", adjustable="datalim")
                if row == 0 and col == 0:
                    ax.legend(fontsize=7, loc="best")
        x_text = "300 km" if sweep == "distance" else "0 dBm"
        regime = "linear" if sweep == "distance" else "SSFM with Kerr"
        fig.suptitle(f"Constellations at {x_text} — {objective_name} training — "
                     f"{family} CDC — {regime}")
        fig.tight_layout()
        fig.savefig(FIGURES / f"{sweep}_constellations_{objective_name.lower()}_"
                    f"{family}_cdc.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def save_constellation_cache(records: list):
    arrays = {}
    index = []
    for i, record in enumerate(records):
        key = f"c{i:02d}"
        arrays[f"{key}_tx"] = np.asarray(record["tx"], dtype=complex)
        arrays[f"{key}_rx"] = np.asarray(record["rx"], dtype=complex)
        index.append({
            "key": key,
            "sweep": record["sweep"],
            "x": record["x"],
            "linewidth_hz": record["linewidth_hz"],
            "objective": record["objective"],
            "family": record["family"],
            "strategy": record["strategy"],
        })
    np.savez_compressed(CONSTELLATION_NPZ, **arrays)
    CONSTELLATION_INDEX.write_text(json.dumps(index, indent=2), encoding="utf-8")


def main():
    train_symbols, train_bits = generate_qpsk_symbols(NUM_TRAIN, seed=SEED)
    test_symbols, test_bits = generate_qpsk_symbols(NUM_TEST, seed=SEED + 1)
    tx_train, t_train, _ = make_tx(train_symbols, 0.0)
    tx_test, t_test, _ = make_tx(test_symbols, 0.0)
    phase_bank = {}
    for i, linewidth in enumerate(LINEWIDTHS_HZ):
        phase_bank[float(linewidth)] = (
            phase_pair(len(tx_train), linewidth, SEED + 200 + i),
            phase_pair(len(tx_test), linewidth, SEED + 300 + i),
        )

    all_records = []
    constellation_records = []
    csv_path = REPORTS / "pnn_cdc_ber_evm_sweeps.csv"
    csv_path.write_text("", encoding="utf-8")
    for sweep, values, include_kerr in (
        ("distance", DISTANCES_KM, False),
        ("power", POWER_DBM, True),
    ):
        for li, linewidth in enumerate(LINEWIDTHS_HZ):
            phases = phase_bank[float(linewidth)]
            for pi, x in enumerate(values):
                if sweep == "power":
                    tx_train, t_train, _ = make_tx(train_symbols, float(x))
                    tx_test, t_test, _ = make_tx(test_symbols, float(x))
                else:
                    tx_train, t_train, _ = make_tx(train_symbols, 0.0)
                    tx_test, t_test, _ = make_tx(test_symbols, 0.0)
                records, _ = run_point(
                    sweep, float(x), float(linewidth),
                    tx_train, t_train, tx_test, t_test,
                    train_symbols, train_bits, test_symbols, test_bits,
                    phases[0], phases[1], include_kerr,
                    point_index=(0 if sweep == "distance" else MAX_DISTANCE_KM) + pi * 10 + li,
                    constellation_records=constellation_records,
                )
                all_records.extend(records)
                pd.DataFrame(records).to_csv(
                    csv_path, mode="a", index=False,
                    header=csv_path.stat().st_size == 0,
                )
                print(f"Completed {sweep}={x:g}, linewidth={linewidth/1e3:g} kHz",
                      flush=True)
    frame = pd.DataFrame(all_records)
    frame.to_csv(csv_path, index=False)
    summary_path = REPORTS / "pnn_cdc_ber_evm_sweeps_metadata.json"
    metadata = {
        "symbols_train": NUM_TRAIN, "symbols_test": NUM_TEST,
        "symbol_rate_baud": SYMBOL_RATE, "samples_per_symbol": SPS,
        "distances_km": DISTANCES_KM.tolist(), "powers_dbm": POWER_DBM.tolist(),
        "linewidths_hz": list(LINEWIDTHS_HZ), "phase_correlation_rho": RHO,
        "fiber_regime": {"distance": "linear", "power": "SSFM with Kerr"},
        "pnn_taps": PNN_TAPS, "pnn_delay_ps": PNN_DELAY_PS,
        "pso_swarm": PSO_SWARM, "pso_iterations": PSO_ITERATIONS,
        "joint_cdc_regularization_relative": REG_REL,
        "phase_correction": "constant phase estimated from known training symbols and applied to the independent test frame for BER; no time-varying CPE/BPS",
        "plot_ber_floor": PLOT_FLOOR,
        "note": "Joint CDC inverts linear fiber dispersion and the PNN field response; it does not invert Kerr nonlinearity.",
    }
    summary_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for sweep in ("distance", "power"):
        plot_curves(frame, sweep)
    save_constellation_figures(constellation_records)
    save_constellation_cache(constellation_records)
    print(f"CSV results: {csv_path}")
    print(f"Metadata: {summary_path}")


if __name__ == "__main__":
    main()
