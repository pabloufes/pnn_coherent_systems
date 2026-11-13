"""Recreate PNN/CDC BER, EVM, and constellation figures from saved results.

This script never runs PSO. It reads the sweep CSV and metadata, then loads
or reconstructs only the requested constellation points using the saved PNN
weights. The small constellation reconstruction is cached as NPZ/JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MAX_DISTANCE_KM = 200.0
REPORTS = ROOT / "artifacts" / str(MAX_DISTANCE_KM) / "reports"
FIGURES = ROOT / "artifacts" / str(MAX_DISTANCE_KM) / "figures"
CSV_PATH = REPORTS / "pnn_cdc_ber_evm_sweeps.csv"
METADATA_PATH = REPORTS / "pnn_cdc_ber_evm_sweeps_metadata.json"
CONSTELLATION_PATH = REPORTS / "pnn_cdc_constellation_points.npz"
CONSTELLATION_INDEX_PATH = REPORTS / "pnn_cdc_constellation_points.json"

OBJECTIVES = ("BER", "EVM")
FAMILIES = ("conventional", "joint")
LINEWIDTHS_HZ = (0.0, 100e3, 1e6)
STRATEGIES = (
    "No PNN / no CDC",
    "No PNN / CDC",
    "PNN / no CDC",
    "PNN + conventional CDC (joint training)",
    "PNN + joint CDC (joint training)",
)
COMMON_STRATEGIES = STRATEGIES[:3]


def read_inputs():
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"Sweep results not found: {CSV_PATH}")
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(f"Sweep metadata not found: {METADATA_PATH}")
    frame = pd.read_csv(CSV_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    required = {
        "sweep", "distance_km", "launch_power_dbm", "linewidth_hz",
        "training_objective", "equalizer_family", "strategy", "ber",
        "evm_percent", "amplitude_weights", "phase_weights",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Results CSV is missing columns: {sorted(missing)}")
    return frame, metadata


def recreate_curve_figures(frame: pd.DataFrame, metadata: dict):
    floor = float(metadata["plot_ber_floor"])
    linewidths = tuple(float(v) for v in metadata["linewidths_hz"])
    for sweep in ("distance", "power"):
        x_col = "distance_km" if sweep == "distance" else "launch_power_dbm"
        x_label = "Fiber distance (km)" if sweep == "distance" else "Launch power (dBm)"
        for objective in OBJECTIVES:
            for family in FAMILIES:
                subset = frame[(frame.sweep == sweep) &
                               (frame.training_objective == objective) &
                               (frame.equalizer_family == family)]
                if subset.empty:
                    raise ValueError(f"No data for {sweep}/{objective}/{family}")
                pnn_strategy = STRATEGIES[3] if family == "conventional" else STRATEGIES[4]
                plotted = COMMON_STRATEGIES + (pnn_strategy,)
                for metric, ylabel, logarithmic in (
                    ("ber", "BER", True),
                    ("evm_percent", "RMS EVM (%)", False),
                ):
                    metric_tag = "ber" if metric == "ber" else "evm"
                    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
                    for ax, linewidth in zip(axes, linewidths):
                        group = subset[subset.linewidth_hz == linewidth]
                        for strategy in plotted:
                            curve = group[group.strategy == strategy].sort_values(x_col)
                            y = curve[metric].to_numpy(dtype=float)
                            if logarithmic:
                                y = np.maximum(y, floor)
                                ax.semilogy(curve[x_col], y, marker="o", ms=3,
                                            label=strategy)
                            else:
                                ax.plot(curve[x_col], y, marker="o", ms=3,
                                        label=strategy)
                        ax.set_title(f"TX/LO linewidth: {linewidth / 1e3:g} kHz")
                        ax.set_xlabel(x_label)
                        ax.grid(True, which="both" if logarithmic else "major", alpha=.35)
                    axes[0].set_ylabel(ylabel)
                    axes[-1].legend(fontsize=8, loc="best")
                    regime = "linear fiber" if sweep == "distance" else "SSFM with Kerr"
                    fig.suptitle(
                        f"{ylabel} vs {x_label.lower()} — {objective} objective — "
                        f"{family} CDC — {regime}"
                    )
                    fig.tight_layout()
                    output = FIGURES / f"{sweep}_{metric_tag}_{objective.lower()}_{family}_cdc.png"
                    fig.savefig(output, dpi=220, bbox_inches="tight")
                    plt.close(fig)


def _load_analysis_helpers():
    # Import definitions/config only; the analysis module is protected by main().
    from scripts import generate_pnn_cdc_analysis as analysis
    return analysis


def reconstruct_constellation_cache(frame: pd.DataFrame, metadata: dict):
    """Rebuild only the two requested test-frame constellation locations.

    The saved PNN tap weights are reused. No optimization is performed.
    """
    analysis = _load_analysis_helpers()
    train_symbols, train_bits = analysis.generate_qpsk_symbols(
        analysis.NUM_TRAIN, seed=analysis.SEED
    )
    test_symbols, test_bits = analysis.generate_qpsk_symbols(
        analysis.NUM_TEST, seed=analysis.SEED + 1
    )
    records = []
    arrays = {}

    for sweep, x_value, kerr in (("distance", 300.0, False), ("power", 0.0, True)):
        distance_km = x_value if sweep == "distance" else MAX_DISTANCE_KM
        tx_train, t_train, _ = analysis.make_tx(train_symbols, 0.0)
        tx_test, t_test, _ = analysis.make_tx(test_symbols, 0.0)
        for linewidth_index, linewidth in enumerate(
            float(v) for v in metadata["linewidths_hz"]
        ):
            (phi_tx_train, phi_lo_train), (phi_tx_test, phi_lo_test) = (
                analysis.phase_pair(len(tx_train), linewidth,
                                    analysis.SEED + 200 + linewidth_index),
                analysis.phase_pair(len(tx_test), linewidth,
                                    analysis.SEED + 300 + linewidth_index),
            )
            train_field = analysis.propagate_spans(
                tx_train * np.exp(1j * phi_tx_train), t_train, distance_km, kerr
            )
            test_field = analysis.propagate_spans(
                tx_test * np.exp(1j * phi_tx_test), t_test, distance_km, kerr
            )

            fixed_models = {
                "No PNN / no CDC": (None, "none"),
                "No PNN / CDC": (None, "CDC"),
            }
            for objective in OBJECTIVES:
                for family in FAMILIES:
                    pnn_strategy = STRATEGIES[3] if family == "conventional" else STRATEGIES[4]
                    model_specs = dict(fixed_models)
                    model_specs["PNN / no CDC"] = ("PNN / no CDC", "none")
                    model_specs[pnn_strategy] = (
                        pnn_strategy, "CDC" if family == "conventional" else "joint_CDC"
                    )

                    for strategy, (saved_strategy, equalizer) in model_specs.items():
                        if saved_strategy is None:
                            model = None
                        else:
                            rows = frame[(frame.sweep == sweep) &
                                         (frame.x == x_value) &
                                         (frame.linewidth_hz == linewidth) &
                                         (frame.training_objective == objective) &
                                         (frame.strategy == saved_strategy)]
                            if saved_strategy == "PNN / no CDC":
                                rows = rows[rows.equalizer_family == family]
                            else:
                                rows = rows[rows.equalizer_family == family]
                            if rows.empty:
                                raise ValueError(
                                    f"No saved PNN weights for {sweep}/{objective}/"
                                    f"{family}/{saved_strategy}/{linewidth}"
                                )
                            row = rows.iloc[0]
                            amplitudes = np.asarray(json.loads(row.amplitude_weights), dtype=float)
                            phases = np.asarray(json.loads(row.phase_weights), dtype=float)
                            model = analysis.PhotonicNeuralNetwork(
                                num_channels=int(metadata["pnn_taps"]),
                                base_delay_ps=float(metadata["pnn_delay_ps"]),
                                normalize_power=True, responsivity=1.0,
                            )
                            model.set_weights(amplitudes, phases)

                        train_ref = analysis.evaluate(
                            train_field, phi_lo_train, train_symbols, train_bits,
                            model, equalizer, distance_km,
                            estimate_phase_from_known_symbols=True,
                        )
                        test_result = analysis.evaluate(
                            test_field, phi_lo_test, test_symbols, test_bits,
                            model, equalizer, distance_km,
                            phase_offset=train_ref["phase_offset_rad"],
                        )
                        key = f"c{len(records):02d}"
                        arrays[f"{key}_tx"] = test_symbols[:len(test_result["symbols"])]
                        arrays[f"{key}_rx"] = test_result["symbols"]
                        records.append({
                            "key": key, "sweep": sweep, "x": x_value,
                            "linewidth_hz": linewidth, "objective": objective,
                            "family": family, "strategy": strategy,
                        })

    np.savez_compressed(CONSTELLATION_PATH, **arrays)
    CONSTELLATION_INDEX_PATH.write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )


def recreate_constellation_figures():
    if not CONSTELLATION_PATH.is_file() or not CONSTELLATION_INDEX_PATH.is_file():
        raise FileNotFoundError("Constellation cache is missing after reconstruction.")
    records = json.loads(CONSTELLATION_INDEX_PATH.read_text(encoding="utf-8"))
    with np.load(CONSTELLATION_PATH, allow_pickle=False) as cache:
        for sweep, objective, family in (
            (s, o, f) for s in ("distance", "power")
            for o in OBJECTIVES for f in FAMILIES
        ):
            selected = [r for r in records if r["sweep"] == sweep and
                        r["objective"] == objective and r["family"] == family]
            if not selected:
                raise ValueError(f"Constellation samples missing: {sweep}/{objective}/{family}")
            pnn_strategy = STRATEGIES[3] if family == "conventional" else STRATEGIES[4]
            columns = COMMON_STRATEGIES + (pnn_strategy,)
            fig, axes = plt.subplots(3, 4, figsize=(16, 11))
            for row, linewidth in enumerate(LINEWIDTHS_HZ):
                for col, strategy in enumerate(columns):
                    ax = axes[row, col]
                    item = next((r for r in selected
                                 if r["linewidth_hz"] == linewidth and
                                 r["strategy"] == strategy), None)
                    if item is None:
                        ax.set_axis_off()
                        continue
                    tx = cache[f"{item['key']}_tx"]
                    rx = cache[f"{item['key']}_rx"]
                    tx = tx / (np.sqrt(np.mean(np.abs(tx) ** 2)) + 1e-30)
                    rx = rx / (np.sqrt(np.mean(np.abs(rx) ** 2)) + 1e-30)
                    ax.scatter(tx.real, tx.imag, s=8, marker="x", alpha=.45,
                               color="black", label="TX reference")
                    ax.scatter(rx.real, rx.imag, s=7, alpha=.35,
                               color="tab:blue", label="RX")
                    ax.set_title(f"{linewidth / 1e3:g} kHz — {strategy}", fontsize=8)
                    ax.set_xlabel("I (normalized)"); ax.set_ylabel("Q (normalized)")
                    ax.grid(True, alpha=.3); ax.set_aspect("equal", adjustable="datalim")
                    if row == 0 and col == 0:
                        ax.legend(fontsize=7, loc="best")
            point = "300 km" if sweep == "distance" else "0 dBm"
            regime = "linear fiber" if sweep == "distance" else "SSFM with Kerr"
            fig.suptitle(f"Constellations at {point} — {objective} training — "
                         f"{family} CDC — {regime}")
            fig.tight_layout()
            fig.savefig(FIGURES / f"{sweep}_constellations_{objective.lower()}_"
                        f"{family}_cdc.png", dpi=220, bbox_inches="tight")
            plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    frame, metadata = read_inputs()
    recreate_curve_figures(frame, metadata)
    if not CONSTELLATION_PATH.is_file() or not CONSTELLATION_INDEX_PATH.is_file():
        reconstruct_constellation_cache(frame, metadata)
    recreate_constellation_figures()
    print(f"Recreated figures from {CSV_PATH}")
    print(f"Constellation cache: {CONSTELLATION_PATH}")


if __name__ == "__main__":
    main()
