# Exploration of PNN in Coherent QPSK Transmission with Phase Correlation

This notebook establishes a physical baseline and compares the effect of a PNN in a coherent QPSK system. BER is the primary figure of merit. The notebook evaluates (1) chromatic dispersion without nonlinearity as a function of distance, and (2) chromatic dispersion + Kerr effect as a function of launch power for a 1000 km link. For each analysis and linewidth, a PNN is trained separately and evaluated with fixed weights.

The model uses an IQ-MZM QPSK transmitter, a seventh-order Gaussian electrical filter, and three equal TX/LO linewidths: 0, 100 kHz, and 1 MHz. TX and LO phases are independent realizations. There is no digital chromatic dispersion compensation or carrier phase recovery in this baseline.


The correlation coefficient `PHASE_CORRELATION_TX_RX` is configurable between 0 (independent processes) and 1 (ideal common phase).


```python
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import json

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / 'src').is_dir():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dsp.transmitter import (
    ElectricalFilterParams,
    IQMZMParams,
    generate_qpsk_symbols,
    modulate_optical_signal,
)
from src.dsp.fiber import FiberParams, fibra_monomodo
from src.utils.edfa import EDFAIdeal
from src.dsp.noise_models import generate_correlated_phases
from src.dsp.receiver import coherent_detection, cd_compensation, sample_to_symbols, demap_qpsk
from src.utils.metrics import calculate_ber

FIG_DIR = PROJECT_ROOT / 'artifacts' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)
print(f'Project root: {PROJECT_ROOT}')
from src.utils.pnn_model import PhotonicNeuralNetwork
from scripts.optimize_pnn_pso import particle_swarm_optimize

```

    Project root: D:\gits\PHOTONICS_2026\Codigos\python\pnn_coerente
    

## 1. Parameters and Link Configuration

`SPAN_LENGTH_KM` is the single fiber span length used across the entire link. The total link distance must be an integer multiple of it; the notebook halts execution if this condition is not met, preventing fractional terminal spans. To compare spans of 50, 100, or 200 km, adjust this parameter.

Inline EDFAs exactly compensate the loss of each span. The terminal EDFA has fixed gain, and a variable attenuator adjusts the receiver input power to `RX_TARGET_DBM`. At this stage, EDFAs are ideal: they do not include ASE noise, noise figure, or saturation.



```python
NUM_SYMBOLS = 2**12
SPS = 8
SYMBOL_RATE = 10e9
WAVELENGTH = 1550e-9

# Span configuration: every span has this exact length.
SPAN_LENGTH_KM = 50.0  # experimente 50.0 ou 200.0 km
DISTANCE_SWEEP_MAX_KM = 1200.0
DISTANCE_NONLINEAR_KM = 1000.0
NUM_STEPS_PER_SPAN = 5

# In-line EDFA compensates alpha_db * SPAN_LENGTH_KM in each span.
# The EDFA and attenuator stage at the receiver input sets this power.
RX_EDFA_GAIN_DB = 20.0
RX_TARGET_DBM = 0.0

TX_POWER_DISPERSION_DBM = 0.0
POWER_SWEEP_DBM = np.arange(-20.0, 20.1, 2.0)
LINEWIDTHS_HZ = np.array([0.0, 100e3, 1e6])

FIBER_PARAMS = FiberParams(
    alpha_db=0.2,
    d_1550=16.0,
    slope_s=0.09,
    n2=3.2e-20,
    a_eff=70e-12,
)
IQ_PARAMS = IQMZMParams(electrical_response=False)
EDFA_MODEL = EDFAIdeal  # replacement point for future analytical model
FILTER_PARAMS = ElectricalFilterParams(
    tipo='cosseno_levantado',
    largura_banda_hz=2*SYMBOL_RATE,
    ordem=7,
)
MODULATOR = 'iq_mzm'


# Correlation between transmitter phase and receiver local oscillator phase.
# rho=0: independent lasers; rho=1: ideal common phase.
PHASE_CORRELATION_TX_RX = 0.99
if not np.isfinite(PHASE_CORRELATION_TX_RX) or not 0.0 <= PHASE_CORRELATION_TX_RX <= 1.0:
    raise ValueError('PHASE_CORRELATION_TX_RX must be in the range [0, 1].')
CORRELATION_TAG = f'rho_{PHASE_CORRELATION_TX_RX:.2f}'.replace('.', 'p')
FIG_DIR = PROJECT_ROOT / 'artifacts' / 'figures' / f'coerente_{CORRELATION_TAG}'
FIG_DIR.mkdir(parents=True, exist_ok=True)


# PNN: each model is trained for its own scenario and linewidth.
PNN_NUM_CHANNELS = 8
PNN_BASE_DELAY_PS = 25.0
PNN_NORMALIZE_POWER = True
PNN_TRAIN_SYMBOLS = 2**10
PNN_TRAIN_SEED = 20261101
PNN_SWARM_SIZE = 60
PNN_ITERATIONS = 100
PNN_NONLINEAR_TRAIN_POWER_DBM = 0.0
PNN_POWER_SWEEP_SWARM_SIZE = 60
PNN_POWER_SWEEP_ITERATIONS = 100
PNN_DISTANCE_SWEEP_SWARM_SIZE = 60
PNN_DISTANCE_SWEEP_ITERATIONS = 100

if SPAN_LENGTH_KM <= 0:
    raise ValueError('SPAN_LENGTH_KM must be positive.')
if DISTANCE_SWEEP_MAX_KM % SPAN_LENGTH_KM != 0:
    raise ValueError('DISTANCE_SWEEP_MAX_KM must be a multiple of SPAN_LENGTH_KM.')
if DISTANCE_NONLINEAR_KM % SPAN_LENGTH_KM != 0:
    raise ValueError('DISTANCE_NONLINEAR_KM must be a multiple of SPAN_LENGTH_KM.')
if RX_EDFA_GAIN_DB < RX_TARGET_DBM - min(TX_POWER_DISPERSION_DBM, POWER_SWEEP_DBM.min()):
    raise ValueError('The terminal EDFA gain is insufficient to reach the target power.')

DISTANCES_KM = np.arange(
    SPAN_LENGTH_KM, DISTANCE_SWEEP_MAX_KM + SPAN_LENGTH_KM, SPAN_LENGTH_KM
)
print(f'Span: {SPAN_LENGTH_KM:g} km | EDFA/fiber at 1000 km: {DISTANCE_NONLINEAR_KM / SPAN_LENGTH_KM:g}')
print(f'Target power at receiver: {RX_TARGET_DBM:.1f} dBm')
print(f'TX = LO linewidths: {LINEWIDTHS_HZ} Hz')
print(f'TX-RX phase correlation: rho = {PHASE_CORRELATION_TX_RX:.3f}')

```

    Span: 50 km | EDFA/fiber at 1000 km: 20
    Target power at receiver: 0.0 dBm
    TX = LO linewidths: [      0.  100000. 1000000.] Hz
    TX-RX phase correlation: rho = 0.990
    

## 2. Propagation and Power Functions

Each span propagates the field through the fiber and applies an ideal EDFA with gain equal to the loss of that span. The terminal stage applies fixed gain and computes the required attenuation to achieve the target receiver power.



```python
def mean_power_dbm(field):
    """Mean optical envelope power in dBm."""
    power_w = np.mean(np.abs(field) ** 2)
    return 10.0 * np.log10(power_w / 1e-3)


def propagate_spanned_link(
    field, t, total_length_km, include_nonlinearity,
    span_length_km=SPAN_LENGTH_KM, fiber_params=FIBER_PARAMS,
    num_steps_per_span=NUM_STEPS_PER_SPAN,
):
    """Propaga spans uniformes, cada um seguido por EDFA de ganho compensador."""
    n_spans_float = total_length_km / span_length_km
    n_spans = int(round(n_spans_float))
    if n_spans <= 0 or not np.isclose(n_spans_float, n_spans, rtol=0.0, atol=1e-9):
        raise ValueError(
            f'{total_length_km:g} km não é múltiplo inteiro do span de '
            f'{span_length_km:g} km.'
        )

    regime = 1 if include_nonlinearity else 0
    field_out = np.asarray(field, dtype=complex).copy()
    span_loss_db = fiber_params.alpha_db * span_length_km
    inline_edfa = EDFA_MODEL(gain_db=span_loss_db)

    for _ in range(n_spans):
        field_out, _, _ = fibra_monomodo(
            t,
            field_out,
            wavelength=WAVELENGTH,
            t0=1.0 / SYMBOL_RATE,
            length_km=span_length_km,
            regime=regime,
            params=fiber_params,
            num_steps=num_steps_per_span,
        )
        # The ideal EDFA restores the passive loss of this span.
        field_out = inline_edfa.amplify(field_out)

    return field_out


def receiver_edfa_and_attenuator(field, target_dbm=RX_TARGET_DBM, edfa_gain_db=RX_EDFA_GAIN_DB):
    """Applies fixed terminal EDFA and ideal VOA to set received power."""
    receiver_edfa = EDFA_MODEL(gain_db=edfa_gain_db)
    amplified = receiver_edfa.amplify(field)
    power_after_edfa_dbm = mean_power_dbm(amplified)
    attenuation_db = power_after_edfa_dbm - target_dbm
    if attenuation_db < -1e-9:
        raise ValueError(
            'The terminal EDFA gain is insufficient to reach target power; '
            'increase RX_EDFA_GAIN_DB.'
        )
    attenuation_db = max(0.0, attenuation_db)
    received = amplified * 10.0 ** (-attenuation_db / 20.0)
    return received, attenuation_db, power_after_edfa_dbm

```

## 3. Reference Signal and Phase Noise

The same QPSK frame is reused across sweeps. TX and LO phases are Wiener processes with equal linewidth; their correlation is set by `PHASE_CORRELATION_TX_RX`. The correlated realizations are kept fixed across distances and powers to facilitate direct comparison across curve points.



```python
symbols, bits = generate_qpsk_symbols(NUM_SYMBOLS, seed=42)

# 0 dBm reference; the power sweep linearly scales the field.
u_tx_0dbm, t, dt = modulate_optical_signal(
    symbols,
    sps=SPS,
    launch_power_dbm=0.0,
    symbol_rate=SYMBOL_RATE,
    modulator=MODULATOR,
    iq_params=IQ_PARAMS,
    filter_params=FILTER_PARAMS,
    plot_filter=True,
    n_plot=20,
)

phase_noise = {}
for idx, linewidth_hz in enumerate(LINEWIDTHS_HZ):
    phase_tx, phase_lo = generate_correlated_phases(
        len(t), dt, float(linewidth_hz),
        correlation_rho=PHASE_CORRELATION_TX_RX,
        seed=1000 + idx,
    )
    phase_noise[float(linewidth_hz)] = (phase_tx, phase_lo)

print(f'Symbols: {NUM_SYMBOLS} | Samples/symbol: {SPS} | dt: {dt:.3e} s')
print(f'Mean reference power: {mean_power_dbm(u_tx_0dbm):.2f} dBm')
print(f'Correlation used in TX-RX phases: {PHASE_CORRELATION_TX_RX:.3f}')

```


    
![png](exploracao_pnn_dispersao_edfa_correlacao_fase_files/exploracao_pnn_dispersao_edfa_correlacao_fase_7_0.png)
    


    Symbols: 4096 | Samples/symbol: 8 | dt: 1.250e-11 s
    Mean reference power: 0.00 dBm
    Correlation used in TX-RX phases: 0.990
    

## 4. Analysis 1 — BER vs. Distance: Chromatic Dispersion

The fiber retains physical attenuation and receives ideal EDFAs between spans, but propagation uses the linear regime (`regime=0`), without Kerr nonlinearity. The curve compares the receiver without CDC against digital chromatic dispersion compensation (CDC) DSP applied after coherent detection and before symbol sampling. Linewidths continue to be swept; therefore, the 0 Hz curve isolates dispersion, while 100 kHz and 1 MHz also include independent phase noise.



```python
ber_vs_distance_no_cdc = {float(lw): [] for lw in LINEWIDTHS_HZ}
ber_vs_distance_with_cdc = {float(lw): [] for lw in LINEWIDTHS_HZ}
rx_power_distance = {float(lw): [] for lw in LINEWIDTHS_HZ}

for linewidth_hz in LINEWIDTHS_HZ:
    phase_tx, phase_lo = phase_noise[float(linewidth_hz)]
    for distance_km in DISTANCES_KM:
        field_tx = u_tx_0dbm * np.exp(1j * phase_tx)
        field_link = propagate_spanned_link(
            field_tx, t, distance_km, include_nonlinearity=False
        )
        field_rx, atten_db, _ = receiver_edfa_and_attenuator(field_link)
        rx_power_distance[float(linewidth_hz)].append(mean_power_dbm(field_rx))

        r_bb = coherent_detection(field_rx, phase_lo=phase_lo)
        rx_symbols = sample_to_symbols(r_bb, sps=SPS)
        ber_no_cdc = calculate_ber(bits, demap_qpsk(rx_symbols))
        ber_vs_distance_no_cdc[float(linewidth_hz)].append(ber_no_cdc)

        r_bb_cdc = cd_compensation(
            r_bb,
            dt=dt,
            length_km=distance_km,
            dispersion_d=FIBER_PARAMS.d_1550,
        )
        rx_symbols_cdc = sample_to_symbols(r_bb_cdc, sps=SPS)
        ber_with_cdc = calculate_ber(bits, demap_qpsk(rx_symbols_cdc))
        ber_vs_distance_with_cdc[float(linewidth_hz)].append(ber_with_cdc)

BER_PLOT_FLOOR = 1.0 / (2 * NUM_SYMBOLS)
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['tab:blue', 'tab:orange', 'tab:green']
for linewidth_hz, color in zip(LINEWIDTHS_HZ, colors):
    values_no = np.asarray(ber_vs_distance_no_cdc[float(linewidth_hz)])
    values_cdc = np.asarray(ber_vs_distance_with_cdc[float(linewidth_hz)])
    ax.semilogy(
        DISTANCES_KM,
        np.maximum(values_no, BER_PLOT_FLOOR),
        linestyle='--', marker='x', color=color,
        label=f'{linewidth_hz / 1e3:g} kHz, sem CDC',
    )
    ax.semilogy(
        DISTANCES_KM,
        np.maximum(values_cdc, BER_PLOT_FLOOR),
        linestyle='-', marker='o', color=color,
        label=f'{linewidth_hz / 1e3:g} kHz, com CDC',
    )
ax.set_title('BER versus distância — dispersão cromática, com e sem CDC')
ax.set_xlabel('Distância [km]')
ax.set_ylabel('BER')
ax.grid(True, which='both', linestyle='--', alpha=0.5)
ax.legend(title='Largura de linha TX = LO / DSP', ncol=2, fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / 'pnn_analise_1_ber_vs_distancia_com_sem_cdc.png', dpi=250)
plt.show()

rx_power_distance_values = [value for curve in rx_power_distance.values() for value in curve]
print(f'Potência no receptor (mín./máx.): {np.min(rx_power_distance_values):.4f} / {np.max(rx_power_distance_values):.4f} dBm')
print(f'Valores BER zero são exibidos no piso {BER_PLOT_FLOOR:.2e} no gráfico.')

```


    
![png](exploracao_pnn_dispersao_edfa_correlacao_fase_files/exploracao_pnn_dispersao_edfa_correlacao_fase_9_0.png)
    


    Received power (min/max): -0.0000 / 0.0000 dBm
    Zero BER values are displayed at the 1.22e-04 floor in the plot.
    

## 5. Analysis 2 — BER vs. Launch Power: Dispersion + Nonlinearity

The link is 1000 km long and uses the SSFM regime (`regime=1`), with dispersion and Kerr effect. Launch power ranges from −20 to +20 dBm. The EDFA in each span restores fiber loss, maintaining roughly the same power level at the beginning of each span. After the final span, the fixed-gain terminal EDFA and variable optical attenuator ensure 0 dBm at the receiver input. BER is compared with and without CDC; at the receiver, CDC follows coherent detection and precedes symbol sampling.

The notebook computes propagation for each combination of power and linewidth. Because TX and LO phases are independent and the TX phase is applied before fiber transmission, phase noise also interacts with dispersion and nonlinearity.



```python
ber_vs_power_no_cdc = {float(lw): [] for lw in LINEWIDTHS_HZ}
ber_vs_power_with_cdc = {float(lw): [] for lw in LINEWIDTHS_HZ}
rx_power_sweep = []
rx_stage_attenuation = []

for launch_power_dbm in POWER_SWEEP_DBM:
    # 0 dBm reference field scaled to desired launch power.
    tx_field = u_tx_0dbm * 10.0 ** (launch_power_dbm / 20.0)

    for linewidth_hz in LINEWIDTHS_HZ:
        phase_tx, phase_lo = phase_noise[float(linewidth_hz)]
        field_tx = tx_field * np.exp(1j * phase_tx)
        field_link = propagate_spanned_link(
            field_tx,
            t,
            DISTANCE_NONLINEAR_KM,
            include_nonlinearity=True,
        )
        field_rx, atten_db, power_after_rx_edfa_dbm = receiver_edfa_and_attenuator(field_link)
        rx_power_sweep.append(mean_power_dbm(field_rx))
        rx_stage_attenuation.append(atten_db)

        r_bb = coherent_detection(field_rx, phase_lo=phase_lo)
        rx_symbols = sample_to_symbols(r_bb, sps=SPS)
        ber_no_cdc = calculate_ber(bits, demap_qpsk(rx_symbols))
        ber_vs_power_no_cdc[float(linewidth_hz)].append(ber_no_cdc)

        r_bb_cdc = cd_compensation(
            r_bb,
            dt=dt,
            length_km=DISTANCE_NONLINEAR_KM,
            dispersion_d=FIBER_PARAMS.d_1550,
        )
        rx_symbols_cdc = sample_to_symbols(r_bb_cdc, sps=SPS)
        ber_with_cdc = calculate_ber(bits, demap_qpsk(rx_symbols_cdc))
        ber_vs_power_with_cdc[float(linewidth_hz)].append(ber_with_cdc)

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['tab:blue', 'tab:orange', 'tab:green']
for linewidth_hz, color in zip(LINEWIDTHS_HZ, colors):
    values_no = np.asarray(ber_vs_power_no_cdc[float(linewidth_hz)])
    values_cdc = np.asarray(ber_vs_power_with_cdc[float(linewidth_hz)])
    ax.semilogy(
        POWER_SWEEP_DBM,
        np.maximum(values_no, BER_PLOT_FLOOR),
        linestyle='--', marker='x', color=color,
        label=f'{linewidth_hz / 1e3:g} kHz, without CDC',
    )
    ax.semilogy(
        POWER_SWEEP_DBM,
        np.maximum(values_cdc, BER_PLOT_FLOOR),
        linestyle='-', marker='o', color=color,
        label=f'{linewidth_hz / 1e3:g} kHz, with CDC',
    )
ax.set_title(f'BER vs. Launch Power — {DISTANCE_NONLINEAR_KM:g} km, Dispersion + Kerr, with and without CDC')
ax.set_xlabel('Launch Power [dBm]')
ax.set_ylabel('BER')
ax.grid(True, which='both', linestyle='--', alpha=0.5)
ax.legend(title='Linewidth TX = LO / DSP', ncol=2, fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / 'pnn_analise_2_ber_vs_potencia_com_sem_cdc.png', dpi=250)
plt.show()

print(f'Measured power at receiver (min/max): {np.min(rx_power_sweep):.4f} / {np.max(rx_power_sweep):.4f} dBm')
print(f'Terminal VOA attenuation (min/max): {np.min(rx_stage_attenuation):.2f} / {np.max(rx_stage_attenuation):.2f} dB')
print(f'Terminal EDFA: fixed gain of {RX_EDFA_GAIN_DB:.1f} dB; receiver target: {RX_TARGET_DBM:.1f} dBm.')

```


    
![png](exploracao_pnn_dispersao_edfa_correlacao_fase_files/exploracao_pnn_dispersao_edfa_correlacao_fase_11_0.png)
    


    Measured power at receiver (min/max): -0.0000 / 0.0000 dBm
    Terminal VOA attenuation (min/max): 0.00 / 40.00 dB
    Terminal EDFA: fixed gain of 20.0 dB; receiver target: 0.0 dBm.
    

## 6. Independent PNN Training for Each Analysis

Six PNNs are trained: one for each of the two analyses and for each linewidth. In the distance analysis, training uses the maximum distance and launch power `TX_POWER_DISPERSION_DBM`; in the power analysis, it uses `DISTANCE_NONLINEAR_KM` and the 0 dBm reference launch power. Weights of each PNN are fixed during the evaluation of the corresponding curve.

Training uses a short QPSK sequence and dedicated phase noise realizations. Evaluation reuses the longer sequence and independent realizations from this baseline. The objective function is BER without CDC, so that the PSO trains the PNN to equalize the link; we also measure the PNN followed by CDC.



```python
# Short training sequence, separated from evaluation sequence (symbols, bits).
train_symbols, train_bits = generate_qpsk_symbols(
    PNN_TRAIN_SYMBOLS, seed=PNN_TRAIN_SEED
)
train_tx_0dbm, t_train, dt_train = modulate_optical_signal(
    train_symbols,
    sps=SPS,
    launch_power_dbm=0.0,
    symbol_rate=SYMBOL_RATE,
    modulator=MODULATOR,
    iq_params=IQ_PARAMS,
    filter_params=FILTER_PARAMS,
    plot_filter=False,
    n_plot=0,
)

pnn_model = PhotonicNeuralNetwork(
    num_channels=PNN_NUM_CHANNELS,
    base_delay_ps=PNN_BASE_DELAY_PS,
    normalize_power=PNN_NORMALIZE_POWER,
    responsivity=1.0,
)
pnn_bounds = [(0.0, 1.0)] * PNN_NUM_CHANNELS + [(-np.pi, np.pi)] * PNN_NUM_CHANNELS
pnn_training_results = {}
pnn_joint_cdc_results = {}
joint_training_histories = {'dispersion': {}, 'nonlinear': {}}
training_histories = {'dispersion': {}, 'nonlinear': {}}

for scenario, distance_km, include_nonlinearity, launch_power_dbm in [
    ('dispersion', DISTANCE_SWEEP_MAX_KM, False, TX_POWER_DISPERSION_DBM),
    ('nonlinear', DISTANCE_NONLINEAR_KM, True, PNN_NONLINEAR_TRAIN_POWER_DBM),
]:
    train_field_at_power = train_tx_0dbm * 10.0 ** (launch_power_dbm / 20.0)
    for linewidth_index, linewidth_hz in enumerate(LINEWIDTHS_HZ):
        linewidth_key = float(linewidth_hz)
        phase_tx_train, phase_lo_train = generate_correlated_phases(
            len(t_train), dt_train, linewidth_key,
            correlation_rho=PHASE_CORRELATION_TX_RX,
            seed=PNN_TRAIN_SEED + 100 + 2 * linewidth_index,
        )
        field_train_tx = train_field_at_power * np.exp(1j * phase_tx_train)
        field_train_link = propagate_spanned_link(
            field_train_tx,
            t_train,
            distance_km,
            include_nonlinearity=include_nonlinearity,
        )
        field_train_rx, _, _ = receiver_edfa_and_attenuator(field_train_link)

        def training_objective(position):
            pnn_model.set_weights(
                position[:PNN_NUM_CHANNELS],
                position[PNN_NUM_CHANNELS:],
            )
            _, field_pnn = pnn_model.forward(
                field_train_rx, dt_s=dt_train, return_optical_field=True
            )
            r_train = coherent_detection(field_pnn, phase_lo=phase_lo_train)
            rx_train_symbols = sample_to_symbols(r_train, sps=SPS)
            return calculate_ber(train_bits, demap_qpsk(rx_train_symbols))

        pso = particle_swarm_optimize(
            training_objective,
            pnn_bounds,
            swarm_size=PNN_SWARM_SIZE,
            max_iterations=PNN_ITERATIONS,
            periodic_dimensions=range(PNN_NUM_CHANNELS, 2 * PNN_NUM_CHANNELS),
            seed=PNN_TRAIN_SEED + 1000 * (scenario == 'nonlinear') + linewidth_index,
        )
        pnn_model.set_weights(pso['amplitude_weights'], pso['phase_weights'])
        _, best_field_pnn = pnn_model.forward(
            field_train_rx, dt_s=dt_train, return_optical_field=True
        )
        best_r_train = coherent_detection(best_field_pnn, phase_lo=phase_lo_train)
        best_train_ber = calculate_ber(
            train_bits,
            demap_qpsk(sample_to_symbols(best_r_train, sps=SPS)),
        )
        key = (scenario, linewidth_key)
        pnn_training_results[key] = {
            'scenario': scenario,
            'linewidth_hz': linewidth_key,
            'distance_km': float(distance_km),
            'include_nonlinearity': bool(include_nonlinearity),
            'training_power_dbm': float(launch_power_dbm),
            'amplitude_weights': pso['amplitude_weights'].copy(),
            'phase_weights': pso['phase_weights'].copy(),
            'best_training_ber': float(best_train_ber),
            'best_fitness_history': pso['best_fitness_history'].copy(),
            'evaluations': int(pso['evaluations']),
            'seed': int(PNN_TRAIN_SEED + 1000 * (scenario == 'nonlinear') + linewidth_index),
        }

        def joint_cdc_objective(position):
            pnn_model.set_weights(position[:PNN_NUM_CHANNELS], position[PNN_NUM_CHANNELS:])
            _, joint_field = pnn_model.forward(field_train_rx, dt_s=dt_train, return_optical_field=True)
            joint_r = coherent_detection(joint_field, phase_lo=phase_lo_train)
            joint_r = cd_compensation(joint_r, dt=dt_train, length_km=distance_km, dispersion_d=FIBER_PARAMS.d_1550)
            return calculate_ber(train_bits, demap_qpsk(sample_to_symbols(joint_r, sps=SPS)))

        joint_seed = PNN_TRAIN_SEED + 50000 + 1000 * (scenario == 'nonlinear') + linewidth_index
        joint_pso = particle_swarm_optimize(
            joint_cdc_objective, pnn_bounds,
            swarm_size=PNN_SWARM_SIZE,
            max_iterations=PNN_ITERATIONS,
            periodic_dimensions=range(PNN_NUM_CHANNELS, 2 * PNN_NUM_CHANNELS),
            seed=joint_seed,
        )
        pnn_joint_cdc_results[key] = {
            'scenario': scenario,
            'linewidth_hz': linewidth_key,
            'distance_km': float(distance_km),
            'include_nonlinearity': bool(include_nonlinearity),
            'training_power_dbm': float(launch_power_dbm),
            'amplitude_weights': joint_pso['amplitude_weights'].copy(),
            'phase_weights': joint_pso['phase_weights'].copy(),
            'best_training_ber': float(joint_pso['best_fitness']),
            'best_fitness_history': joint_pso['best_fitness_history'].copy(),
            'evaluations': int(joint_pso['evaluations']),
            'seed': int(joint_seed),
        }
        joint_training_histories[scenario][linewidth_key] = joint_pso['best_fitness_history'].copy()
        print(f"{scenario} + CDC, linewidth={linewidth_key/1e3:g} kHz: BER treino={joint_pso['best_fitness']:.6g}")
        training_histories[scenario][linewidth_key] = pso['best_fitness_history']
        print(
            f"{scenario}, linewidth={linewidth_key/1e3:g} kHz: "
            f"BER treino={best_train_ber:.6g}, avaliacoes={pso['evaluations']}"
        )

# Treinamento adicional independente para cada distancia (cada span da varredura).
# Cada distancia/linewidth recebe uma PNN para BER sem CDC e outra para BER apos PNN + CDC.
distance_sweep_training_results = {}
span_training_ber_no_cdc = {float(lw): [] for lw in LINEWIDTHS_HZ}
span_training_ber_with_cdc = {float(lw): [] for lw in LINEWIDTHS_HZ}
span_tx_base = train_tx_0dbm * 10.0 ** (TX_POWER_DISPERSION_DBM / 20.0)

for distance_index, span_distance_km in enumerate(DISTANCES_KM):
    for linewidth_index, linewidth_hz in enumerate(LINEWIDTHS_HZ):
        linewidth_key = float(linewidth_hz)
        phase_seed = PNN_TRAIN_SEED + 400000 + 2 * linewidth_index
        phase_tx_span, phase_lo_span = generate_correlated_phases(
            len(t_train), dt_train, linewidth_key,
            correlation_rho=PHASE_CORRELATION_TX_RX,
            seed=phase_seed,
        )
        train_field_span = span_tx_base * np.exp(1j * phase_tx_span)
        train_link_span = propagate_spanned_link(
            train_field_span, t_train, float(span_distance_km),
            include_nonlinearity=False,
        )
        train_rx_span, _, _ = receiver_edfa_and_attenuator(train_link_span)

        def span_pnn_objective(position):
            pnn_model.set_weights(position[:PNN_NUM_CHANNELS], position[PNN_NUM_CHANNELS:])
            _, e_span = pnn_model.forward(train_rx_span, dt_s=dt_train, return_optical_field=True)
            r_span = coherent_detection(e_span, phase_lo=phase_lo_span)
            return calculate_ber(train_bits, demap_qpsk(sample_to_symbols(r_span, sps=SPS)))

        span_seed = PNN_TRAIN_SEED + 400000 + 1000 * distance_index + linewidth_index
        pso_span = particle_swarm_optimize(
            span_pnn_objective, pnn_bounds,
            swarm_size=PNN_DISTANCE_SWEEP_SWARM_SIZE,
            max_iterations=PNN_DISTANCE_SWEEP_ITERATIONS,
            periodic_dimensions=range(PNN_NUM_CHANNELS, 2 * PNN_NUM_CHANNELS),
            seed=span_seed,
        )

        def span_pnn_cdc_objective(position):
            pnn_model.set_weights(position[:PNN_NUM_CHANNELS], position[PNN_NUM_CHANNELS:])
            _, e_span = pnn_model.forward(train_rx_span, dt_s=dt_train, return_optical_field=True)
            r_span = coherent_detection(e_span, phase_lo=phase_lo_span)
            r_span = cd_compensation(
                r_span, dt=dt_train, length_km=float(span_distance_km),
                dispersion_d=FIBER_PARAMS.d_1550,
            )
            return calculate_ber(train_bits, demap_qpsk(sample_to_symbols(r_span, sps=SPS)))

        span_joint_seed = span_seed + 50000
        pso_span_joint = particle_swarm_optimize(
            span_pnn_cdc_objective, pnn_bounds,
            swarm_size=PNN_DISTANCE_SWEEP_SWARM_SIZE,
            max_iterations=PNN_DISTANCE_SWEEP_ITERATIONS,
            periodic_dimensions=range(PNN_NUM_CHANNELS, 2 * PNN_NUM_CHANNELS),
            seed=span_joint_seed,
        )
        distance_sweep_training_results[(float(span_distance_km), linewidth_key)] = {
            'pnn_only': {
                'amplitude_weights': pso_span['amplitude_weights'].copy(),
                'phase_weights': pso_span['phase_weights'].copy(),
                'best_training_ber': float(pso_span['best_fitness']),
                'best_fitness_history': pso_span['best_fitness_history'].copy(),
                'evaluations': int(pso_span['evaluations']),
                'seed': int(span_seed),
            },
            'pnn_plus_cdc': {
                'amplitude_weights': pso_span_joint['amplitude_weights'].copy(),
                'phase_weights': pso_span_joint['phase_weights'].copy(),
                'best_training_ber': float(pso_span_joint['best_fitness']),
                'best_fitness_history': pso_span_joint['best_fitness_history'].copy(),
                'evaluations': int(pso_span_joint['evaluations']),
                'seed': int(span_joint_seed),
            },
        }
        span_training_ber_no_cdc[linewidth_key].append(float(pso_span['best_fitness']))
        span_training_ber_with_cdc[linewidth_key].append(float(pso_span_joint['best_fitness']))
    print(f'Treinamento independente concluido para {span_distance_km:g} km.')

# BER de treino obtida em cada distancia, para acompanhar as PNNs especializadas.
fig, axes = plt.subplots(1, len(LINEWIDTHS_HZ), figsize=(15, 4.8), sharey=True)
for ax, linewidth_hz in zip(axes, LINEWIDTHS_HZ):
    key = float(linewidth_hz)
    ax.semilogy(
        DISTANCES_KM,
        np.maximum(span_training_ber_no_cdc[key], 1.0 / (2 * len(train_bits))),
        'o-', label='PNN sem CDC',
    )
    ax.semilogy(
        DISTANCES_KM,
        np.maximum(span_training_ber_with_cdc[key], 1.0 / (2 * len(train_bits))),
        's--', label='PNN + CDC',
    )
    ax.set_title(f'{linewidth_hz/1e3:g} kHz')
    ax.set_xlabel('Distancia de treino [km]')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend()
axes[0].set_ylabel('BER de treino')
fig.suptitle('BER de treino das PNNs especializadas por distancia')
fig.tight_layout()
fig.savefig(FIG_DIR / 'pnn_coerente_ber_treino_por_distancia.png', dpi=250)
plt.show()

# Treino adicional: uma PNN independente para cada potencia e linewidth da varredura.
# O ponto de 0 dBm reutiliza o treino nominal acima, evitando duplicacao.
pnn_power_training_results = {}
pnn_power_joint_cdc_results = {}
for power_index, launch_power_dbm in enumerate(POWER_SWEEP_DBM):
    for linewidth_index, linewidth_hz in enumerate(LINEWIDTHS_HZ):
        linewidth_key = float(linewidth_hz)
        power_key = float(launch_power_dbm)
        result_key = (power_key, linewidth_key)
        if np.isclose(power_key, PNN_NONLINEAR_TRAIN_POWER_DBM):
            pnn_power_training_results[result_key] = pnn_training_results[
                ('nonlinear', linewidth_key)
            ]
            pnn_power_joint_cdc_results[result_key] = pnn_joint_cdc_results[
                ('nonlinear', linewidth_key)
            ]
            continue

        independent_tx_seed = PNN_TRAIN_SEED + 200000 + 20 * power_index + 2 * linewidth_index
        phase_tx_power_train, phase_lo_power_train = generate_correlated_phases(
            len(t_train), dt_train, linewidth_key,
            correlation_rho=PHASE_CORRELATION_TX_RX,
            seed=independent_tx_seed,
        )
        tx_power_train = train_tx_0dbm * 10.0 ** (power_key / 20.0)
        tx_power_train *= np.exp(1j * phase_tx_power_train)
        link_power_train = propagate_spanned_link(
            tx_power_train,
            t_train,
            DISTANCE_NONLINEAR_KM,
            include_nonlinearity=True,
        )
        rx_power_train, _, _ = receiver_edfa_and_attenuator(link_power_train)

        def power_training_objective(position):
            pnn_model.set_weights(
                position[:PNN_NUM_CHANNELS], position[PNN_NUM_CHANNELS:]
            )
            _, field_power_pnn = pnn_model.forward(
                rx_power_train, dt_s=dt_train, return_optical_field=True
            )
            r_power_train = coherent_detection(
                field_power_pnn, phase_lo=phase_lo_power_train
            )
            return calculate_ber(
                train_bits,
                demap_qpsk(sample_to_symbols(r_power_train, sps=SPS)),
            )

        power_seed = PNN_TRAIN_SEED + 300000 + 100 * power_index + linewidth_index
        power_pso = particle_swarm_optimize(
            power_training_objective,
            pnn_bounds,
            swarm_size=PNN_POWER_SWEEP_SWARM_SIZE,
            max_iterations=PNN_POWER_SWEEP_ITERATIONS,
            periodic_dimensions=range(PNN_NUM_CHANNELS, 2 * PNN_NUM_CHANNELS),
            seed=power_seed,
        )
        pnn_power_training_results[result_key] = {
            'scenario': 'nonlinear_power_sweep',
            'linewidth_hz': linewidth_key,
            'distance_km': float(DISTANCE_NONLINEAR_KM),
            'include_nonlinearity': True,
            'training_power_dbm': power_key,
            'amplitude_weights': power_pso['amplitude_weights'].copy(),
            'phase_weights': power_pso['phase_weights'].copy(),
            'best_training_ber': float(power_pso['best_fitness']),
            'best_fitness_history': power_pso['best_fitness_history'].copy(),
            'evaluations': int(power_pso['evaluations']),
            'seed': int(power_seed),
        }

        def power_joint_cdc_objective(position):
            pnn_model.set_weights(position[:PNN_NUM_CHANNELS], position[PNN_NUM_CHANNELS:])
            _, field_joint = pnn_model.forward(rx_power_train, dt_s=dt_train, return_optical_field=True)
            r_joint = coherent_detection(field_joint, phase_lo=phase_lo_power_train)
            r_joint = cd_compensation(r_joint, dt=dt_train, length_km=DISTANCE_NONLINEAR_KM, dispersion_d=FIBER_PARAMS.d_1550)
            return calculate_ber(train_bits, demap_qpsk(sample_to_symbols(r_joint, sps=SPS)))

        power_joint_seed = power_seed + 50000
        power_joint_pso = particle_swarm_optimize(
            power_joint_cdc_objective, pnn_bounds,
            swarm_size=PNN_POWER_SWEEP_SWARM_SIZE,
            max_iterations=PNN_POWER_SWEEP_ITERATIONS,
            periodic_dimensions=range(PNN_NUM_CHANNELS, 2 * PNN_NUM_CHANNELS),
            seed=power_joint_seed,
        )
        pnn_power_joint_cdc_results[result_key] = {
            'scenario': 'nonlinear_power_sweep_joint_cdc',
            'linewidth_hz': linewidth_key,
            'distance_km': float(DISTANCE_NONLINEAR_KM),
            'include_nonlinearity': True,
            'training_power_dbm': power_key,
            'amplitude_weights': power_joint_pso['amplitude_weights'].copy(),
            'phase_weights': power_joint_pso['phase_weights'].copy(),
            'best_training_ber': float(power_joint_pso['best_fitness']),
            'best_fitness_history': power_joint_pso['best_fitness_history'].copy(),
            'evaluations': int(power_joint_pso['evaluations']),
            'seed': int(power_joint_seed),
        }
        print(
            f"Treino por potencia: {power_key:g} dBm, "
            f"linewidth={linewidth_key/1e3:g} kHz, "
            f"BER={power_pso['best_fitness']:.6g}, "
            f"avaliacoes={power_pso['evaluations']}"
        )

# Resumo da BER de treino dos modelos especializados em cada potência.
fig, axes = plt.subplots(1, len(LINEWIDTHS_HZ), figsize=(15, 4.8), sharey=True)
for ax, linewidth_hz in zip(axes, LINEWIDTHS_HZ):
    key = float(linewidth_hz)
    training_ber_by_power = [
        pnn_power_training_results[(float(power), key)]['best_training_ber']
        for power in POWER_SWEEP_DBM
    ]
    joint_training_ber_by_power = [pnn_power_joint_cdc_results[(float(power), key)]['best_training_ber'] for power in POWER_SWEEP_DBM]
    ax.semilogy(POWER_SWEEP_DBM, np.maximum(training_ber_by_power, 1.0 / (2 * len(train_bits))), 'o-', label='PNN sem CDC')
    ax.semilogy(POWER_SWEEP_DBM, np.maximum(joint_training_ber_by_power, 1.0 / (2 * len(train_bits))), 's--', label='PNN + CDC')
    ax.legend()
    ax.set_title(f'{linewidth_hz/1e3:g} kHz')
    ax.set_xlabel('Potencia de treino [dBm]')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
axes[0].set_ylabel('BER de treino apos PSO')
fig.suptitle('BER de treino: PNN isolada e PNN treinada com CDC')
fig.tight_layout()
fig.savefig(FIG_DIR / 'pnn_coerente_ber_treino_por_potencia.png', dpi=250)
plt.show()


# Convergencia das PNNs treinadas em conjunto com a CDC.
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
colors = ['tab:blue', 'tab:orange', 'tab:green']
for ax, scenario, title in zip(axes, ('dispersion', 'nonlinear'), ('Dispersao cromatica + CDC', 'Dispersao + Kerr + CDC')):
    for linewidth_hz, color in zip(LINEWIDTHS_HZ, colors):
        history = joint_training_histories[scenario][float(linewidth_hz)]
        ax.plot(np.arange(history.size), history, color=color, marker='.', markersize=3, label=f'{linewidth_hz/1e3:g} kHz')
    ax.set_title(title)
    ax.set_xlabel('Iteracao (0 = enxame inicial)')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(title='Largura de linha TX = LO')
axes[0].set_ylabel('Melhor BER de treino apos CDC')
fig.suptitle('Convergencia do PSO para PNN + CDC')
fig.tight_layout()
fig.savefig(FIG_DIR / 'pnn_coerente_convergencia_pso_pnn_mais_cdc.png', dpi=250)
plt.show()

# Convergencia: uma curva para cada largura de linha em cada cenário treinado.
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
colors = ['tab:blue', 'tab:orange', 'tab:green']
for ax, scenario, title in zip(
    axes,
    ('dispersion', 'nonlinear'),
    ('Dispersao cromatica', 'Dispersao + Kerr'),
):
    for linewidth_hz, color in zip(LINEWIDTHS_HZ, colors):
        history = training_histories[scenario][float(linewidth_hz)]
        ax.plot(
            np.arange(history.size), history,
            color=color, marker='.', markersize=3,
            label=f'{linewidth_hz/1e3:g} kHz',
        )
    ax.set_title(title)
    ax.set_xlabel('Iteracao (0 = enxame inicial)')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(title='Largura de linha TX = LO')
axes[0].set_ylabel('Melhor BER de treino')
fig.suptitle('Convergencia do PSO para os cenarios coerentes')
fig.tight_layout()
fig.savefig(FIG_DIR / 'pnn_coerente_convergencia_pso.png', dpi=250)
plt.show()

PNN_MODEL_DIR = PROJECT_ROOT / 'artifacts' / 'models'
PNN_MODEL_DIR.mkdir(parents=True, exist_ok=True)
PNN_RESULT_PATH = PNN_MODEL_DIR / f'pnn_coerente_pso_treinamento_{CORRELATION_TAG}.json'
pnn_training_record = {
    'training_symbols': int(PNN_TRAIN_SYMBOLS),
    'evaluation_symbols': int(NUM_SYMBOLS),
    'training_seed': int(PNN_TRAIN_SEED),
    'evaluation_symbol_seed': 42,
    'phase_correlation_tx_rx': float(PHASE_CORRELATION_TX_RX),
    'channels': int(PNN_NUM_CHANNELS),
    'base_delay_ps': float(PNN_BASE_DELAY_PS),
    'normalize_power': bool(PNN_NORMALIZE_POWER),
    'swarm_size': int(PNN_SWARM_SIZE),
    'iterations': int(PNN_ITERATIONS),
    'results': [
        {
            'scenario': result['scenario'],
            'linewidth_hz': result['linewidth_hz'],
            'distance_km': result['distance_km'],
            'include_nonlinearity': result['include_nonlinearity'],
            'training_power_dbm': result['training_power_dbm'],
            'amplitude_weights': result['amplitude_weights'].tolist(),
            'phase_weights_rad': result['phase_weights'].tolist(),
            'training_ber': result['best_training_ber'],
            'best_ber_history': result['best_fitness_history'].tolist(),
            'evaluations': result['evaluations'],
            'seed': result['seed'],
        }
        for result in pnn_training_results.values()
    ],
    'joint_cdc_results': [
        {
            'scenario': result['scenario'],
            'linewidth_hz': result['linewidth_hz'],
            'distance_km': result['distance_km'],
            'include_nonlinearity': result['include_nonlinearity'],
            'training_power_dbm': result['training_power_dbm'],
            'amplitude_weights': result['amplitude_weights'].tolist(),
            'phase_weights_rad': result['phase_weights'].tolist(),
            'training_ber_after_cdc': result['best_training_ber'],
            'best_ber_history': result['best_fitness_history'].tolist(),
            'evaluations': result['evaluations'], 'seed': result['seed'],
        }
        for result in pnn_joint_cdc_results.values()
    ],
}
pnn_training_record['power_sweep_training'] = [
    {
        'training_power_dbm': float(power),
        'linewidth_hz': float(linewidth),
        'amplitude_weights': pnn_power_training_results[(float(power), float(linewidth))]['amplitude_weights'].tolist(),
        'phase_weights_rad': pnn_power_training_results[(float(power), float(linewidth))]['phase_weights'].tolist(),
        'training_ber': float(pnn_power_training_results[(float(power), float(linewidth))]['best_training_ber']),
        'best_ber_history': pnn_power_training_results[(float(power), float(linewidth))]['best_fitness_history'].tolist(),
        'evaluations': int(pnn_power_training_results[(float(power), float(linewidth))]['evaluations']),
        'seed': int(pnn_power_training_results[(float(power), float(linewidth))]['seed']),
    }
    for power in POWER_SWEEP_DBM
    for linewidth in LINEWIDTHS_HZ
]
pnn_training_record['power_sweep_joint_cdc_training'] = [
    {
        'training_power_dbm': float(power),
        'linewidth_hz': float(linewidth),
        'amplitude_weights': pnn_power_joint_cdc_results[(float(power), float(linewidth))]['amplitude_weights'].tolist(),
        'phase_weights_rad': pnn_power_joint_cdc_results[(float(power), float(linewidth))]['phase_weights'].tolist(),
        'training_ber_after_cdc': float(pnn_power_joint_cdc_results[(float(power), float(linewidth))]['best_training_ber']),
        'best_ber_history': pnn_power_joint_cdc_results[(float(power), float(linewidth))]['best_fitness_history'].tolist(),
        'evaluations': int(pnn_power_joint_cdc_results[(float(power), float(linewidth))]['evaluations']),
        'seed': int(pnn_power_joint_cdc_results[(float(power), float(linewidth))]['seed']),
    }
    for power in POWER_SWEEP_DBM
    for linewidth in LINEWIDTHS_HZ
]
pnn_training_record['distance_sweep_training'] = [
    {
        'distance_km': float(distance),
        'linewidth_hz': float(linewidth),
        'pnn_only': {
            'amplitude_weights': distance_sweep_training_results[(float(distance), float(linewidth))]['pnn_only']['amplitude_weights'].tolist(),
            'phase_weights_rad': distance_sweep_training_results[(float(distance), float(linewidth))]['pnn_only']['phase_weights'].tolist(),
            'training_ber': distance_sweep_training_results[(float(distance), float(linewidth))]['pnn_only']['best_training_ber'],
            'history': distance_sweep_training_results[(float(distance), float(linewidth))]['pnn_only']['best_fitness_history'].tolist(),
            'evaluations': distance_sweep_training_results[(float(distance), float(linewidth))]['pnn_only']['evaluations'],
        },
        'pnn_plus_cdc': {
            'amplitude_weights': distance_sweep_training_results[(float(distance), float(linewidth))]['pnn_plus_cdc']['amplitude_weights'].tolist(),
            'phase_weights_rad': distance_sweep_training_results[(float(distance), float(linewidth))]['pnn_plus_cdc']['phase_weights'].tolist(),
            'training_ber_after_cdc': distance_sweep_training_results[(float(distance), float(linewidth))]['pnn_plus_cdc']['best_training_ber'],
            'history': distance_sweep_training_results[(float(distance), float(linewidth))]['pnn_plus_cdc']['best_fitness_history'].tolist(),
            'evaluations': distance_sweep_training_results[(float(distance), float(linewidth))]['pnn_plus_cdc']['evaluations'],
        },
    }
    for distance in DISTANCES_KM
    for linewidth in LINEWIDTHS_HZ
]
PNN_RESULT_PATH.write_text(json.dumps(pnn_training_record, indent=2), encoding='utf-8')
print(f'Pesos e historicos do treinamento salvos em: {PNN_RESULT_PATH}')

```

    dispersion + CDC, linewidth=0 kHz: BER treino=0
    dispersion, linewidth=0 kHz: BER treino=0.292969, avaliacoes=6060
    dispersion + CDC, linewidth=100 kHz: BER treino=0
    dispersion, linewidth=100 kHz: BER treino=0.294922, avaliacoes=6060
    dispersion + CDC, linewidth=1000 kHz: BER treino=0
    dispersion, linewidth=1000 kHz: BER treino=0.304199, avaliacoes=6060
    nonlinear + CDC, linewidth=0 kHz: BER treino=0
    nonlinear, linewidth=0 kHz: BER treino=0.310547, avaliacoes=6060
    nonlinear + CDC, linewidth=100 kHz: BER treino=0
    nonlinear, linewidth=100 kHz: BER treino=0.316895, avaliacoes=6060
    nonlinear + CDC, linewidth=1000 kHz: BER treino=0.000488281
    nonlinear, linewidth=1000 kHz: BER treino=0.305664, avaliacoes=6060
    Treinamento independente concluido para 50 km.
    Treinamento independente concluido para 100 km.
    Treinamento independente concluido para 150 km.
    Treinamento independente concluido para 200 km.
    Treinamento independente concluido para 250 km.
    Treinamento independente concluido para 300 km.
    Treinamento independente concluido para 350 km.
    Treinamento independente concluido para 400 km.
    Treinamento independente concluido para 450 km.
    Treinamento independente concluido para 500 km.
    Treinamento independente concluido para 550 km.
    Treinamento independente concluido para 600 km.
    Treinamento independente concluido para 650 km.
    Treinamento independente concluido para 700 km.
    Treinamento independente concluido para 750 km.
    Treinamento independente concluido para 800 km.
    Treinamento independente concluido para 850 km.
    Treinamento independente concluido para 900 km.
    Treinamento independente concluido para 950 km.
    Treinamento independente concluido para 1000 km.
    Treinamento independente concluido para 1050 km.
    Treinamento independente concluido para 1100 km.
    Treinamento independente concluido para 1150 km.
    Treinamento independente concluido para 1200 km.
    


    
![png](exploracao_pnn_dispersao_edfa_correlacao_fase_files/exploracao_pnn_dispersao_edfa_correlacao_fase_13_1.png)
    


    Treino por potencia: -20 dBm, linewidth=0 kHz, BER=0.294434, avaliacoes=6060
    Treino por potencia: -20 dBm, linewidth=100 kHz, BER=0.296875, avaliacoes=6060
    Treino por potencia: -20 dBm, linewidth=1000 kHz, BER=0.292969, avaliacoes=6060
    Treino por potencia: -18 dBm, linewidth=0 kHz, BER=0.296875, avaliacoes=6060
    Treino por potencia: -18 dBm, linewidth=100 kHz, BER=0.289551, avaliacoes=6060
    Treino por potencia: -18 dBm, linewidth=1000 kHz, BER=0.303711, avaliacoes=6060
    Treino por potencia: -16 dBm, linewidth=0 kHz, BER=0.297852, avaliacoes=6060
    Treino por potencia: -16 dBm, linewidth=100 kHz, BER=0.298828, avaliacoes=6060
    Treino por potencia: -16 dBm, linewidth=1000 kHz, BER=0.305664, avaliacoes=6060
    Treino por potencia: -14 dBm, linewidth=0 kHz, BER=0.291504, avaliacoes=6060
    Treino por potencia: -14 dBm, linewidth=100 kHz, BER=0.296387, avaliacoes=6060
    Treino por potencia: -14 dBm, linewidth=1000 kHz, BER=0.302734, avaliacoes=6060
    Treino por potencia: -12 dBm, linewidth=0 kHz, BER=0.296875, avaliacoes=6060
    Treino por potencia: -12 dBm, linewidth=100 kHz, BER=0.29541, avaliacoes=6060
    Treino por potencia: -12 dBm, linewidth=1000 kHz, BER=0.294434, avaliacoes=6060
    Treino por potencia: -10 dBm, linewidth=0 kHz, BER=0.299316, avaliacoes=6060
    Treino por potencia: -10 dBm, linewidth=100 kHz, BER=0.29541, avaliacoes=6060
    Treino por potencia: -10 dBm, linewidth=1000 kHz, BER=0.296387, avaliacoes=6060
    Treino por potencia: -8 dBm, linewidth=0 kHz, BER=0.29834, avaliacoes=6060
    Treino por potencia: -8 dBm, linewidth=100 kHz, BER=0.294922, avaliacoes=6060
    Treino por potencia: -8 dBm, linewidth=1000 kHz, BER=0.293945, avaliacoes=6060
    Treino por potencia: -6 dBm, linewidth=0 kHz, BER=0.292969, avaliacoes=6060
    Treino por potencia: -6 dBm, linewidth=100 kHz, BER=0.29834, avaliacoes=6060
    Treino por potencia: -6 dBm, linewidth=1000 kHz, BER=0.29541, avaliacoes=6060
    Treino por potencia: -4 dBm, linewidth=0 kHz, BER=0.295898, avaliacoes=6060
    Treino por potencia: -4 dBm, linewidth=100 kHz, BER=0.298828, avaliacoes=6060
    Treino por potencia: -4 dBm, linewidth=1000 kHz, BER=0.302734, avaliacoes=6060
    Treino por potencia: -2 dBm, linewidth=0 kHz, BER=0.307129, avaliacoes=6060
    Treino por potencia: -2 dBm, linewidth=100 kHz, BER=0.305664, avaliacoes=6060
    Treino por potencia: -2 dBm, linewidth=1000 kHz, BER=0.309082, avaliacoes=6060
    Treino por potencia: 2 dBm, linewidth=0 kHz, BER=0.331543, avaliacoes=6060
    Treino por potencia: 2 dBm, linewidth=100 kHz, BER=0.328125, avaliacoes=6060
    Treino por potencia: 2 dBm, linewidth=1000 kHz, BER=0.333496, avaliacoes=6060
    Treino por potencia: 4 dBm, linewidth=0 kHz, BER=0.385742, avaliacoes=6060
    Treino por potencia: 4 dBm, linewidth=100 kHz, BER=0.376953, avaliacoes=6060
    Treino por potencia: 4 dBm, linewidth=1000 kHz, BER=0.375488, avaliacoes=6060
    Treino por potencia: 6 dBm, linewidth=0 kHz, BER=0.426758, avaliacoes=6060
    Treino por potencia: 6 dBm, linewidth=100 kHz, BER=0.429199, avaliacoes=6060
    Treino por potencia: 6 dBm, linewidth=1000 kHz, BER=0.42334, avaliacoes=6060
    Treino por potencia: 8 dBm, linewidth=0 kHz, BER=0.433594, avaliacoes=6060
    Treino por potencia: 8 dBm, linewidth=100 kHz, BER=0.43457, avaliacoes=6060
    Treino por potencia: 8 dBm, linewidth=1000 kHz, BER=0.432129, avaliacoes=6060
    Treino por potencia: 10 dBm, linewidth=0 kHz, BER=0.427734, avaliacoes=6060
    Treino por potencia: 10 dBm, linewidth=100 kHz, BER=0.423828, avaliacoes=6060
    Treino por potencia: 10 dBm, linewidth=1000 kHz, BER=0.433105, avaliacoes=6060
    Treino por potencia: 12 dBm, linewidth=0 kHz, BER=0.446289, avaliacoes=6060
    Treino por potencia: 12 dBm, linewidth=100 kHz, BER=0.451172, avaliacoes=6060
    Treino por potencia: 12 dBm, linewidth=1000 kHz, BER=0.431152, avaliacoes=6060
    Treino por potencia: 14 dBm, linewidth=0 kHz, BER=0.4375, avaliacoes=6060
    Treino por potencia: 14 dBm, linewidth=100 kHz, BER=0.446777, avaliacoes=6060
    Treino por potencia: 14 dBm, linewidth=1000 kHz, BER=0.42627, avaliacoes=6060
    Treino por potencia: 16 dBm, linewidth=0 kHz, BER=0.44043, avaliacoes=6060
    Treino por potencia: 16 dBm, linewidth=100 kHz, BER=0.449707, avaliacoes=6060
    Treino por potencia: 16 dBm, linewidth=1000 kHz, BER=0.447754, avaliacoes=6060
    Treino por potencia: 18 dBm, linewidth=0 kHz, BER=0.439941, avaliacoes=6060
    Treino por potencia: 18 dBm, linewidth=100 kHz, BER=0.436523, avaliacoes=6060
    Treino por potencia: 18 dBm, linewidth=1000 kHz, BER=0.437012, avaliacoes=6060
    Treino por potencia: 20 dBm, linewidth=0 kHz, BER=0.455078, avaliacoes=6060
    Treino por potencia: 20 dBm, linewidth=100 kHz, BER=0.461914, avaliacoes=6060
    Treino por potencia: 20 dBm, linewidth=1000 kHz, BER=0.452637, avaliacoes=6060
    


    
![png](exploracao_pnn_dispersao_edfa_correlacao_fase_files/exploracao_pnn_dispersao_edfa_correlacao_fase_13_3.png)
    



    
![png](exploracao_pnn_dispersao_edfa_correlacao_fase_files/exploracao_pnn_dispersao_edfa_correlacao_fase_13_4.png)
    



    
![png](exploracao_pnn_dispersao_edfa_correlacao_fase_files/exploracao_pnn_dispersao_edfa_correlacao_fase_13_5.png)
    


    Pesos e historicos do treinamento salvos em: D:\gits\PHOTONICS_2026\Codigos\python\pnn_coerente\artifacts\models\pnn_coerente_pso_treinamento_rho_0p99.json
    

## 7. Avaliacao com PNN treinada

Nesta etapa os pesos nao sao mais atualizados. Para a curva por distancia aplicamos a PNN treinada no enlace de distancia maxima; para a curva por potencia usamos a PNN treinada em 1000 km e 0 dBm. Cada largura de linha TX/LO usa o modelo treinado para aquela largura. A avaliacao usa a sequencia e o ruido de fase independentes da linha de base. As curvas mostram PNN sem CDC e PNN seguida da CDC convencional.


Adicionalmente, outra PNN independente para cada distancia da varredura e treinada para cada largura de linha e objetivo (antes da CDC ou em conjunto com a CDC). A avaliacao gera uma curva separada para essas PNNs especializadas por span, permitindo compara-las com as PNNs treinadas apenas na distancia maxima.


```python
# Analise 1: pesos treinados na distancia maxima, fixos ao longo da curva.
pnn_ber_vs_distance_no_cdc = {float(lw): [] for lw in LINEWIDTHS_HZ}
pnn_ber_vs_distance_with_cdc = {float(lw): [] for lw in LINEWIDTHS_HZ}
pnn_ber_vs_distance_per_span_no_cdc = {float(lw): [] for lw in LINEWIDTHS_HZ}
pnn_ber_vs_distance_per_span_with_cdc = {float(lw): [] for lw in LINEWIDTHS_HZ}
for linewidth_hz in LINEWIDTHS_HZ:
    linewidth_key = float(linewidth_hz)
    phase_tx, phase_lo = phase_noise[linewidth_key]
    trained = pnn_training_results[('dispersion', linewidth_key)]
    pnn_model.set_weights(trained['amplitude_weights'], trained['phase_weights'])
    for distance_km in DISTANCES_KM:
        field_tx = u_tx_0dbm * 10.0 ** (TX_POWER_DISPERSION_DBM / 20.0)
        field_tx = field_tx * np.exp(1j * phase_tx)
        field_link = propagate_spanned_link(
            field_tx, t, distance_km, include_nonlinearity=False
        )
        field_rx, _, _ = receiver_edfa_and_attenuator(field_link)
        _, field_pnn = pnn_model.forward(
            field_rx, dt_s=dt, return_optical_field=True
        )
        r_pnn = coherent_detection(field_pnn, phase_lo=phase_lo)
        rx_pnn_symbols = sample_to_symbols(r_pnn, sps=SPS)
        pnn_ber_vs_distance_no_cdc[linewidth_key].append(
            calculate_ber(bits, demap_qpsk(rx_pnn_symbols))
        )
        joint_trained = pnn_joint_cdc_results[('dispersion', linewidth_key)]
        pnn_model.set_weights(joint_trained['amplitude_weights'], joint_trained['phase_weights'])
        _, joint_field = pnn_model.forward(field_rx, dt_s=dt, return_optical_field=True)
        r_joint_pnn = coherent_detection(joint_field, phase_lo=phase_lo)
        r_joint_pnn_cdc = cd_compensation(r_joint_pnn, dt=dt, length_km=distance_km, dispersion_d=FIBER_PARAMS.d_1550)
        joint_symbols_cdc = sample_to_symbols(r_joint_pnn_cdc, sps=SPS)
        pnn_ber_vs_distance_with_cdc[linewidth_key].append(
            calculate_ber(bits, demap_qpsk(joint_symbols_cdc))
        )

        # Modelos treinados especificamente para este span da varredura.
        span_models = distance_sweep_training_results[(float(distance_km), linewidth_key)]
        pnn_model.set_weights(
            span_models['pnn_only']['amplitude_weights'],
            span_models['pnn_only']['phase_weights'],
        )
        _, span_field = pnn_model.forward(field_rx, dt_s=dt, return_optical_field=True)
        span_r = coherent_detection(span_field, phase_lo=phase_lo)
        pnn_ber_vs_distance_per_span_no_cdc[linewidth_key].append(
            calculate_ber(bits, demap_qpsk(sample_to_symbols(span_r, sps=SPS)))
        )

        pnn_model.set_weights(
            span_models['pnn_plus_cdc']['amplitude_weights'],
            span_models['pnn_plus_cdc']['phase_weights'],
        )
        _, span_joint_field = pnn_model.forward(field_rx, dt_s=dt, return_optical_field=True)
        span_joint_r = coherent_detection(span_joint_field, phase_lo=phase_lo)
        span_joint_r = cd_compensation(
            span_joint_r, dt=dt, length_km=distance_km,
            dispersion_d=FIBER_PARAMS.d_1550,
        )
        pnn_ber_vs_distance_per_span_with_cdc[linewidth_key].append(
            calculate_ber(bits, demap_qpsk(sample_to_symbols(span_joint_r, sps=SPS)))
        )

fig, axes = plt.subplots(1, len(LINEWIDTHS_HZ), figsize=(16, 5), sharey=True)
for ax, linewidth_hz in zip(axes, LINEWIDTHS_HZ):
    key = float(linewidth_hz)
    ax.semilogy(
        DISTANCES_KM,
        np.maximum(ber_vs_distance_no_cdc[key], BER_PLOT_FLOOR),
        '--x', color='0.45', label='Sem PNN, sem CDC',
    )
    ax.semilogy(
        DISTANCES_KM,
        np.maximum(ber_vs_distance_with_cdc[key], BER_PLOT_FLOOR),
        ':o', color='tab:blue', label='Sem PNN, com CDC',
    )
    ax.semilogy(
        DISTANCES_KM,
        np.maximum(pnn_ber_vs_distance_no_cdc[key], BER_PLOT_FLOOR),
        '-s', color='tab:orange', label='PNN sem CDC (treinada sem CDC)',
    )
    ax.semilogy(
        DISTANCES_KM,
        np.maximum(pnn_ber_vs_distance_with_cdc[key], BER_PLOT_FLOOR),
        '-^', color='tab:green', label='PNN + CDC (treinamento conjunto)',
    )
    ax.set_title(f'{linewidth_hz/1e3:g} kHz')
    ax.set_xlabel('Distancia [km]')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
axes[0].set_ylabel('BER')
axes[-1].legend(fontsize=8, loc='best')
fig.suptitle('Analise 1: BER versus distancia com pesos PNN treinados')
fig.tight_layout()
fig.savefig(FIG_DIR / 'pnn_analise_1_ber_distancia_com_pnn.png', dpi=250)
plt.show()

# Comparacao com pesos PNN retreinados independentemente para cada distancia.
fig, axes = plt.subplots(1, len(LINEWIDTHS_HZ), figsize=(16, 5), sharey=True)
for ax, linewidth_hz in zip(axes, LINEWIDTHS_HZ):
    key = float(linewidth_hz)
    ax.semilogy(DISTANCES_KM, np.maximum(ber_vs_distance_no_cdc[key], BER_PLOT_FLOOR), '--x', color='0.45', label='Sem PNN e sem CDC')
    ax.semilogy(DISTANCES_KM, np.maximum(ber_vs_distance_with_cdc[key], BER_PLOT_FLOOR), ':o', color='tab:blue', label='Sem PNN e com CDC')
    ax.semilogy(DISTANCES_KM, np.maximum(pnn_ber_vs_distance_per_span_no_cdc[key], BER_PLOT_FLOOR), '-s', color='tab:orange', label='Com PNN treinada por span, sem CDC')
    ax.semilogy(DISTANCES_KM, np.maximum(pnn_ber_vs_distance_per_span_with_cdc[key], BER_PLOT_FLOOR), '-^', color='tab:green', label='Com PNN + CDC treinadas por span')
    ax.set_title(f'{linewidth_hz/1e3:g} kHz')
    ax.set_xlabel('Distancia [km]')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
axes[0].set_ylabel('BER')
axes[-1].legend(fontsize=8, loc='best')
fig.suptitle('BER versus distancia — PNN treinada independentemente por span')
fig.tight_layout()
fig.savefig(FIG_DIR / 'pnn_analise_1_ber_distancia_treino_por_span.png', dpi=250)
plt.show()

# Analise 2: os dois tipos de PNN sao especializados para cada potencia.
pnn_ber_vs_power_retrained_no_cdc = {float(lw): [] for lw in LINEWIDTHS_HZ}
pnn_ber_vs_power_retrained_with_cdc = {float(lw): [] for lw in LINEWIDTHS_HZ}
for launch_power_dbm in POWER_SWEEP_DBM:
    tx_field = u_tx_0dbm * 10.0 ** (launch_power_dbm / 20.0)
    for linewidth_hz in LINEWIDTHS_HZ:
        linewidth_key = float(linewidth_hz)
        phase_tx, phase_lo = phase_noise[linewidth_key]
        field_tx = tx_field * np.exp(1j * phase_tx)
        field_link = propagate_spanned_link(field_tx, t, DISTANCE_NONLINEAR_KM, include_nonlinearity=True)
        field_rx, _, _ = receiver_edfa_and_attenuator(field_link)

        # PNN treinada pela BER antes da CDC.
        pnn_only = pnn_power_training_results[(float(launch_power_dbm), linewidth_key)]
        pnn_model.set_weights(pnn_only['amplitude_weights'], pnn_only['phase_weights'])
        _, field_pnn_only = pnn_model.forward(field_rx, dt_s=dt, return_optical_field=True)
        r_pnn_only = coherent_detection(field_pnn_only, phase_lo=phase_lo)
        pnn_ber_vs_power_retrained_no_cdc[linewidth_key].append(
            calculate_ber(bits, demap_qpsk(sample_to_symbols(r_pnn_only, sps=SPS)))
        )

        # PNN treinada com CDC dentro do objetivo; CDC tambem e aplicada nesta avaliacao.
        pnn_joint = pnn_power_joint_cdc_results[(float(launch_power_dbm), linewidth_key)]
        pnn_model.set_weights(pnn_joint['amplitude_weights'], pnn_joint['phase_weights'])
        _, field_pnn_joint = pnn_model.forward(field_rx, dt_s=dt, return_optical_field=True)
        r_pnn_joint = coherent_detection(field_pnn_joint, phase_lo=phase_lo)
        r_pnn_joint_cdc = cd_compensation(r_pnn_joint, dt=dt, length_km=DISTANCE_NONLINEAR_KM, dispersion_d=FIBER_PARAMS.d_1550)
        pnn_ber_vs_power_retrained_with_cdc[linewidth_key].append(
            calculate_ber(bits, demap_qpsk(sample_to_symbols(r_pnn_joint_cdc, sps=SPS)))
        )

fig, axes = plt.subplots(1, len(LINEWIDTHS_HZ), figsize=(16, 5), sharey=True)
for ax, linewidth_hz in zip(axes, LINEWIDTHS_HZ):
    key = float(linewidth_hz)
    ax.semilogy(POWER_SWEEP_DBM, np.maximum(ber_vs_power_no_cdc[key], BER_PLOT_FLOOR), '--x', color='0.45', label='Sem PNN e sem CDC')
    ax.semilogy(POWER_SWEEP_DBM, np.maximum(ber_vs_power_with_cdc[key], BER_PLOT_FLOOR), ':o', color='tab:blue', label='Sem PNN e com CDC')
    ax.semilogy(POWER_SWEEP_DBM, np.maximum(pnn_ber_vs_power_retrained_no_cdc[key], BER_PLOT_FLOOR), '-s', color='tab:orange', label='Com PNN e sem CDC')
    ax.semilogy(POWER_SWEEP_DBM, np.maximum(pnn_ber_vs_power_retrained_with_cdc[key], BER_PLOT_FLOOR), '-^', color='tab:green', label='Com PNN + CDC (treinamento conjunto)')
    ax.set_title(f'{linewidth_hz/1e3:g} kHz')
    ax.set_xlabel('Potencia de lancamento [dBm]')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
axes[0].set_ylabel('BER')
axes[-1].legend(fontsize=8, loc='best')
fig.suptitle(f'Analise 2: BER versus potencia, {DISTANCE_NONLINEAR_KM:g} km — quatro cenarios')
fig.tight_layout()
fig.savefig(FIG_DIR / 'pnn_analise_2_ber_potencia_quatro_cenarios.png', dpi=250)
plt.show()

# Acrescenta as BERs de avaliacao ao arquivo de checkpoints/pesos.
for scenario, curves in [
    ('dispersion', pnn_ber_vs_distance_no_cdc),
    
]:
    for item in pnn_training_record['results']:
        if item['scenario'] != scenario:
            continue
        key = float(item['linewidth_hz'])
        item['evaluation_ber_without_cdc'] = [float(v) for v in curves[key]]
pnn_training_record['distance_sweep_km'] = DISTANCES_KM.tolist()
pnn_training_record['evaluation_ber_distance_per_span_no_cdc'] = {
    str(key): [float(v) for v in values]
    for key, values in pnn_ber_vs_distance_per_span_no_cdc.items()
}
pnn_training_record['evaluation_ber_distance_per_span_with_cdc'] = {
    str(key): [float(v) for v in values]
    for key, values in pnn_ber_vs_distance_per_span_with_cdc.items()
}
pnn_training_record['power_sweep_dbm'] = POWER_SWEEP_DBM.tolist()
pnn_training_record['evaluation_ber_with_cdc_distance'] = {
    str(key): [float(v) for v in values]
    for key, values in pnn_ber_vs_distance_with_cdc.items()
}
pnn_training_record['evaluation_ber_with_cdc_power'] = {
    str(key): [float(v) for v in values]
    for key, values in pnn_ber_vs_power_retrained_with_cdc.items()
}
pnn_training_record['evaluation_ber_power_retrained_without_cdc'] = {
    str(key): [float(v) for v in values]
    for key, values in pnn_ber_vs_power_retrained_no_cdc.items()
}
pnn_training_record['power_curve_semantics'] = {
    'baseline_no_cdc': 'sem PNN e sem CDC',
    'baseline_with_cdc': 'sem PNN e com CDC',
    'pnn_no_cdc': 'PNN treinada sem CDC para cada potencia',
    'pnn_with_cdc': 'PNN treinada com CDC para cada potencia; CDC aplicada na avaliacao',
}
pnn_training_record['evaluation_ber_power_retrained_with_cdc'] = {
    str(key): [float(v) for v in values]
    for key, values in pnn_ber_vs_power_retrained_with_cdc.items()
}
PNN_RESULT_PATH.write_text(json.dumps(pnn_training_record, indent=2), encoding='utf-8')
print(f'Avaliacao com PNN adicionada ao registro: {PNN_RESULT_PATH}')

```


    
![png](exploracao_pnn_dispersao_edfa_correlacao_fase_files/exploracao_pnn_dispersao_edfa_correlacao_fase_15_0.png)
    



    
![png](exploracao_pnn_dispersao_edfa_correlacao_fase_files/exploracao_pnn_dispersao_edfa_correlacao_fase_15_1.png)
    



    
![png](exploracao_pnn_dispersao_edfa_correlacao_fase_files/exploracao_pnn_dispersao_edfa_correlacao_fase_15_2.png)
    


    PNN evaluation added to registry: D:\gits\PHOTONICS_2026\Codigos\python\pnn_coerente\artifacts\models\pnn_coerente_pso_treinamento_rho_0p99.json
    

## Observations and Next Steps

- Span lengths are uniform. For instance, with `SPAN_LENGTH_KM = 200` and a 1000 km link, the function uses exactly five 200 km spans.
- Received power is maintained at 0 dBm by a fixed terminal EDFA followed by variable optical attenuation. The value is printed after analyses for verification.
- The `EDFAIdeal` device in `src/utils/edfa.py` applies fixed field gain, without ASE noise or saturation. Propagation calls the same method for in-line and terminal EDFAs. It can be replaced by future analytical models via `EDFA_MODEL`. Inclusion of ASE/noise figure can be a future analysis.
- The two analyses demonstrate BER with and without conventional CDC. CDC is applied digitally after coherent detection and before symbol decision; in the nonlinear case, it removes second-order dispersion but does not undo Kerr nonlinear distortions.
- Sections 6 and 7 train and evaluate separate PNNs per scenario and linewidth. In the power sweep, results include both a PNN trained at 0 dBm and held fixed, as well as specialized PNNs retrained per power. All trainings use the same `PHASE_CORRELATION_TX_RX` coefficient.

- The distance analysis also trains independent models per span, in addition to evaluating models trained solely at maximum distance.

