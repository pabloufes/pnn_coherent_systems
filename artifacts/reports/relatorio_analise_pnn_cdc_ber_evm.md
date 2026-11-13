# PNN and CDC Sweep Analysis: BER, EVM, and Constellations

## Purpose

This report documents a reproducible simulation study for a coherent QPSK link with a photonic tapped-delay-line equalizer (PNN). It compares BER and RMS EVM across fiber distance and launch power, using separate PSO training objectives (BER and EVM) and two DSP equalizer families (conventional CDC and joint PNN-CDC).

The executable analysis is [`scripts/generate_pnn_cdc_analysis.py`](../../scripts/generate_pnn_cdc_analysis.py), launched by [`notebooks/geracao_analise_pnn_cdc.ipynb`](../../notebooks/geracao_analise_pnn_cdc.ipynb). The full point-by-point results, including final tap weights, are in [`pnn_cdc_ber_evm_sweeps.csv`](pnn_cdc_ber_evm_sweeps.csv); run settings are in [`pnn_cdc_ber_evm_sweeps_metadata.json`](pnn_cdc_ber_evm_sweeps_metadata.json).

To recreate all figures without running PSO or repeating the parameter sweeps, run [`scripts/plot_pnn_cdc_analysis.py`](../../scripts/plot_pnn_cdc_analysis.py). It reads the saved CSV/metadata and reuses a compact cache of received constellation samples at the requested points, [`pnn_cdc_constellation_points.npz`](pnn_cdc_constellation_points.npz). If the cache is absent, it reconstructs only those two constellation operating points from the saved PNN weights; it does not retrain the PNN.

## Simulation design

| Item | Setting |
|---|---|
| Modulation / symbol rate | Gray QPSK, 10 GBaud |
| Sampling | 8 samples/symbol (80 GSa/s) |
| Training / test frame | 1,024 / 4,096 symbols; separate deterministic seeds |
| Fiber spans | 50 km, ideal EDFA restores span loss |
| TX/LO linewidths | 0, 100 kHz, 1 MHz; phase correlation `ρ = 0.99` |
| PNN | 8 taps, 25 ps spacing (`T/4` for `Ts = 100 ps`), normalized weights |
| PSO | 30 particles, 60 iterations; phase dimensions are periodic |
| Joint CDC regularization | `ε = 10⁻³ × mean(|H_PNN|²)` for each candidate PNN |
| Distance sweep | 50–1,000 km in 50 km steps; launch power fixed at 0 dBm; linear fiber |
| Launch-power sweep | −20 to +20 dBm in 2 dB steps; distance fixed at 1,000 km; SSFM with Kerr |

The five strategies in the result table are no PNN/no CDC, no PNN with conventional CDC, PNN without CDC, PNN jointly trained with conventional CDC, and PNN jointly trained with joint CDC. Figures comparing the two equalizer families share the same two no-PNN baselines and PNN-only curve; the final PNN strategy changes by family.

For BER, a constant phase offset is estimated using known symbols in the training frame and applied to the independent test frame. This avoids scoring an arbitrary constant PNN phase rotation as bit errors. It is a training-aided constant phase correction, not time-varying carrier phase recovery (CPE/BPS). EVM follows the project metric: RMS error after average-power normalization and optimal constant-phase removal. Thus, residual time-varying phase noise remains, while a global phase ambiguity is removed from both metrics.

The receiver uses no ASE noise, frequency offset, timing recovery, or time-varying CPE/BPS. The ideal EDFAs add no ASE. The joint CDC inverts the modeled linear fiber dispersion and the PNN field response; it does not invert Kerr. The power sweep therefore measures the behavior of this linear equalizer in a Kerr-impaired link, not a nonlinear inverse equalizer.

## Main observations

Medians below are across all points and all three linewidths within the named sweep and training objective. BER maxima are included to expose isolated failure points. BER points at the plot floor (`1/(2 × 4,096) = 1.22 × 10⁻⁴`) are floored for visualization; a floor point is not an exact BER measurement. With 8,192 test bits, zero observed errors also does not establish zero underlying BER.

### Linear distance sweep

- **No PNN / no CDC:** BER median is 0.324 for both training-objective panels; EVM median is 107.2%. It degrades substantially with distance.
- **No PNN / conventional CDC:** BER was zero in the finite test frames across this sweep. EVM median is 5.37% (range 3.02–6.77%). This is the strongest baseline in the distance study.
- **PNN / no CDC:** BER medians are 0.223 (BER-trained set) and 0.212 (EVM-trained set); EVM medians are 86.1% and 83.5%, respectively. A PNN alone does not replace dispersion compensation here.
- **PNN + conventional CDC, BER objective:** median BER is zero, with a maximum of 0.0155; median EVM is 36.9%. The BER objective yields low BER at most distances but does not minimize waveform error.
- **PNN + conventional CDC, EVM objective:** median EVM is 4.71%, and BER was zero in the finite test frames over the evaluated distance grid.
- **PNN + joint CDC:** BER was at or near the display floor for both training objectives (maximum 0.0002 with BER objective; 0.0001 with EVM objective). Median EVM is 5.78% (BER-trained) and 5.01% (EVM-trained). In this linear regime, joint CDC is competitive with conventional CDC, but does not outperform the no-PNN CDC reference in a way established by this single-seed experiment.

At 300 km, the notebook saves the requested constellations for each linewidth, objective and four plotted strategies. The plotted RX points are normalized in average power; the training-estimated phase correction is applied before display.

### Kerr launch-power sweep at 1,000 km

- **No PNN / no CDC:** median BER is 0.376 and median EVM is 118.1%.
- **No PNN / conventional CDC:** median BER is 0.0028, but the maximum reaches 0.509 at high launch powers; median EVM is 29.9%, with a maximum above 140%. Linear CDC cannot undo Kerr distortion.
- **PNN / no CDC:** median BER is 0.320 (BER-trained) and 0.311 (EVM-trained); median EVM is about 106.7% and 104.9%. It remains weak without DSP compensation.
- **PNN + conventional CDC:** the BER-trained set has median BER 0.0034 and median EVM 46.6%; the EVM-trained set has median EVM 26.2% and median BER 0.0018. Both sets reach BER around 0.5 at the highest-power points, showing a strong nonlinear penalty.
- **PNN + joint CDC:** median BER is 0.0039 (BER objective) and 0.0031 (EVM objective); median EVM is 31.1% and 29.6%. It does not remove the high-power Kerr penalty and offers no clear sweep-wide advantage over conventional CDC in this setup.

At 0 dBm, the notebook saves the requested power-sweep constellations for each linewidth, objective and strategy. The power sweep is a separate physical regime from the distance sweep; its curves should not be treated as a direct continuation of the linear-distance curves.

## Interpretation and limits

The linear distance experiment supports the expected role of CDC: conventional CDC without PNN is a strong reference. PNN weights trained jointly with CDC can approach that reference in BER, while the metric minimized by PSO affects whether the optimized solution emphasizes bit decisions or RMS waveform error. A PNN with 25 ps spacing is fractionally spaced relative to the 100 ps symbol interval, but finite taps, amplitude constraints, ideal phase controls and simplified optical/detection models prevent claiming full equivalence to an adaptive electronic FSE.

In the Kerr power sweep, joint CDC is only a regularized inverse of the *linear* fiber and PNN responses. Its performance is expected to degrade when nonlinear propagation dominates. The high-power results should therefore be read as a limitation study, not as evidence that joint CDC should compensate Kerr.

Each operating point uses a PNN optimized specifically for that linewidth, sweep value and objective. This is not a single fixed PNN operating over all distances or powers. One PSO seed and one test frame were used per point; no confidence intervals or multi-seed variability estimates are available. The reported medians are descriptive, not population estimates.

## Figures and artifacts

All figures are saved under `artifacts/figures/`:

- `distance_ber_{ber|evm}_{conventional|joint}_cdc.png`
- `distance_evm_{ber|evm}_{conventional|joint}_cdc.png`
- `power_ber_{ber|evm}_{conventional|joint}_cdc.png`
- `power_evm_{ber|evm}_{conventional|joint}_cdc.png`
- `distance_constellations_{ber|evm}_{conventional|joint}_cdc.png` — 300 km
- `power_constellations_{ber|evm}_{conventional|joint}_cdc.png` — 0 dBm

The notebook's theory section cites Gitlin, Meadors & Weinstein (1981), [*Fractionally-Spaced Equalization: An Improved Digital Transversal Equalizer*](https://doi.org/10.1002/j.1538-7305.1981.tb00240.x). For the previously documented distance-only experiment using a different PNN/PSO configuration, see [`relatorio_ber_distancia_pnn_cdc_joint_cdc.md`](relatorio_ber_distancia_pnn_cdc_joint_cdc.md).
