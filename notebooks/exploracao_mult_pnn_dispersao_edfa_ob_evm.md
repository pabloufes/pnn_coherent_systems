# Distributed Optical PNNs in Multi-Span Coherent QPSK Links

## Overview

This document describes the algorithm implemented in the notebook `exploracao_mult_pnn_dispersao_edfa_ob_evm.ipynb`. It investigates the performance of **Distributed Optical Photonic Neural Networks (PNNs)** in a multi-span EDFA-amplified coherent optical link at 10 GBaud QPSK over up to **10,000 km** (200 spans × 50 km each).

The core idea is: instead of placing a single PNN at the receiver, an optical PNN is deployed **in-line after each fiber span**, allowing distributed analog optical equalization of chromatic dispersion as the signal propagates.

---

## System Architecture

Each span follows the structure:

```
Transmitter → [Fiber_k → PNN_k → EDFA_k]^(k=1..N_spans) → Terminal EDFA + VOA → Coherent Rx → (Optional Digital CDC)
```

**Per-span elements:**
1. **Fiber propagation** (50 km SMF-28): accumulates chromatic dispersion, passive loss (10 dB/span), and optional Kerr nonlinear phase shift.
2. **Optical PNN_k** (passive TDL mesh of tunable delays and phase shifters): performs distributed analog optical dispersion equalization.
3. **In-line EDFA_k** (ideal): compensates the span fiber loss + PNN insertion loss exactly, restoring the launch power.

**Receiver stage:**
- Terminal EDFA (20 dB fixed gain) followed by a Variable Optical Attenuator (VOA) that sets received power to 0 dBm.
- Coherent detection (IQ hybrid + balanced photodiodes).
- Optional digital Chromatic Dispersion Compensation (CDC) in the frequency domain.
- Downsample to 1 sample/symbol → Gray QPSK demapping → BER/EVM metrics.

---

## Four Comparative Scenarios

| Scenario | In-line PNNs | Digital CDC at Rx | Description |
|----------|-------------|-------------------|-------------|
| 1 | ✗ | ✗ | Baseline uncompensated link |
| 2 | ✗ | ✓ | Conventional digital CDC only |
| 3 | ✓ (trained w/o CDC) | ✗ | Distributed PNNs only — tests if optical PNNs can replace digital CDC |
| 4 | ✓ (trained jointly w/ CDC) | ✓ | Distributed PNNs + digital CDC jointly optimized |

---

## Configuration Parameters

### Simulation Parameters

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| `NUM_SYMBOLS` | N | 2^12 = 4096 | Number of test symbols |
| `PNN_TRAIN_SYMBOLS` | N_train | 2^10 = 1024 | Number of training symbols |
| `SPS` | — | 8 | Samples per symbol |
| `SYMBOL_RATE` | Rs | 10 GHz | QPSK symbol rate |
| `WAVELENGTH` | λ | 1550 nm | Optical carrier wavelength |
| `DT` | Δt_s | 12.5 ps | Sampling period (= 1 / (SPS × Rs)) |
| Sampling frequency | fs | 80 GHz | = SPS × Rs |

### Link Geometry

| Parameter | Value | Description |
|-----------|-------|-------------|
| `SPAN_LENGTH_KM` | 50 km | Length of each fiber span |
| `DISTANCE_SWEEP_MAX_KM` | 10,000 km | Maximum total link length |
| `N_SPANS_MAX` | 200 | Total number of spans = 10000 / 50 |
| `NUM_STEPS_PER_SPAN` | 50 | SSFM steps per span (for nonlinear regime) |
| `DISTANCES_KM` | 50, 100, …, 10000 km | Distance sweep vector |

### Fiber Parameters (SMF-28)

| Parameter | Symbol | Value |
|-----------|--------|-------|
| `alpha_db` | α | 0.2 dB/km |
| `d_1550` | D | 16.0 ps/(nm·km) |
| `slope_s` | S | 0.09 ps/(nm²·km) |
| `n2` | n₂ | 3.2×10⁻²⁰ m²/W |
| `a_eff` | A_eff | 70×10⁻¹² m² |

> **Note:** Span loss = α × L_span = 0.2 × 50 = **10 dB/span**.

### Transmitter Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `MODULATOR` | `'iq_mzm'` | IQ Mach-Zehnder Modulator |
| `LUNCH_POWER_DBM` | 0 dBm | Launch power into fiber |
| `IQ_PARAMS.electrical_response` | False | No distributed electrode response |
| `FILTER_PARAMS.tipo` | `'gaussiano'` | Gaussian electrical filter |
| `FILTER_PARAMS.largura_banda_hz` | 20 GHz (= 2 × Rs) | Electrical filter bandwidth |
| `FILTER_PARAMS.ordem` | 7 | Super-Gaussian order |

### Receiver Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `RX_EDFA_GAIN_DB` | 20 dB | Terminal EDFA gain |
| `RX_TARGET_DBM` | 0 dBm | Target received power after VOA |
| `CORRECT_PHASE_ROTATION` | False | No constant-phase rotation correction in EVM |

### PNN Architecture

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| `PNN_NUM_CHANNELS` | N | 4 | Number of parallel delay/phase branches |
| `PNN_BASE_DELAY_PS` | Δt | 25 ps | Incremental base delay per channel |
| `PNN_NORMALIZE_POWER` | — | True | Weights divided by N (1/4 each) |
| Total delay range | τ_max | (N−1)×Δt = 75 ps | Equivalent to 6 samples at 80 GHz |

### PSO Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `PNN_SWARM_SIZE` | 30 | Number of particles |
| `PNN_ITERATIONS` | 50 | Maximum iterations |
| `PSO_OBJECTIVE` | `'EVM'` | Fitness function (EVM or BER) |
| Amplitude bounds | [0.0, 1.0] × 4 | Per-channel transmittance |
| Phase bounds | [−π, π] × 4 | Per-channel phase shift (periodic) |
| Total parameters | 8 | 4 amplitudes + 4 phases per PNN |

### Phase Noise (Linewidths Tested)

| Linewidth (Hz) | Description |
|---------------|-------------|
| 0 | Ideal laser (no phase noise) |
| 100,000 (100 kHz) | Moderate linewidth |
| 1,000,000 (1 MHz) | High linewidth |

---

## External Module Implementations

### 1. `generate_qpsk_symbols` — QPSK Symbol Generator

**Location:** `src/dsp/transmitter.py`

**Purpose:** Generates a random bit sequence and Gray-mapped QPSK symbols normalized to unit power.

**Gray mapping (QPSK, unit average power):**

| Bits (I, Q) | Symbol |
|------------|--------|
| 00 | (+1 + j) / √2 |
| 01 | (+1 − j) / √2 |
| 11 | (−1 − j) / √2 |
| 10 | (−1 + j) / √2 |

```python
def generate_qpsk_symbols(num_symbols: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generates a random bit sequence and Gray-mapped QPSK symbols.

    Returns:
        symbols: Vector of normalized complex symbols E[|s|^2] = 1.
        bits: Vector of transmitted bits (size: 2 * num_symbols).
    """
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=2 * num_symbols)
    bit_i = bits[0::2]
    bit_q = bits[1::2]
    sym_i = 1.0 - 2.0 * bit_i   # 0 -> +1, 1 -> -1
    sym_q = 1.0 - 2.0 * bit_q
    symbols = (sym_i + 1j * sym_q) / np.sqrt(2.0)
    return symbols, bits
```

**In this notebook:**
- Training: `generate_qpsk_symbols(1024, seed=20261101)` → 1024 symbols
- Test: `generate_qpsk_symbols(4096, seed=20261102)` → 4096 symbols

---

### 2. `modulate_optical_signal` — Optical Modulator

**Location:** `src/dsp/transmitter.py`

**Purpose:** Modulates discrete QPSK symbols into a continuous-time optical field envelope, using an ideal or IQ-MZM transmitter model.

**In this notebook:** Uses `modulator='iq_mzm'` with an IQ Mach-Zehnder Modulator.

**IQ-MZM signal chain:**

1. Each symbol determines the push-pull electrical levels for I and Q branches.
2. Gaussian electrical filter (bandwidth = 2×Rs = 20 GHz, order = 7) is applied to the drive signals.
3. Each sub-MZM uses the dual-drive equation:

```
E_out = E_in × cos(π/2 × (U1 - U2 - U0)/Vπ) × exp(-j × π/2 × (U1 + U2)/Vπ)
```

4. I and Q branches are combined in quadrature: `E_total = E_I + E_Q × exp(-j×π/2)`
5. Output power is normalized to `launch_power_dbm = 0 dBm`.

**Default MZM parameters:**
- `v_pi_i = 3.1 V`, `v_pi_q = 3.25 V`
- `null_bias_i = 3.1 V`, `null_bias_q = 3.25 V`
- `quadrature_phase_rad = -π/2`

**Key outputs:**
- `u_tx`: optical field envelope [sqrt(W)] of length `num_symbols × SPS`
- `t`: time vector [s]
- `dt = 1 / (SPS × Rs) = 12.5 ps`

```python
def modulate_optical_signal(
    symbols, sps=8, launch_power_dbm=0.0, symbol_rate=10e9,
    pulse_type='nrz', modulator='ideal', iq_params=None,
    normalize_output_power=True, filter_params=None, ...
) -> tuple[np.ndarray, np.ndarray, float]:
    """Returns (u_tx, t, dt)."""
```

---

### 3. `fibra_monomodo` / `FiberParams` — Fiber Propagation

**Location:** `src/dsp/fiber.py`

**Purpose:** Simulates optical field propagation through single-mode fiber using either a linear transfer function or the Split-Step Fourier Method (SSFM).

#### `FiberParams` dataclass (SMF-28 defaults):

```python
@dataclass
class FiberParams:
    alpha_db: float = 0.2       # Attenuation [dB/km]
    d_1550: float = 16.0        # Chromatic dispersion at 1550 nm [ps/(nm·km)]
    slope_s: float = 0.09       # Dispersion slope [ps/(nm²·km)]
    n2: float = 3.2e-20         # Nonlinear refractive index [m²/W]
    a_eff: float = 70e-12       # Effective area [m²]
    c: float = 3e8              # Speed of light [m/s]
```

#### Derived physical quantities:

- **Attenuation constant:** α_Np = (α_dB / (10 × log₁₀(e))) × 10⁻³ [Np/m]
- **Group velocity dispersion:** β₂ = −(λ² × D) / (2π × c) [s²/m]
- **Third-order dispersion:** β₃ = (D × λ³) / (2π × c)² [s³/m]
- **Nonlinear coefficient:** γ = n₂ × ω_c / (c × A_eff) [1/(W·m)]

#### Mode 0 — Linear (no Kerr effect):

Applies the exact linear transfer function in the frequency domain:

```
H_linear(ω) = exp[j/2 × β₂ × L × ω² − (α/2) × L]
U_out(ω) = U_in(ω) × H_linear(ω)
```

#### Mode 1 — SSFM with Kerr nonlinearity:

Uses the symmetrized Split-Step Fourier Method (SSF) with `num_steps = 50` per span:

```
Dispersive operator: D(ω) = −α/2 + j/2 × β₂ × ω² − j/6 × β₃ × ω³

For each step dz:
  1. Half-step dispersive:   A_mid = IFFT[FFT(A) × exp(D × dz/2)]
  2. Full nonlinear step:    A_nl  = A_mid × exp(j × γ × dz × |A_mid|²)
  3. Half-step dispersive:   A_out = IFFT[FFT(A_nl) × exp(D × dz/2)]
```

**In this notebook:** `regime=0` (linear) is used for all training and evaluation, since the nonlinear distance (`DISTANCE_NONLINEAR_KM = 10000 km`) is set but the `include_nonlinearity=False` flag is passed to `propagate_single_span()`.

```python
def fibra_monomodo(t, u, wavelength, t0, length_km, regime=1,
                   params=None, num_steps=100) -> tuple[np.ndarray, float, float]:
    """Returns (field_out, L_D, L_NL)."""
```

---

### 4. `EDFAIdeal` — Ideal Erbium-Doped Fiber Amplifier

**Location:** `src/dsp/edfa.py` (also re-exported from `src/utils/edfa.py`)

**Purpose:** Ideal noiseless EDFA model with fixed power gain.

**Mathematical model:**

```
E_out(t) = E_in(t) × sqrt(G_linear)   where   G_linear = 10^(G_dB / 10)
```

Equivalently, the field amplitude is multiplied by `10^(G_dB / 20)`.

**Two uses in this notebook:**

1. **In-line EDFA** (after each span + PNN): gain = span_loss_dB + PNN_loss_dB
   - span_loss_dB = α_dB × L_span = 0.2 × 50 = 10 dB
   - PNN_loss_dB = measured as power difference before/after PNN forward pass
2. **Terminal EDFA** (at receiver): fixed gain = 20 dB

```python
@dataclass(frozen=True)
class EDFAIdeal:
    gain_db: float

    @property
    def field_gain(self) -> float:
        return 10.0 ** (self.gain_db / 20.0)

    def amplify(self, optical_field: np.ndarray) -> np.ndarray:
        return np.asarray(optical_field, dtype=complex) * self.field_gain
```

> **Important constraint:** The PNN is passive (no gain), so `PNN_loss_dB ≥ 0` is always enforced. If the PNN appears to amplify (power after > power before), a `ValueError` is raised.

---

### 5. `generate_wiener_phase_noise` — Laser Phase Noise Generator

**Location:** `src/dsp/noise_models.py`

**Purpose:** Generates a Wiener (random walk) stochastic phase noise process representing the intrinsic linewidth of a CW laser.

**Physics:** Spontaneous emission in the laser cavity causes each sample's phase increment to be Gaussian-distributed:

- Variance of phase increment: σ²_Δφ = 2π × Δν × Δt
- Standard deviation: σ_Δφ = √(2π × Δν × Δt)
- Phase accumulation: φ[n] = Σ Δφ[k]  (cumulative sum → Wiener process)

**Applied to both TX laser and local oscillator (LO) independently:**

```python
def generate_wiener_phase_noise(num_samples, dt, linewidth_hz, seed=None) -> np.ndarray:
    """Returns phi: instantaneous phase vector [rad]."""
    if linewidth_hz <= 0:
        return np.zeros(num_samples)
    var_delta_phi = 2.0 * np.pi * linewidth_hz * dt
    sigma_delta_phi = np.sqrt(var_delta_phi)
    delta_phi = rng.normal(loc=0.0, scale=sigma_delta_phi, size=num_samples)
    delta_phi[0] = 0.0
    return np.cumsum(delta_phi)
```

**In this notebook:** Separate phase noise realizations are pre-generated for all 3 linewidths × (train + test) × (TX + LO). The TX phase noise is added to the optical field before transmission, and the LO phase noise is passed to `coherent_detection()`.

---

### 6. `coherent_detection` — Coherent Optical Receiver

**Location:** `src/dsp/receiver.py`

**Purpose:** Simulates intradyne/homodyne coherent beating between the received optical field and the local oscillator in a 90° optical hybrid with balanced photodiodes.

**Mathematical model:**

```
E_LO(t) = sqrt(P_LO) × exp(j × φ_LO(t))
I_out(t) = R × E_rx(t) × conj(E_LO(t))
         = R × sqrt(P_LO) × E_rx(t) × exp(−j × φ_LO(t))
```

This recovers the complex baseband signal where:
- Real part → In-phase (I)
- Imaginary part → Quadrature (Q)

```python
def coherent_detection(optical_field, phase_lo, lo_power_dbm=10.0,
                        responsivity=1.0) -> np.ndarray:
    """Returns r_baseband: complex electrical baseband signal."""
    p_lo_w = 10.0 ** ((lo_power_dbm - 30.0) / 10.0)
    lo_field = np.sqrt(p_lo_w) * np.exp(1j * phase_lo)
    return responsivity * optical_field * np.conj(lo_field)
```

---

### 7. `cd_compensation` — Digital Chromatic Dispersion Compensation

**Location:** `src/dsp/receiver.py`

**Purpose:** Static frequency-domain equalizer that applies the exact inverse transfer function of the fiber to undo chromatic dispersion.

**Transfer function (inverse of fiber linear response):**

```
H_CDC(ω) = exp(+j × β₂/2 × L × ω²)
         = exp(−j × D × λ² × L / (4π × c) × ω²)
```

```python
def cd_compensation(signal, dt, length_km, dispersion_d=16.0,
                    wavelength=1550e-9, c=3e8) -> np.ndarray:
    """Returns signal_cdc: dispersion-compensated signal."""
    freq_grid = np.fft.fftfreq(len(signal), d=dt)
    omega = 2 * np.pi * freq_grid
    beta2 = -(wavelength**2 * dispersion_d * 1e-6) / (2 * np.pi * c)
    h_cdc = np.exp(-(1j / 2.0) * beta2 * length_km * 1e3 * omega**2)
    return np.fft.ifft(np.fft.fft(signal) * h_cdc)
```

---

### 8. `sample_to_symbols` — Downsampler

**Location:** `src/dsp/receiver.py`

**Purpose:** Decimates the oversampled signal (SPS = 8) to 1 sample per symbol at the ideal sampling instant.

```python
def sample_to_symbols(signal, sps, sample_offset=None) -> np.ndarray:
    """Decimates from SPS samples/symbol to 1 sample/symbol."""
    if sample_offset is None:
        sample_offset = sps // 2  # center of pulse: sample 4 of 8
    return signal[sample_offset::sps]
```

---

### 9. `demap_qpsk` — QPSK Bit Demapper

**Location:** `src/dsp/receiver.py`

**Purpose:** Euclidean (nearest-neighbor) QPSK demapping using quadrant decisions.

**Decision rules:**
- Re(symbol) ≥ 0 → bit_I = 0;  Re(symbol) < 0 → bit_I = 1
- Im(symbol) ≥ 0 → bit_Q = 0;  Im(symbol) < 0 → bit_Q = 1

```python
def demap_qpsk(symbols_rx) -> np.ndarray:
    """Returns bits_rx: 1D binary vector of size 2 × len(symbols_rx)."""
    bit_i = np.where(np.real(symbols_rx) >= 0.0, 0, 1)
    bit_q = np.where(np.imag(symbols_rx) >= 0.0, 0, 1)
    bits_rx[0::2] = bit_i
    bits_rx[1::2] = bit_q
    return bits_rx
```

---

### 10. `calculate_ber` — Bit Error Rate

**Location:** `src/utils/metrics.py`

```python
def calculate_ber(bits_tx, bits_rx) -> float:
    """BER = number of bit errors / total bits."""
    min_len = min(len(bits_tx), len(bits_rx))
    errors = np.sum(bits_tx[:min_len] != bits_rx[:min_len])
    return float(errors) / float(min_len)
```

**BER floor:** With 4096 test symbols × 2 bits/symbol = 8192 bits → BER floor = 1/8192 ≈ 1.22×10⁻⁴.

---

### 11. `calculate_evm` — Error Vector Magnitude

**Location:** `src/utils/metrics.py`

**Purpose:** Computes EVM RMS (%) relative to the nearest ideal QPSK constellation point.

**Algorithm:**
1. Normalize received symbols by their average power.
2. For each received symbol, find the nearest ideal QPSK point (optional phase rotation correction).
3. Compute EVM_RMS = √(mean(|error_vector|²)) × 100%

**Ideal QPSK constellation:** {(±1 ± j)/√2}

```python
def calculate_evm(rx_symbols, ideal_constellation=None,
                   correct_phase_rotation=False) -> float:
    """Returns EVM RMS in %."""
    if ideal_constellation is None:
        ideal_constellation = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2.0)
    # normalize
    p_rx = np.mean(np.abs(rx_symbols)**2)
    norm_rx = rx_symbols / np.sqrt(p_rx + 1e-30)
    # optional: correct constant phase rotation
    if correct_phase_rotation:
        ...
    # find nearest ideal point for each symbol
    distances = np.abs(norm_rx[:, None] - norm_ideal[None, :])
    s_ref = norm_ideal[np.argmin(distances, axis=1)]
    # EVM RMS
    error_vector = norm_rx - s_ref
    return float(np.sqrt(np.mean(np.abs(error_vector)**2)) * 100.0)
```

**In this notebook:** `CORRECT_PHASE_ROTATION = False` — no phase rotation correction is applied. This means any constant carrier-phase offset (from the coherent beating LO) degrades EVM.

---

### 12. `PhotonicNeuralNetwork` — PNN Model

**Location:** `src/models/pnn.py` (also re-exported from `src/utils/pnn_model.py`)

**Purpose:** Theoretical model of a silicon photonic TDL (Tapped Delay Line) FIR filter with quadratic photodetection.

#### Mathematical formulation:

**Channel delays:**
```
τ_i = i × Δt,   i = 0, 1, ..., N-1
```

**Complex weights:**
```
w_i = a_i × exp(j × φ_i) / N   (with normalize_power=True)
```
where a_i ∈ [0, 1] (transmittance) and φ_i ∈ ℝ (phase shift).

**Optical combiner (interferometric sum):**
```
E_sum(t) = Σᵢ wᵢ × E_in(t − i × Δt)
```

**Optical frequency response:**
```
H(f) = Σᵢ wᵢ × exp(−j × 2π × f × i × Δt)
```

**Quadratic detection (photonic activation function):**
```
I_out(t) = R × |E_sum(t)|²
```

#### Forward pass implementation:

For each channel `i`:
- If the delay `τ_i/dt` is an integer: direct sample shift (no distortion)
- If fractional: apply frequency-domain phase shift `exp(−j × 2π × f × τ_i)`

The PNN can return both the photocurrent AND the optical field `E_sum(t)`, which is used by the notebook to chain the equalized field through EDFA and the next span.

```python
class PhotonicNeuralNetwork:
    def __init__(self, num_channels=8, base_delay_ps=25.0,
                 amplitude_weights=None, phase_weights=None,
                 normalize_power=True, responsivity=1.0):
        ...

    @property
    def complex_weights(self) -> np.ndarray:
        w = self.amplitude_weights * np.exp(1j * self.phase_weights)
        if self.normalize_power:
            w = w / self.num_channels  # divides by 4 in this notebook
        return w

    def forward(self, optical_field, dt_s,
                return_optical_field=False):
        """Returns (photocurrent,) or (photocurrent, E_sum)."""
        E_sum = sum(w_i * delay(E_in, tau_i))
        photocurrent = R * |E_sum|²
        if return_optical_field:
            return photocurrent, E_sum
        return photocurrent
```

**In this notebook:**
- N = 4 channels, Δt = 25 ps, normalize_power = True → each weight divided by 4
- Total delay range: 0, 25, 50, 75 ps (= 0, 2, 4, 6 samples at 80 GHz)
- Called with `return_optical_field=True` to get E_sum for chaining

---

### 13. `particle_swarm_optimize` — PSO Optimizer

**Location:** `scripts/optimize_pnn_pso.py`

**Purpose:** Minimizes a continuous objective function using Particle Swarm Optimization (PSO) with boundary handling and periodic dimension support (for phase variables).

#### Algorithm:

**Initialization:**
- Randomly place `swarm_size` particles in the search space within bounds.
- Random initial velocities ∈ [−0.1×span, +0.1×span].

**Velocity update (each iteration):**
```
v[t+1] = w × v[t] 
        + c1 × r1 × (personal_best − position)
        + c2 × r2 × (global_best − position)
```
where:
- `w = 0.72` (inertia)
- `c1 = c2 = 1.49` (cognitive and social coefficients)
- `r1, r2 ~ Uniform(0,1)` (random factors)

**Position update:**
```
position[t+1] = position[t] + v[t+1]
```
- Non-periodic dimensions: clipped to bounds
- Periodic dimensions (phase): wrapped modulo the period

**Periodic wrapping for phase dimensions [−π, π]:**
```
delta = (target − current + π) % (2π) − π   # shortest-arc direction
```

**Output dictionary:**
```python
{
    "position": best_position,          # full parameter vector
    "amplitude_weights": a[0:N],        # first N entries
    "phase_weights": phi[N:2N],         # last N entries
    "best_fitness": best_value,
    "best_fitness_history": [f_0, f_1, ..., f_T],
    "evaluations": swarm_size * (1 + max_iterations),
    "iterations": max_iterations,
    "swarm_size": swarm_size,
}
```

**In this notebook:**
- `swarm_size = 30`, `max_iterations = 50`
- `periodic_dimensions = range(4, 8)` (the 4 phase dimensions)
- Total evaluations per PSO run: 30 × (1 + 50) = 1,530
- Total PSO runs: 2 chains × 200 spans × 3 linewidths = **1,200 runs**
- Total objective evaluations: 1,200 × 1,530 = **1,836,000**

```python
def particle_swarm_optimize(objective, bounds, *, swarm_size=40,
                             max_iterations=60, inertia=0.72,
                             cognitive=1.49, social=1.49,
                             periodic_dimensions=(), seed=42):
    """Minimize objective using PSO. Returns dict with best solution."""
```

---

## Inline Pipeline Functions (Defined in Notebook)

### `mean_power_dbm(field)`

```python
def mean_power_dbm(field):
    """Mean optical envelope power in dBm."""
    power_w = np.mean(np.abs(field) ** 2)
    return 10.0 * np.log10(power_w / 1e-3 + 1e-30)
```

---

### `propagate_single_span(field_in, t, span_length_km, include_nonlinearity, num_steps=50)`

```python
def propagate_single_span(field_in, t, span_length_km, include_nonlinearity, num_steps=50):
    """Propagates the field through a single fiber span.
    
    regime = 1 (SSFM) if include_nonlinearity else 0 (linear).
    Uses global FIBER_PARAMS (SMF-28), WAVELENGTH=1550nm, SYMBOL_RATE=10GHz.
    Returns: field_out (no EDFA amplification here)
    """
    regime = 1 if include_nonlinearity else 0
    field_out, _, _ = fibra_monomodo(t, field_in, wavelength=WAVELENGTH,
                                       t0=1.0/SYMBOL_RATE, length_km=span_length_km,
                                       regime=regime, params=FIBER_PARAMS,
                                       num_steps=num_steps)
    return field_out
```

---

### `propagate_distributed_link(field_in, t, n_spans, pnn_chain=None, include_nonlinearity=False, ...)`

This is the core propagation function. It iterates through `n_spans`, applying `[Fiber → PNN_k → EDFA_k]` per span.

```
Input field (at Tx)
  │
  ▼
  For k = 0, 1, ..., n_spans-1:
    1. fibra_monomodo(50 km, linear) → field_after_fiber
    2. If pnn_chain[k] is not None:
         a. Set PNN weights: (a[0..3], φ[0..3])
         b. pnn.forward(field_after_fiber) → (photocurrent, E_sum)
         c. Measure PNN_loss_dB = P_before - P_after [dBm]
         d. inline_EDFA gain = span_loss_dB + PNN_loss_dB
       Else:
         d. inline_EDFA gain = span_loss_dB (10 dB)
    3. inline_edfa.amplify(field) → field (restored to ~launch power)
  │
  ▼
Output field (at end of n_spans)
```

**Key point:** The inline EDFA gain is adaptive — it always compensates the exact span loss plus the PNN's insertion loss, keeping the power constant regardless of PNN weight settings.

```python
def propagate_distributed_link(field_in, t, n_spans, pnn_chain=None,
                                include_nonlinearity=False,
                                span_length_km=50.0, num_steps_per_span=50):
    """Propagates through n_spans with [Fiber → PNN_k → EDFA_k] per span.
    Returns: field (output optical field after all spans)
    """
```

---

### `receiver_edfa_and_attenuator(field, target_dbm=0.0, edfa_gain_db=20.0)`

```python
def receiver_edfa_and_attenuator(field, target_dbm=0.0, edfa_gain_db=20.0):
    """Terminal EDFA (20 dB) followed by VOA to set received power.
    
    1. Amplify: field × 10^(20/20) = field × 10
    2. Measure power after EDFA
    3. Attenuate to target_dbm = 0 dBm
    
    Returns: (received_field, attenuation_db, power_after_edfa_dbm)
    Raises ValueError if 20 dB gain is insufficient to reach 0 dBm.
    """
```

---

### `compute_objective_value(rx_symbols, bits_ref, objective_type='EVM')`

```python
def compute_objective_value(rx_symbols, bits_ref, objective_type='EVM'):
    """PSO fitness function: returns EVM (%) or BER.
    
    'EVM': calculate_evm(rx_symbols, correct_phase_rotation=False)
    'BER': demap_qpsk(rx_symbols) → calculate_ber(bits_ref, bits_rx)
    """
```

**Why EVM is preferred over BER for PSO:**
- EVM is a continuous differentiable function → smooth optimization landscape
- BER is piecewise constant → PSO particles receive no gradient signal when BER = 0 or saturated
- EVM provides implicit guidance even when all bits are correct

---

### `detect_and_evaluate(field_rx, phase_lo, bits_ref, symbols_ref, dt_val=DT, cdc_distance_km=None)`

```python
def detect_and_evaluate(field_rx, phase_lo, bits_ref, symbols_ref,
                         dt_val=DT, cdc_distance_km=None):
    """Full receiver pipeline for evaluation:
    1. coherent_detection(field_rx, phase_lo) → r_bb
    2. If cdc_distance_km > 0: cd_compensation(r_bb, length_km=cdc_distance_km)
    3. sample_to_symbols(r_bb, sps=8) → rx_symbols
    4. calculate_evm(rx_symbols) → evm_val
    5. demap_qpsk(rx_symbols) → detected_bits
    6. calculate_ber(bits_ref, detected_bits) → ber_val
    
    Returns: (ber_val, evm_val, rx_symbols)
    """
```

---

## Training Algorithm: Stage-wise (Span-by-Span) PSO

The training avoids the curse of dimensionality of a joint 1600-parameter optimization (200 spans × 8 params each) by training **one PNN at a time, in sequential order from span 1 to span 200**.

### Why stage-wise training works:

- Span k depends only on the output field of span k−1.
- The fiber output of span k is computed **once** before PSO starts.
- The PSO objective for span k only executes: PNN_k forward pass + coherent detection + EVM — this is very fast per particle evaluation.
- Once trained, PNN_k's weights are **frozen** and the equalized field passes through EDFA_k to become the input of span k+1.

### Chain 1: Distributed PNNs (No Digital CDC)

Each PNN_k is optimized to minimize EVM **without** digital CDC at the receiver:

```
PSO objective for span k (No CDC):
  1. Set PNN weights: position[:4] → amplitudes, position[4:] → phases
  2. PNN.forward(field_after_fiber_k) → E_sum
  3. coherent_detection(E_sum, phi_LO) → r_bb
  4. sample_to_symbols(r_bb, sps=8) → rx_symbols
  5. return calculate_evm(rx_symbols[:1024])
```

The PNN must learn to compensate all accumulated chromatic dispersion from spans 1..k using **only optical processing**.

### Chain 2: Distributed PNNs + Digital CDC (Joint Training)

Each PNN_k is optimized jointly **assuming digital CDC will be applied** at the receiver for the cumulative distance d_k = k × 50 km:

```
PSO objective for span k (Joint CDC):
  1. Set PNN weights
  2. PNN.forward(field_after_fiber_k) → E_sum
  3. coherent_detection(E_sum, phi_LO) → r_bb
  4. cd_compensation(r_bb, length_km=k × 50) → r_comp
  5. sample_to_symbols(r_comp, sps=8) → rx_symbols
  6. return calculate_evm(rx_symbols[:1024])
```

Here, the PNN supplements the digital CDC by handling residual distortions (nonlinearity, transient effects, amplitude filtering effects).

### Pseudocode for the full training loop:

```
For each linewidth in [0 Hz, 100 kHz, 1 MHz]:
    Apply TX phase noise to u_train
    
    # Chain 1: No CDC
    current_field = u_train_with_phase_noise
    for span_idx in 0..199:
        field_after_fiber = fibra_monomodo(current_field, 50 km, linear)
        
        pso_result = PSO(
            objective = EVM after: PNN + coherent_detection,
            bounds = [(0,1)]×4 + [(-π,π)]×4,
            swarm_size=30, max_iterations=50
        )
        
        # Freeze this span's weights and advance the field
        PNN.set_weights(pso_result.amplitudes, pso_result.phases)
        _, E_sum = PNN.forward(field_after_fiber)
        inline_edfa_gain = 10 dB + PNN_loss_dB
        current_field = EDFA(E_sum, gain=inline_edfa_gain)
    
    # Chain 2: Joint CDC (same structure, objective includes cd_compensation)
    ...
```

**Training duration (actual execution):** ~11 minutes total for 2 chains × 3 linewidths × 200 spans each.

### PSO Seed Strategy

Each PSO run uses a deterministic seed to ensure reproducibility:
```
# Chain 1 (No CDC):   seed = PNN_TRAIN_SEED + 10000 × lw_idx + 100 × span_idx
# Chain 2 (Joint CDC): seed = PNN_TRAIN_SEED + 50000 + 10000 × lw_idx + 100 × span_idx
```

---

## Evaluation Pipeline (Section 6 of Notebook)

After training, all 4 scenarios are evaluated on the **test set** (4096 symbols, different seed from training). The evaluation sweeps from 50 km to 10,000 km in 50 km steps (200 evaluation points per linewidth).

For each distance d = k × 50 km:

```
1. Apply TX phase noise to u_test
2. propagate_distributed_link(u_test, n_spans=k, pnn_chain=...) → field_rx
3. receiver_edfa_and_attenuator(field_rx) → field_rx_normalized
4. detect_and_evaluate(field_rx_normalized, phi_LO, ..., cdc_distance_km=...) → (BER, EVM, symbols_rx)
```

**Scenario-specific settings:**

| Scenario | `pnn_chain` argument | `cdc_distance_km` |
|----------|----------------------|-------------------|
| 1 (No PNN, No CDC) | `None` | `None` |
| 2 (No PNN, with CDC) | `None` | `d` (cumulative distance) |
| 3 (Distributed PNNs, No CDC) | `trained_chains_no_cdc[lw_key][:k]` | `None` |
| 4 (Distributed PNNs + CDC) | `trained_chains_joint_cdc[lw_key][:k]` | `d` |

---

## Actual Training Results (From Notebook Execution)

The table shows the **final stage fitness (EVM%)** at span 200 (10,000 km):

| Linewidth | Chain 1 (No CDC) Final EVM | Chain 2 (Joint CDC) Final EVM |
|-----------|---------------------------|-------------------------------|
| 0 Hz | 84.50% | 99.12% |
| 100 kHz | 95.83% | 97.77% |
| 1 MHz | 91.25% | 97.06% |

> **Interpretation:** EVM > 100% is physically possible (it represents normalized error magnitude relative to the ideal constellation), but values approaching or exceeding 100% indicate the signal is barely recoverable or unrecoverable at that stage.

---

## Expected Outputs

### Output Plots Generated

1. **Stage-wise Training Fitness per Span** (`mult_pnn_stage_fitness_per_span.png`)
   - 3 subplots (one per linewidth)
   - X-axis: Cumulative distance [km] from 50 km to 10,000 km
   - Y-axis: Best PSO fitness (EVM %)
   - 3 curves per subplot:
     - CDC only (reference, no PNN)
     - PNN stage fitness without CDC
     - PNN stage fitness with joint CDC

2. **BER vs Distance** (from evaluation section)
   - X-axis: Distance [km]
   - Y-axis: BER (log scale, floor = 1.22×10⁻⁴)
   - 4 curves (one per scenario)
   - Repeated for 3 linewidths

3. **EVM vs Distance** (from evaluation section)
   - Same structure as BER plots, Y-axis: EVM [%]

4. **Constellation diagrams** (at selected distances)
   - Received QPSK symbols (I vs Q scatter)
   - One per scenario, at short, medium, and long distances

### Expected Behavior per Scenario

**Scenario 1 (No PNN, No CDC):** BER rises rapidly with distance due to uncompensated chromatic dispersion. For 10 GBaud QPSK on SMF-28, the dispersion length is L_D ≈ T²₀/|β₂| ≈ several hundred km. Beyond ~100-200 km, BER should approach 0.5 (random) and EVM approaches ~100%.

**Scenario 2 (No PNN, Digital CDC):** Near-ideal performance at all distances since digital CDC exactly cancels linear chromatic dispersion. BER floor limited by phase noise (scales with linewidth × distance). For 0 Hz linewidth, BER should remain at floor across all distances.

**Scenario 3 (Distributed PNNs, No CDC):** Performance is an open research question. The PNN with 4 channels and 25 ps delays has a maximum optical delay range of 75 ps. The accumulated dispersion at distance d is Δτ = D × L × Δλ, where Δλ is the signal bandwidth. For 10 GBaud QPSK with ~20 GHz bandwidth, Δλ ≈ 0.16 nm, giving Δτ = 16 × 50k × 0.16 ≈ 128 ps/span cumulative. This suggests a single 4-channel PNN cannot fully compensate even one span's dispersion, but may partially reduce it.

**Scenario 4 (Distributed PNNs + Digital CDC):** Should approach Scenario 2 performance, with PNNs potentially reducing residual impairments that the linear CDC cannot handle.

---

## File Organization

```
pnn_coerente/
├── notebooks/
│   └── exploracao_mult_pnn_dispersao_edfa_ob_evm.ipynb  (this algorithm)
├── src/
│   ├── dsp/
│   │   ├── transmitter.py    (generate_qpsk_symbols, modulate_optical_signal)
│   │   ├── fiber.py          (FiberParams, fibra_monomodo)
│   │   ├── edfa.py           (EDFAIdeal - canonical)
│   │   ├── noise_models.py   (generate_wiener_phase_noise)
│   │   └── receiver.py       (coherent_detection, cd_compensation, 
│   │                          sample_to_symbols, demap_qpsk)
│   ├── models/
│   │   └── pnn.py            (PhotonicNeuralNetwork - canonical)
│   └── utils/
│       ├── edfa.py           (backward compat re-export of EDFAIdeal)
│       ├── metrics.py        (calculate_ber, calculate_evm)
│       └── pnn_model.py      (backward compat re-export of PhotonicNeuralNetwork)
├── scripts/
│   └── optimize_pnn_pso.py   (particle_swarm_optimize)
└── artifacts/
    ├── figures/mult_pnn_edfa_evm/   (output plots)
    └── models/
        └── pnn_mult_span_pso_treinamento.json  (saved PNN weights)
```

---

## Key Physical Insights

1. **Dispersion accumulation is linear:** D × L × Δλ grows linearly with distance. For 10,000 km at D=16 ps/(nm·km), total accumulated dispersion = 160,000 ps/nm.

2. **PNN as an optical FIR filter:** The 4-channel TDL with 25 ps base delay implements a 4-tap FIR filter with 75 ps total delay. This is equivalent to a bandwidth-limited equalizer that can partially compensate dispersion.

3. **Stage-wise training efficiency:** Instead of a 1600-parameter joint optimization, the stage-wise approach reduces each PSO run to 8 parameters, making it tractable in ~11 minutes.

4. **Phase noise vs. distance trade-off:** Wider linewidth × longer distance = more accumulated phase variance. At 1 MHz linewidth and 10,000 km, the accumulated phase noise may overwhelm any dispersion compensation gains.

5. **Ideal EDFA assumption:** The model ignores ASE noise from EDFAs. In practice, each EDFA adds optical noise proportional to its gain (noise figure ~4-6 dB), which is a major degradation mechanism at long distances. This simulation represents an upper bound on PNN performance.

---

## Scenario Estimation for Different Configurations

### How to Predict Performance for a New Configuration:

| Parameter | Effect on Performance |
|-----------|----------------------|
| More PNN channels (N > 4) | Longer delay range → better dispersion coverage per span |
| Larger base delay (Δt > 25 ps) | Each tap covers more delay, but coarser resolution |
| Shorter spans (< 50 km) | Less dispersion per span → easier for PNN to compensate |
| Lower launch power | Less Kerr nonlinearity but more vulnerable to noise |
| Higher launch power | More nonlinear distortion, diminishing returns |
| BER objective instead of EVM | Slower optimization convergence (plateau problem) |
| More PSO particles/iterations | Better optimization quality, longer training time |

### Scaling Analysis:

- **Training time scales as:** O(N_spans × N_linewidths × N_chains × swarm_size × iterations × objective_cost)
- Current: 200 × 3 × 2 × 30 × 50 × (fast Python call) ≈ 11 min
- Doubling channels doubles PNN `forward()` cost (linear)
- Doubling PSO size doubles evaluations linearly

---

*This document was generated from: `notebooks/exploracao_mult_pnn_dispersao_edfa_ob_evm.ipynb`*
*External source files referenced: `src/dsp/transmitter.py`, `src/dsp/fiber.py`, `src/dsp/edfa.py`, `src/dsp/noise_models.py`, `src/dsp/receiver.py`, `src/utils/metrics.py`, `src/models/pnn.py`, `scripts/optimize_pnn_pso.py`*
