# Coherent Optical Transmission Simulator & Silicon Photonic Neural Network (PNN)

## Overview

This repository provides an end-to-end simulation framework for high-speed coherent optical communication links integrated with **Silicon Photonic Neural Networks (PNNs)** and **Digital Signal Processing (DSP)** algorithms.

Developed as part of research in integrated photonics and optical equalization (UFES / IN-Foton), the framework models 10 GBaud QPSK and 64 GBaud optical transmission across single-mode optical fibers (SSMF). It investigates the performance of **Photonic Neural Networks** operating as **optical Fractionally-Spaced Equalizers (FSE)** to mitigate Chromatic Dispersion (CD) and nonlinear distortions both at the receiver and distributed *in-line* across multi-span amplified links.

The simulation toolset models the full optical signal chain:
1. **Transmitter (TX)**: QPSK mapping, electrical drive filtering, and dual-drive analytical IQ Mach-Zehnder Modulators (IQ-MZM) with zero-chirp push-pull operation.
2. **Optical Channel (SSMF)**: Single-Mode Fiber propagation modeled via the Split-Step Fourier Method (SSFM) incorporating Chromatic Dispersion (CD), attenuation, third-order dispersion, and Kerr self-phase modulation (SPM).
3. **Optical Amplification (EDFA)**: Ideal Erbium-Doped Fiber Amplifiers with configurable gain to compensate for fiber span attenuation and passive PNN insertion loss.
4. **Photonic Neural Network (PNN)**: Integrated Silicon-on-Insulator (SOI) Tapped Delay Line (TDL) optical FIR filter with complex-valued weights a_k · e^(j·φ_k), passive 1/N power normalization, and quadratic photodetector activation.
5. **Coherent Receiver (RX)**: 90-degree optical hybrid beating with a Local Oscillator (LO) under Wiener process phase noise and phase correlation coupling (ρ).
6. **Receiver DSP**: Frequency-domain Digital Chromatic Dispersion Compensation (CDC), Joint Optical-Digital Equalization H_joint(f), symbol slicing, and metrics calculation (BER & EVM).
7. **Optimization (PSO)**: Particle Swarm Optimization for automated tuning of PNN optical weights.

---

## Project Directory Structure

```
pnn_coerente/
├── README.md                                    # Root documentation and project overview
├── requirements.txt                            # Python environment dependencies
├── .gitignore                                  # Git ignore specifications
│
├── notebooks/                                  # Research & exploratory Jupyter Notebooks
│   ├── exploracao_mult_pnn_dispersao_edfa_ob_evm.ipynb    # Distributed in-line PNNs over multi-span links (up to 10,000 km)
│   ├── exploracao_pnn_dispersao_edfa_correlacao_fase.ipynb # PNN under correlated laser phase noise (rho) and linewidth sweeps
│   ├── simulacao_coerente_qpsk.ipynb           # Baseline QPSK coherent transmission with ideal modulation and CDC
│   ├── simulacao_coerente_qpsk_iq_mzm.ipynb     # Analytical IQ-MZM transmitter with distance and linewidth sweeps
│   ├── teste_modulador_iq_mzm.ipynb            # Static optical transfer characterization (E_out vs. drive voltage)
│   ├── teste_modelo_fibra_scm_fechamentoCW.ipynb # SSFM fiber model validation against MATLAB reference scripts
│   ├── ook_mzm_pushpull_ber_distancia.ipynb    # Direct detection OOK push-pull link with PNN dispersion mitigation
│   ├── exploracao_pnn_dispersao_edfa.ipynb     # Single-span/multi-span PNN with fiber loss and EDFA gain compensation
│   ├── avaliacao_equalizacao_conjunta_pnn_cdc.ipynb # Comparative study: Conventional CDC vs. Joint Optical-Digital CDC
│   ├── ber_distancia_pnn_cdc_joint_cdc.ipynb   # BER vs. distance (50-1000 km) across 7 equalization strategies
│   └── evm_pnn_cdc_treinamento_conjunto.ipynb  # EVM-driven vs. BER-driven PSO training and constellation synthesis
│
├── src/                                        # Core simulator source packages
│   ├── dsp/                                    # Digital & Optical Signal Processing modules
│   │   ├── transmitter.py                      # QPSK mapping, pulse shaping, and IQ-MZM modulator
│   │   ├── fiber.py                            # FiberParams dataclass, SSFM, and linear transfer function
│   │   ├── edfa.py                             # Ideal EDFA model with field gain calculation
│   │   ├── noise_models.py                     # Wiener phase noise and correlated phase noise (rho)
│   │   ├── receiver.py                         # Coherent detection (90° hybrid), CDC, downsampling, demapping
│   │   └── joint_pnn_cdc_equalizer.py          # Joint Optical-Digital Inversion Equalizer H_joint(f)
│   ├── models/                                 # Photonic hardware models
│   │   └── pnn.py                              # PhotonicNeuralNetwork class (SOI TDL FIR filter)
│   └── utils/                                  # Utility functions and metrics
│       ├── edfa.py                             # Backward compatibility re-export for EDFAIdeal
│       ├── metrics.py                          # BER and decision-directed EVM calculations
│       ├── pnn_model.py                        # Backward compatibility re-export for PNN
│       └── filtros.py                          # Electrical drive filters (Gaussian, Raised Cosine, Butterworth)
│
├── scripts/                                    # Standalone execution & optimization scripts
│   ├── run_coherent_simulation.py              # Automated simulation pipeline execution
│   ├── run_scm_fechamento_cw.py                # SCM fiber validation runner
│   └── optimize_pnn_pso.py                     # Particle Swarm Optimization (PSO) script for PNN weights
│
├── config/                                     # System configuration files
│   └── system_params.yaml                      # Global baseline reference system parameters
│
├── artifacts/                                  # Generated benchmark outputs, plots, and models
│   ├── figures/                                # Saved visualization charts & constellation diagrams
│   │   ├── joint_pnn_cdc_comparison.png        # Comparative performance curves (BER/EVM)
│   │   ├── qpsk_non_ideal_receiver.png         # Non-ideal coherent receiver constellation plots
│   │   └── mult_pnn_edfa_evm/                  # Output figures for distributed multi-span PNNs
│   ├── reports/                                # Technical markdown documentation & source listings
│   │   ├── analise_desempenho_pnn_cdc_pso.md   # Performance analysis report of PNN + CDC + PSO
│   │   ├── relatorio_algoritmos_implementados.md # Comprehensive technical report of algorithms
│   │   ├── relatorio_ber_distancia_pnn_cdc_joint_cdc.md # Benchmark report of 7 equalization strategies
│   │   ├── algoritmos_codigo_fonte.md          # Full source code listings and documentation
│   │   └── joint_pnn_cdc_reference.md          # Theoretical reference for Joint Equalization
│   ├── models/                                 # Saved PNN weight configurations (JSON)
│   │   └── pnn_mult_span_pso_treinamento.json  # Trained weights across 200 spans
│   └── modules/                                # Modular executable Python deliverables
│       ├── optical_metrics.py                  # Standalone metrics module (BER & EVM)
│       ├── coherent_receiver_non_ideal.py      # Non-ideal coherent receiver module
│       ├── phase_noise_model.py                # Standalone Wiener phase noise module
│       ├── pnn_model-v3.py                     # Enhanced Photonic Neural Network model
│       └── joint_pnn_cdc_equalizer.py          # Standalone joint equalizer module
│
├── data/                                       # Input data vectors and waveforms
└── matlab/                                     # Legacy MATLAB reference scripts
```

---

## Detailed Notebook Descriptions

### 1. `exploracao_mult_pnn_dispersao_edfa_ob_evm.ipynb`
* **Focus**: Investigation of **Distributed Optical PNNs** in multi-span EDFA-amplified QPSK links over long-haul distances up to **10,000 km** (200 spans × 50 km).
* **Architecture**: Evaluates an in-line configuration `[Fiber_k (50 km) -> PNN_k -> EDFA_k]` for k = 1 ... 200.
* **Key Scenarios Evaluated**:
  1. *Uncompensated Link* (Baseline)
  2. *Conventional Digital CDC*
  3. *Distributed Optical PNNs alone* (tests if optical processing can replace digital CDC)
  4. *Distributed PNNs + Digital CDC Jointly Optimized*
* **Optimization**: Uses stage-wise (span-by-span) Particle Swarm Optimization (PSO) driven by EVM RMS minimization.

### 2. `exploracao_pnn_dispersao_edfa_correlacao_fase.ipynb`
* **Focus**: Evaluates PNN performance under **correlated laser phase noise** between the Transmitter (TX) and Local Oscillator (LO) across different correlation factors (ρ ∈ [0.0, 1.0]) and laser linewidths (Δν = 0 Hz, 100 kHz, 1 MHz).
* **Key Findings**: Demonstrates how phase correlation reduces phase variance during intradyne beating and analyzes the interaction between optical PNN filtering and carrier phase jitter.

### 3. `simulacao_coerente_qpsk.ipynb`
* **Focus**: Baseline implementation of a 10 GBaud QPSK coherent link over SSMF using ideal complex modulation and frequency-domain CDC at the receiver.
* **Purpose**: Serves as the theoretical reference for back-to-back (B2B) transmission and uncompensated chromatic dispersion penalty.

### 4. `simulacao_coerente_qpsk_iq_mzm.ipynb`
* **Focus**: Incorporates an **analytical IQ Mach-Zehnder Modulator (IQ-MZM)** into the coherent pipeline.
* **Analysis**: Explores distance sweeps (50 to 1000 km), V_π drive voltage variations, zero-chirp push-pull driving conditions, and laser phase noise impact on receiver constellations.

### 5. `teste_modulador_iq_mzm.ipynb`
* **Focus**: Static optical transfer characterization of the IQ-MZM model.
* **Outputs**: Generates E_out vs. V_drive curves, verifying null bias carrier suppression (U_0 = V_π), quadrature phase shift (φ_quad = -π/2), and push-pull linearity.

### 6. `teste_modelo_fibra_scm_fechamentoCW.ipynb`
* **Focus**: Numerical validation of the Python Split-Step Fourier Method (SSFM) against legacy MATLAB reference models.
* **Verification**: Verifies eye diagrams, pulse broadening, and Chromatic Dispersion accumulation across multi-span SSMF links.

### 7. `ook_mzm_pushpull_ber_distancia.ipynb`
* **Focus**: Direct Detection (IM-DD) On-Off Keying (OOK) link with push-pull MZM driving, PIN photodiode, and a PNN equalizer.
* **Application**: Evaluates PNN dispersion mitigation in short-reach datacom/access networks without coherent receivers.

### 8. `exploracao_pnn_dispersao_edfa.ipynb`
* **Focus**: Single-span and multi-span link analysis combining SSMF attenuation, PNN optical equalization, and inline EDFA gain compensation.
* **Key Insight**: Quantifies the passive insertion loss of PNN chips and verifies EDFA gain matching.

### 9. `avaliacao_equalizacao_conjunta_pnn_cdc.ipynb`
* **Focus**: Comparative study between conventional Digital CDC H_CDC(f) and Joint Optical-Digital Equalization H_joint(f).
* **Core Contribution**: Demonstrates how H_joint(f) cancels both fiber chromatic dispersion and the spectral transfer response introduced by the PNN's optical tap weights.

### 10. `ber_distancia_pnn_cdc_joint_cdc.ipynb`
* **Focus**: Comprehensive sweep of Bit Error Rate (BER) versus distance (50 km to 1000 km) across **7 distinct equalization strategies** and 3 laser linewidths (0 Hz, 100 kHz, 1 MHz).
* **Strategies Compared**: Uncompensated, CDC Digital, PNN Alone, PNN + CDC, Joint CDC, PNN (EVM-trained), Joint CDC (EVM-trained).

### 11. `evm_pnn_cdc_treinamento_conjunto.ipynb`
* **Focus**: Methodological comparison between **BER-driven PSO** and **EVM-driven PSO** training for PNN weights.
* **Key Result**: Demonstrates that continuous EVM RMS minimization eliminates optimization plateaus, resulting in faster PSO convergence and superior constellation quality.

---

## Deep Dive into Core Algorithms & Physical Models

### 1. Mach-Zehnder Modulators (IQ-MZM & Drive Filtering)

The optical transmitter models an analytical **IQ Mach-Zehnder Modulator (IQ-MZM)** (`src/dsp/transmitter.py`), ported from electro-optic physics models:

#### **Sub-MZM Transfer Function (Dual-Drive)**
Each nested sub-MZM (for In-Phase I and Quadrature Q branches) applies the dual-drive electro-optic phase modulation equation:

$$
E_{out}(t) = E_{in}(t) \cdot \cos\left( rac{\pi}{2} rac{U_1(t) - U_2(t) - U_0}{V_\pi} ight) \cdot \exp\left( -j rac{\pi}{2} rac{U_1(t) + U_2(t)}{V_\pi} ight)
$$

* U_1(t), U_2(t): Driving voltages applied to the upper and lower waveguide arms.
* V_π: Half-wave switching voltage (typically 3.1 V to 3.25 V).
* U_0 = V_π: Null-bias condition for suppressed-carrier optical modulation.

#### **Push-Pull Zero-Chirp Driving**
Applying anti-phase drive signals U_1(t) = -U_2(t) = V(t)/2 sets the common-mode phase term to zero:

$$
\exp\left( -j rac{\pi}{2} rac{U_1(t) + U_2(t)}{V_\pi} ight) = \exp(0) = 1
$$

This completely eliminates transmitter residual chirp, producing pure double-sideband suppressed-carrier (DSB-SC) modulation.

#### **Quadrature IQ Combination**
The I and Q branch fields are recombined with a static π/2 phase offset:

$$
E_{TX}(t) = rac{1}{2} \left[ E_I(t) + E_Q(t) \cdot e^{j \phi_{quad}} ight], \quad 	ext{where } \phi_{quad} = -rac{\pi}{2}
$$

#### **Electrical Drive Filtering**
Before modulation, baseband electrical driving signals pass through a digital filter (`src/utils/filtros.py`) simulating DAC bandwidth limits:
* **Super-Gaussian Filter**: H(f) = exp(-ln(2) · (f / f_c)^(2n)) with cutoff f_c = 2 · R_s = 20 GHz and order n = 7.
* **Raised Cosine & Butterworth**: Alternative user-selectable electrical response filters.

---

### 2. Single-Mode Fiber (SSMF) Propagation Model

Fiber propagation is governed by the **Nonlinear Schrödinger Equation (NLSE)** (`src/dsp/fiber.py`):

$$
rac{\partial A}{\partial z} + rac{lpha}{2} A + j rac{eta_2}{2} rac{\partial^2 A}{\partial T^2} - rac{eta_3}{6} rac{\partial^3 A}{\partial T^3} = j \gamma |A|^2 A
$$

#### **Standard SMF-28 Parameters**
* Attenuation (α): 0.2 dB/km (α_Np = α_dB / (10 log10 e) × 10^-3 Np/m).
* Chromatic Dispersion (D): 16.0 ps/(nm·km) at 1550 nm.
* Group Velocity Dispersion (β_2): β_2 = -(λ^2 · D) / (2π · c) ≈ -20.4 ps^2/km.
* Third-Order Dispersion (β_3): β_3 = (λ^3 / (2π · c)^2) · (S + 2D / λ) ≈ 0.1 ps^3/km (slope S = 0.09 ps/(nm^2·km)).
* Kerr Nonlinearity (γ): γ = (2π · n_2) / (λ · A_eff) ≈ 1.3 W^-1 km^-1 (n_2 = 3.2 × 10^-20 m^2/W, A_eff = 70 μm^2).

#### **Split-Step Fourier Method (SSFM - Symmetric 2nd Order)**
For nonlinear propagation (`regime=1`), the fiber is discretized into M spatial steps Δz:

1. **Half-step linear dispersive propagation (Δz / 2)**:

$$
\hat{D}(f) = -rac{lpha}{2} - j rac{eta_2}{2} (2\pi f)^2 + j rac{eta_3}{6} (2\pi f)^3
$$

$$
A\left(z + rac{\Delta z}{2}, tight) = \mathcal{F}^{-1} \left\{ \mathcal{F}\{A(z, t)\} \cdot \exp\left( \hat{D}(f) rac{\Delta z}{2} ight) ight\}
$$

2. **Full-step Kerr nonlinear phase shift (Δz)**:

$$
A_{NL}\left(z + rac{\Delta z}{2}, tight) = A\left(z + rac{\Delta z}{2}, tight) \cdot \exp\left( j \gamma \left|A\left(z + rac{\Delta z}{2}, tight)ight|^2 \Delta z ight)
$$

3. **Second half-step linear dispersive propagation (Δz / 2)**:

$$
A(z + \Delta z, t) = \mathcal{F}^{-1} \left\{ \mathcal{F}\{A_{NL}\} \cdot \exp\left( \hat{D}(f) rac{\Delta z}{2} ight) ight\}
$$

#### **Analytical Linear Mode (`regime=0`)**
In the linear regime, exact frequency-domain transfer function multiplication is performed in a single FFT step:

$$
H_{fiber}(f) = \exp\left( -rac{lpha}{2} L ight) \cdot \exp\left( +j rac{eta_2}{2} L (2\pi f)^2 ight)
$$

---

### 3. Laser Phase Noise & Correlation Models

Laser phase noise is modeled as a continuous stochastic **Wiener Process (Random Walk)** (`src/dsp/noise_models.py`), representing Lorentzian spectral linewidth:

$$
\sigma_{\Delta\phi}^2 = 2 \pi \Delta
u \Delta t
$$

$$
\phi[n] = \sum_{k=0}^{n} \Delta\phi[k], \quad \Delta\phi[k] \sim \mathcal{N}(0, \sigma_{\Delta\phi}^2)
$$

* Δν: Laser 3dB linewidth (0 Hz ideal, 100 kHz DFB, 1 MHz external cavity).
* Δt: Sampling period (Δt = 12.5 ps at 80 GHz sampling frequency).

#### **Correlated Phase Coupling (ρ)**
For self-homodyne or comb-based optical systems, phase noise coupling between TX and LO lasers is parameterized by correlation factor ρ ∈ [0, 1]:

$$
\phi_{TX}(t) = \phi_{common}(t)
$$

$$
\phi_{LO}(t) = ho \cdot \phi_{common}(t) + \sqrt{1 - ho^2} \cdot \phi_{indep}(t)
$$

* When ρ = 1.0: Identical phase fluctuations cancel completely during intradyne beating (φ_TX - φ_LO = 0).
* When ρ = 0.0: Independent phase processes yield a net variance σ_net^2 = 4π · Δν · Δt.

---

### 4. Silicon Photonic Neural Network (PNN) Model

The PNN is modeled as an integrated **Silicon-on-Insulator (SOI) Tapped Delay Line (TDL) optical FIR filter** (`src/models/pnn.py`):

#### **Physical Architecture**
* **Channels (N)**: N = 8 or N = 4 parallel optical delay arms.
* **Base Delay (Δt)**: Δt = 25 ps waveguide delay spirals (FSR = 1/Δt = 40 GHz). At 10 GBaud (T_s = 100 ps), this operates as a **T/4 Fractionally-Spaced Equalizer (FSE)**.

#### **Recombined Optical Complex Field**
The optical field recombined at the output combiner tree is:

$$
E_{sum}(t) = rac{1}{N} \sum_{k=0}^{N-1} a_k e^{j\phi_k} E_{in}(t - k\Delta t)
$$

* a_k ∈ [0.0, 1.0]: Amplitude transmittance weights set by thermo-optic Mach-Zehnder Interferometer (MZI) attenuators.
* φ_k ∈ [0, 2π): Optical phase shifts set by thermo-optic phase shifters.

#### **Passive Optical Field Normalization Factor (1/N)**
The factor 1/N represents physical passive energy conservation on chip:
1. 1 × N optical splitter tree reduces field amplitude by 1/√N.
2. N × 1 optical combiner tree reduces field amplitude by 1/√N.
3. Net passive field scaling: (1/√N) × (1/√N) = 1/N (equivalent to an 18 dB power insertion loss for N = 8).

#### **Optical Frequency Transfer Function**

$$
H_{PNN}(f) = rac{1}{N} \sum_{k=0}^{N-1} a_k e^{j\phi_k} \exp\left( -j 2\pi f k \Delta t ight)
$$

#### **Photonic Quadratic Detection**
For direct detection, the photodiode acts as a square-law activation function:

$$
I_{out}(t) = R \cdot |E_{sum}(t)|^2
$$

*(In coherent transmission, E_sum(t) is passed directly into the 90° optical hybrid before photodetection).*

---

### 5. Particle Swarm Optimization (PSO) Model

PNN optical weights (a_k, φ_k) are optimized using a bounded continuous **Particle Swarm Optimization (PSO)** algorithm (`scripts/optimize_pnn_pso.py`):

#### **Parameter State Vector (2N-Dimensional)**

$$
\mathbf{x} = [a_0, a_1, \dots, a_{N-1}, \, \phi_0, \phi_1, \dots, \phi_{N-1}]
$$

#### **Velocity & Position Update Rules**

$$
\mathbf{v}_i[t+1] = w \mathbf{v}_i[t] + c_1 r_1 (\mathbf{pbest}_i - \mathbf{x}_i[t]) + c_2 r_2 (\mathbf{gbest} - \mathbf{x}_i[t])
$$

$$
\mathbf{x}_i[t+1] = \mathbf{x}_i[t] + \mathbf{v}_i[t+1]
$$

* Inertia w = 0.72, Cognitive c_1 = 1.49, Social c_2 = 1.49.
* **Amplitude Bounds**: a_k ∈ [0.0, 1.0] with hard boundary clipping.
* **Periodic Phase Bounds**: φ_k ∈ [-π, π] with shortest-arc modulo periodic wrapping:

$$
\Delta\phi = ((\phi_{target} - \phi_{current} + \pi) mod 2\pi) - \pi
$$

#### **Objective Functions**
* **Continuous EVM RMS (%)**: Smooth optimization landscape preventing gradient plateaus.
* **Discrete BER**: Hard bit counting over training frames.

---

### 6. Joint Optical-Digital Equalizer & Metrics

#### **Joint Equalizer (H_joint(f))**
To eliminate both fiber chromatic dispersion and spectral distortion introduced by the PNN weights (`src/dsp/joint_pnn_cdc_equalizer.py`):

$$
H_{joint}(f) = H_{fiber}^{-1}(f) \cdot rac{H_{PNN}^*(f)}{|H_{PNN}(f)|^2 + arepsilon}
$$

Where H_fiber^-1(f) = exp(-j · (β_2 / 2) · L · (2π f)^2) and ε = 10^-3 · <|H_PNN|^2> regularizes spectral notches.

#### **Optical Performance Metrics (`optical_metrics.py`)**
* **Bit Error Rate (BER)**: Vectorized bit comparison with frame zero-division safety.
* **Error Vector Magnitude (EVM RMS %)**: Decision-directed slicing to the nearest ideal constellation point (QPSK/16-QAM) with optional constant phase rotation correction:

$$
	ext{EVM}_{	ext{RMS}} = \sqrt{ rac{rac{1}{M} \sum_{m=1}^M |s_{	ext{rx}}[m] - s_{	ext{ref}}[m]|^2}{rac{1}{M} \sum_{m=1}^M |s_{	ext{ref}}[m]|^2} } 	imes 100\%
$$

---

## Getting Started & Execution

### Prerequisites
Install dependencies listed in `requirements.txt`:
```bash
pip install numpy scipy matplotlib pandas openpyxl
```

### Running Simulations
1. **Run Coherent Pipeline**:
   ```bash
   python3 scripts/run_coherent_simulation.py
   ```
2. **Optimize PNN Weights via PSO**:
   ```bash
   python3 scripts/optimize_pnn_pso.py
   ```
3. **Execute Standalone Metrics Test**:
   ```bash
   python3 artifacts/modules/optical_metrics.py
   ```

---

## Citation & Contact

If you use this simulator or PNN models in your research, please cite:

> **PNN Coerente Project**, Silicon Photonics & Coherent Optical Systems Group (UFES / IN-Foton), 2026.
> *References*: D. A. A. Mello & F. A. Barbosa (Springer, 2021); P. R. N. Marciano (Ph.D. Thesis, UFES, 2024).
