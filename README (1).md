# Coherent Optical Transmission Simulator & Silicon Photonic Neural Network (PNN)

## Overview

This repository provides an end-to-end simulation framework for high-speed coherent optical communication links integrated with **Silicon Photonic Neural Networks (PNNs)** and **Digital Signal Processing (DSP)** algorithms.

Developed as part of research in integrated photonics and optical equalization (UFES / IN-Foton), the framework models 10 GBaud QPSK and 64 GBaud optical transmission across single-mode optical fibers (SSMF). It investigates the performance of **Photonic Neural Networks** operating as **optical Fractionally-Spaced Equalizers (FSE)** to mitigate Chromatic Dispersion (CD) and nonlinear distortions both at the receiver and distributed *in-line* across multi-span amplified links.

The simulation toolset models the full optical signal chain:
1. **Transmitter (TX)**: QPSK mapping, electrical drive filtering, and dual-drive analytical IQ Mach-Zehnder Modulators (IQ-MZM) with zero-chirp push-pull operation.
2. **Optical Channel (SSMF)**: Single-Mode Fiber propagation modeled via the Split-Step Fourier Method (SSFM) incorporating Chromatic Dispersion (CD), attenuation, third-order dispersion, and Kerr self-phase modulation (SPM).
3. **Optical Amplification (EDFA)**: Ideal Erbium-Doped Fiber Amplifiers with configurable gain to compensate for fiber span attenuation and passive PNN insertion loss.
4. **Photonic Neural Network (PNN)**: Integrated Silicon-on-Insulator (SOI) Tapped Delay Line (TDL) optical FIR filter with complex-valued weights ($a_k e^{j\phi_k}$), passive $1/N$ power normalization, and quadratic photodetector activation.
5. **Coherent Receiver (RX)**: 90-degree optical hybrid beating with a Local Oscillator (LO) under Wiener process phase noise and phase correlation coupling ($\rho$).
6. **Receiver DSP**: Frequency-domain Digital Chromatic Dispersion Compensation (CDC), Joint Optical-Digital Equalization ($H_{\text{joint}}(f)$), symbol slicing, and metrics calculation (BER & EVM).
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
* **Focus**: Investigation of **Distributed Optical PNNs** in multi-span EDFA-amplified QPSK links over long-haul distances up to **10,000 km** (200 spans $\times$ 50 km).
* **Architecture**: Evaluates an in-line configuration `[Fiber_k (50 km) -> PNN_k -> EDFA_k]` for $k = 1 \dots 200$.
* **Key Scenarios Evaluated**:
  1. *Uncompensated Link* (Baseline)
  2. *Conventional Digital CDC*
  3. *Distributed Optical PNNs alone* (tests if optical processing can replace digital CDC)
  4. *Distributed PNNs + Digital CDC Jointly Optimized*
* **Optimization**: Uses stage-wise (span-by-span) Particle Swarm Optimization (PSO) driven by EVM RMS minimization.

### 2. `exploracao_pnn_dispersao_edfa_correlacao_fase.ipynb`
* **Focus**: Evaluates PNN performance under **correlated laser phase noise** between the Transmitter (TX) and Local Oscillator (LO) across different correlation factors ($\rho \in [0.0, 1.0]$) and laser linewidths ($\Delta\nu = 0\text{ Hz}, 100\text{ kHz}, 1\text{ MHz}$).
* **Key Findings**: Demonstrates how phase correlation reduces phase variance during intradyne beating and analyzes the interaction between optical PNN filtering and carrier phase jitter.

### 3. `simulacao_coerente_qpsk.ipynb`
* **Focus**: Baseline implementation of a 10 GBaud QPSK coherent link over SSMF using ideal complex modulation and frequency-domain CDC at the receiver.
* **Purpose**: Serves as the theoretical reference for back-to-back (B2B) transmission and uncompensated chromatic dispersion penalty.

### 4. `simulacao_coerente_qpsk_iq_mzm.ipynb`
* **Focus**: Incorporates an **analytical IQ Mach-Zehnder Modulator (IQ-MZM)** into the coherent pipeline.
* **Analysis**: Explores distance sweeps (50 to 1000 km), $V_\pi$ drive voltage variations, zero-chirp push-pull driving conditions, and laser phase noise impact on receiver constellations.

### 5. `teste_modulador_iq_mzm.ipynb`
* **Focus**: Static optical transfer characterization of the IQ-MZM model.
* **Outputs**: Generates $E_{\text{out}}$ vs. $V_{\text{drive}}$ curves, verifying null bias carrier suppression ($U_0 = V_\pi$), quadrature phase shift ($\phi_{\text{quad}} = -\pi/2$), and push-pull linearity.

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
* **Focus**: Comparative study between conventional Digital CDC ($H_{\text{CDC}}(f)$) and Joint Optical-Digital Equalization ($H_{\text{joint}}(f)$).
* **Core Contribution**: Demonstrates how $H_{\text{joint}}(f)$ cancels both fiber chromatic dispersion and the spectral transfer response introduced by the PNN's optical tap weights.

### 10. `ber_distancia_pnn_cdc_joint_cdc.ipynb`
* **Focus**: Comprehensive sweep of Bit Error Rate (BER) versus distance (50 km to 1000 km) across **7 distinct equalization strategies** and 3 laser linewidths ($0\text{ Hz}, 100\text{ kHz}, 1\text{ MHz}$).
* **Strategies Compared**: Uncompensated, CDC Digital, PNN Alone, PNN + CDC, Joint CDC, PNN (EVM-trained), Joint CDC (EVM-trained).

### 11. `evm_pnn_cdc_treinamento_conjunto.ipynb`
* **Focus**: Methodological comparison between **BER-driven PSO** and **EVM-driven PSO** training for PNN weights.
* **Key Result**: Demonstrates that continuous EVM RMS minimization eliminates optimization plateaus, resulting in faster PSO convergence and superior constellation quality.

---

## Deep Dive into Core Algorithms & Physical Models

### 1. Mach-Zehnder Modulators (IQ-MZM & Drive Filtering)

The optical transmitter models an analytical **IQ Mach-Zehnder Modulator (IQ-MZM)** (`src/dsp/transmitter.py`), ported from electro-optic physics models:

#### **Sub-MZM Transfer Function (Dual-Drive)**
Each nested sub-MZM (for In-Phase $I$ and Quadrature $Q$ branches) applies the dual-drive electro-optic phase modulation equation:
$$E_{\text{out}}(t) = E_{\text{in}}(t) \cdot \cos\left( \frac{\pi}{2} \frac{U_1(t) - U_2(t) - U_0}{V_\pi} \right) \cdot \exp\left( -j \frac{\pi}{2} \frac{U_1(t) + U_2(t)}{V_\pi} \right)$$

* $U_1(t), U_2(t)$: Driving voltages applied to the upper and lower waveguide arms.
* $V_\pi$: Half-wave switching voltage (typically $3.1\text{ V}$ to $3.25\text{ V}$).
* $U_0 = V_\pi$: Null-bias condition for suppressed-carrier optical modulation.

#### **Push-Pull Zero-Chirp Driving**
Applying anti-phase drive signals $U_1(t) = -U_2(t) = \frac{V(t)}{2}$ sets the common-mode phase term to zero:
$$\exp\left( -j \frac{\pi}{2} \frac{U_1(t) + U_2(t)}{V_\pi} \right) = \exp(0) = 1$$
This completely eliminates transmitter residual chirp, producing pure double-sideband suppressed-carrier (DSB-SC) modulation.

#### **Quadrature IQ Combination**
The $I$ and $Q$ branch fields are recombined with a static $\pi/2$ phase offset:
$$E_{\text{TX}}(t) = \frac{1}{2} \left[ E_I(t) + E_Q(t) \cdot e^{j \phi_{\text{quad}}} \right], \quad \text{where } \phi_{\text{quad}} = -\frac{\pi}{2}$$

#### **Electrical Drive Filtering**
Before modulation, baseband electrical driving signals pass through a digital filter (`src/utils/filtros.py`) simulating DAC bandwidth limits:
* **Super-Gaussian Filter**: $H(f) = \exp\left( -\ln(2) \cdot \left( \frac{f}{f_c} \right)^{2n} \right)$ with cutoff $f_c = 2 \cdot R_s = 20\text{ GHz}$ and order $n=7$.
* **Raised Cosine & Butterworth**: Alternative user-selectable electrical response filters.

---

### 2. Single-Mode Fiber (SSMF) Propagation Model

Fiber propagation is governed by the **Nonlinear Schrödinger Equation (NLSE)** (`src/dsp/fiber.py`):
$$\frac{\partial A}{\partial z} + \frac{\alpha}{2} A + j \frac{\beta_2}{2} \frac{\partial^2 A}{\partial T^2} - \frac{\beta_3}{6} \frac{\partial^3 A}{\partial T^3} = j \gamma |A|^2 A$$

#### **Standard SMF-28 Parameters**
* Attenuation ($\alpha$): $0.2\text{ dB/km}$ ($\alpha_{\text{Np}} = \alpha_{\text{dB}} / (10 \log_{10} e) \times 10^{-3}\text{ Np/m}$).
* Chromatic Dispersion ($D$): $16.0\text{ ps/(nm}\cdot\text{km)}$ at $1550\text{ nm}$.
* Group Velocity Dispersion ($\beta_2$): $\beta_2 = -\frac{\lambda^2 D}{2\pi c} \approx -20.4\text{ ps}^2/\text{km}$.
* Third-Order Dispersion ($\beta_3$): $\beta_3 = \frac{\lambda^3}{(2\pi c)^2} \left( S + \frac{2 D}{\lambda} \right) \approx 0.1\text{ ps}^3/\text{km}$ (slope $S = 0.09\text{ ps/(nm}^2\cdot\text{km)}$).
* Kerr Nonlinearity ($\gamma$): $\gamma = \frac{2\pi n_2}{\lambda A_{\text{eff}}} \approx 1.3\text{ W}^{-1}\text{km}^{-1}$ ($n_2 = 3.2 \times 10^{-20}\text{ m}^2/\text{W}$, $A_{\text{eff}} = 70\text{ }\mu\text{m}^2$).

#### **Split-Step Fourier Method (SSFM - Symmetric 2nd Order)**
For nonlinear propagation (`regime=1`), the fiber is discretized into $M$ spatial steps $\Delta z$:
1. **Half-step linear dispersive propagation ($\Delta z / 2$)**:
   $$\hat{D}(f) = -\frac{\alpha}{2} - j \frac{\beta_2}{2} (2\pi f)^2 + j \frac{\beta_3}{6} (2\pi f)^3$$
   $$A\left(z + \frac{\Delta z}{2}, t\right) = \mathcal{F}^{-1} \left\{ \mathcal{F}\{A(z, t)\} \cdot \exp\left( \hat{D}(f) \frac{\Delta z}{2} \right) \right\}$$
2. **Full-step Kerr nonlinear phase shift ($\Delta z$)**:
   $$A_{\text{NL}}\left(z + \frac{\Delta z}{2}, t\right) = A\left(z + \frac{\Delta z}{2}, t\right) \cdot \exp\left( j \gamma \left|A\left(z + \frac{\Delta z}{2}, t\right)\right|^2 \Delta z \right)$$
3. **Second half-step linear dispersive propagation ($\Delta z / 2$)**:
   $$A(z + \Delta z, t) = \mathcal{F}^{-1} \left\{ \mathcal{F}\{A_{\text{NL}}\} \cdot \exp\left( \hat{D}(f) \frac{\Delta z}{2} \right) \right\}$$

#### **Analytical Linear Mode (`regime=0`)**
In the linear regime, exact frequency-domain transfer function multiplication is performed in a single FFT step:
$$H_{\text{fiber}}(f) = \exp\left( -\frac{\alpha}{2} L \right) \cdot \exp\left( +j \frac{\beta_2}{2} L (2\pi f)^2 \right)$$

---

### 3. Laser Phase Noise & Correlation Models

Laser phase noise is modeled as a continuous stochastic **Wiener Process (Random Walk)** (`src/dsp/noise_models.py`), representing Lorentzian spectral linewidth:

$$\sigma_{\Delta\phi}^2 = 2 \pi \Delta\nu \Delta t$$
$$\phi[n] = \sum_{k=0}^{n} \Delta\phi[k], \quad \Delta\phi[k] \sim \mathcal{N}(0, \sigma_{\Delta\phi}^2)$$

* $\Delta\nu$: Laser 3dB linewidth ($\text{0 Hz}$ ideal, $100\text{ kHz}$ DFB, $1\text{ MHz}$ external cavity).
* $\Delta t$: Sampling period ($\Delta t = 12.5\text{ ps}$ at $80\text{ GHz}$ sampling frequency).

#### **Correlated Phase Coupling ($\rho$)**
For self-homodyne or comb-based optical systems, phase noise coupling between TX and LO lasers is parameterized by correlation factor $\rho \in [0, 1]$:
$$\phi_{\text{TX}}(t) = \phi_{\text{common}}(t)$$
$$\phi_{\text{LO}}(t) = \rho \cdot \phi_{\text{common}}(t) + \sqrt{1 - \rho^2} \cdot \phi_{\text{indep}}(t)$$

* When $\rho = 1.0$: Identical phase fluctuations cancel completely during intradyne beating ($\phi_{\text{TX}} - \phi_{\text{LO}} = 0$).
* When $\rho = 0.0$: Independent phase processes yield a net variance $\sigma_{\text{net}}^2 = 4\pi \Delta\nu \Delta t$.

---

### 4. Silicon Photonic Neural Network (PNN) Model

The PNN is modeled as an integrated **Silicon-on-Insulator (SOI) Tapped Delay Line (TDL) optical FIR filter** (`src/models/pnn.py`):

#### **Physical Architecture**
* **Channels ($N$)**: $N = 8$ or $N = 4$ parallel optical delay arms.
* **Base Delay ($\Delta t$)**: $\Delta t = 25\text{ ps}$ waveguide delay spirals ($\text{FSR} = 1/\Delta t = 40\text{ GHz}$). At $10\text{ GBaud}$ ($T_s = 100\text{ ps}$), this operates as a **$T/4$ Fractionally-Spaced Equalizer (FSE)**.

#### **Recombined Optical Complex Field**
The optical field recombined at the output combiner tree is:
$$E_{\text{sum}}(t) = \frac{1}{N} \sum_{k=0}^{N-1} a_k e^{j\phi_k} E_{\text{in}}(t - k\Delta t)$$

* $a_k \in [0.0, 1.0]$: Amplitude transmittance weights set by thermo-optic Mach-Zehnder Interferometer (MZI) attenuators.
* $\phi_k \in [0, 2\pi)$: Optical phase shifts set by thermo-optic phase shifters.

#### **Passive Optical Field Normalization Factor ($1/N$)**
The factor $1/N$ represents physical passive energy conservation on chip:
1. $1 \times N$ optical splitter tree reduces field amplitude by $1/\sqrt{N}$.
2. $N \times 1$ optical combiner tree reduces field amplitude by $1/\sqrt{N}$.
3. Net passive field scaling: $\frac{1}{\sqrt{N}} \times \frac{1}{\sqrt{N}} = \frac{1}{N}$ (equivalent to an $18\text{ dB}$ power insertion loss for $N=8$).

#### **Optical Frequency Transfer Function**
$$H_{\text{PNN}}(f) = \frac{1}{N} \sum_{k=0}^{N-1} a_k e^{j\phi_k} \exp\left( -j 2\pi f k \Delta t \right)$$

#### **Photonic Quadratic Detection**
For direct detection, the photodiode acts as a square-law activation function:
$$I_{\text{out}}(t) = R \cdot |E_{\text{sum}}(t)|^2$$
*(In coherent transmission, $E_{\text{sum}}(t)$ is passed directly into the 90° optical hybrid before photodetection).*

---

### 5. Particle Swarm Optimization (PSO) Model

PNN optical weights ($a_k, \phi_k$) are optimized using a bounded continuous **Particle Swarm Optimization (PSO)** algorithm (`scripts/optimize_pnn_pso.py`):

#### **Parameter State Vector ($2N$-Dimensional)**
$$\mathbf{x} = [a_0, a_1, \dots, a_{N-1}, \, \phi_0, \phi_1, \dots, \phi_{N-1}]$$

#### **Velocity & Position Update Rules**
$$\mathbf{v}_i[t+1] = w \mathbf{v}_i[t] + c_1 r_1 (\mathbf{pbest}_i - \mathbf{x}_i[t]) + c_2 r_2 (\mathbf{gbest} - \mathbf{x}_i[t])$$
$$\mathbf{x}_i[t+1] = \mathbf{x}_i[t] + \mathbf{v}_i[t+1]$$

* Inertia $w = 0.72$, Cognitive $c_1 = 1.49$, Social $c_2 = 1.49$.
* **Amplitude Bounds**: $a_k \in [0.0, 1.0]$ with hard boundary clipping.
* **Periodic Phase Bounds**: $\phi_k \in [-\pi, \pi]$ with shortest-arc modulo periodic wrapping:
  $$\Delta\phi = ((\phi_{\text{target}} - \phi_{\text{current}} + \pi) \bmod 2\pi) - \pi$$

#### **Objective Functions**
* **Continuous EVM RMS (%)**: Smooth optimization landscape preventing gradient plateaus.
* **Discrete BER**: Hard bit counting over training frames.

---

### 6. Joint Optical-Digital Equalizer & Metrics

#### **Joint Equalizer ($H_{\text{joint}}(f)$)**
To eliminate both fiber chromatic dispersion and spectral distortion introduced by the PNN weights (`src/dsp/joint_pnn_cdc_equalizer.py`):
$$H_{\text{joint}}(f) = H_{\text{fiber}}^{-1}(f) \cdot \frac{H_{\text{PNN}}^*(f)}{|H_{\text{PNN}}(f)|^2 + \varepsilon}$$
Where $H_{\text{fiber}}^{-1}(f) = \exp\left( -j \frac{\beta_2}{2} L (2\pi f)^2 \right)$ and $\varepsilon = 10^{-3} \cdot \langle |H_{\text{PNN}}|^2 \rangle$ regularizes spectral notches.

#### **Optical Performance Metrics (`optical_metrics.py`)**
* **Bit Error Rate (BER)**: Vectorized bit comparison with frame zero-division safety.
* **Error Vector Magnitude (EVM RMS %)**: Decision-directed slicing to the nearest ideal constellation point (QPSK/16-QAM) with optional constant phase rotation correction:
  $$\text{EVM}_{\text{RMS}} = \sqrt{ \frac{\frac{1}{M} \sum_{m=1}^M |s_{\text{rx}}[m] - s_{\text{ref}}[m]|^2}{\frac{1}{M} \sum_{m=1}^M |s_{\text{ref}}[m]|^2} } \times 100\%$$

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
