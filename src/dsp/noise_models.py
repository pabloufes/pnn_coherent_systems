"""
===============================================================================
Module: Phase Noise Modeling, Correlation, and Optical AWGN/ASE Noise
===============================================================================
Description:
    This module implements the stochastic generation of semiconductor laser
    phase noise (Wiener/Random Walk model), the correlated phase mixing
    between the transmission carrier (TX) and the Local Oscillator (LO) for
    conventional and self-homodyne systems (Coherent-Lite / OFC), and the
    addition of complex additive Gaussian white noise (AWGN).

Notebook References:
    - Rodigheri, M. (2026). Doctoral Thesis, UNICAMP (Ch. 2 - Phase Noise Physics).
    - Mello, D. A. A., & Barbosa, F. A. (2021). Digital Coherent Optical Systems
      (Springer, Ch. 2, 4 and 6).
    - Lundberg, L., et al. (2020). Nature Communications (Correlated phase noise in OFC).
    - Binh, L. N. (2013). Digital Processing: Optical Transmission and Coherent Receiving.
===============================================================================
"""

import numpy as np


def generate_wiener_phase_noise(
    num_samples: int,
    dt: float,
    linewidth_hz: float,
    seed: int = None,
) -> np.ndarray:
    """Generates a stochastic Wiener phase noise process (Random Walk).

    Theory:
        The intrinsic phase noise of a continuous-wave (CW) laser originates
        from the quantum phenomenon of spontaneous photon emission in the laser
        cavity. Each spontaneous emission randomly changes the phase of the
        stimulated optical field, generating a Lorentzian line profile in the
        frequency domain, with FWHM linewidth given by the Schawlow-Townes-Henry
        equation:
            Δν = [R_sp / (4 * π * S)] * (1 + α^2)

        In the time domain, the phase performs a random walk (Wiener Process),
        where the phase increment Δφ between consecutive samples separated by dt
        follows a Gaussian distribution with zero mean.

    Applied Equations:
        1. Variance of the phase increment [rad²]:
            σ_Δφ² = 2 * π * Δν * dt
        2. Standard deviation of the increment [rad]:
            σ_Δφ = sqrt(2 * π * Δν * dt)
        3. Temporal phase accumulation (Integration):
            φ[n] = ∑_{k=0}^{n} Δφ[k],   with Δφ ~ N(0, σ_Δφ²)

    References:
        - Mareli Rodigheri (2026), Doctoral Thesis, UNICAMP.
        - Darli Mello & Fabio Barbosa (2021), "Digital Coherent Optical Systems", Ch. 6.
        - Kazuro Kikuchi (2010), "Coherent Optical Communications Systems".

    Args:
        num_samples: Number of samples in the time domain (N).
        dt: Time sampling interval [s] (dt = 1 / sampling_frequency).
        linewidth_hz: Lorentzian FWHM linewidth of the laser Δν [Hz].
        seed: Seed for the pseudo-random number generator (optional).

    Returns:
        phi: Vector with the evolution of the instantaneous phase θ(t) [rad].
    """
    if linewidth_hz <= 0:
        return np.zeros(num_samples)

    rng = np.random.default_rng(seed)

    # Equation (1): Variance of the Wiener phase increment
    # σ_Δφ² = 2 * π * Δν * dt
    var_delta_phi = 2.0 * np.pi * linewidth_hz * dt
    sigma_delta_phi = np.sqrt(var_delta_phi)

    # Equation (2): Generation of independent Gaussian increments Δφ ~ N(0, σ_Δφ)
    delta_phi = rng.normal(loc=0.0, scale=sigma_delta_phi, size=num_samples)
    delta_phi[0] = 0.0  # Initial condition: zero phase at instant t = 0

    # Equation (3): Temporal integration (random walk / Wiener process)
    # φ[n] = φ[n-1] + Δφ[n]
    phi = np.cumsum(delta_phi)
    return phi


def generate_correlated_phases(
    num_samples: int,
    dt: float,
    linewidth_hz: float,
    linewidth_lo_hz=None,
    correlation_rho: float = 1.0,
    seed: int = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates the carrier phase noise (TX) and local oscillator (LO) phase with correlation ρ.

    Theory:
        - Traditional Intradyne Systems (ρ = 0):
          Transmitter and LO operate with independent lasers. The beat signal
          at the photodetector inherits the sum of the linewidths (Δν_beat = Δν_TX + Δν_LO),
          requiring Carrier Phase Recovery (CPE/BPS) algorithms in the DSP.

        - Self-Homodyne / OFC Comb Systems (Coherent-Lite, ρ -> 1):
          The data carrier and the LO reference are derived from the same seed laser
          (or from correlated lines of an OFC comb with correlation > 99.99%).
          In balanced coherent beating, the common-mode phase noise φ_0(t) is
          subtracted, canceling the phase noise in the optical domain.

    Applied Equations:
        1. Correlated Linear Mixing of Wiener Processes:
            φ_TX(t) = φ_common(t)
            φ_LO(t) = ρ * φ_common(t) + sqrt(1 - ρ²) * φ_indep(t)

        2. Residual Phase of the Coherent Beat (Δφ_beat = φ_TX - φ_LO):
            Δφ_beat(t) = (1 - ρ) * φ_common(t) - sqrt(1 - ρ²) * φ_indep(t)

        3. Residual Variance per Symbol in the Beat:
            Var(Δφ_beat) = 2 * (1 - ρ) * (2 * π * Δν * dt)
            - If ρ = 1.0  => Var(Δφ_beat) = 0 (Total Cancellation in Optical Domain)
            - If ρ = 0.0  => Var(Δφ_beat) = 2 * (2 * π * Δν * dt) (Independent Lasers)

    References:
        - Lundberg et al. (2020), "Phase-correlated comb lines", Nature Comm.
        - SBFoton (2025), "Self-Homodyne Coherent-Lite Technique for DWDM".
        - Mareli Rodigheri (2026), Doctoral Thesis, UNICAMP (Ch. 2).
        - Darli Mello & Fabio Barbosa (2021), "Digital Coherent Optical Systems", Ch. 6.

    Args:
        num_samples: Number of samples (N).
        dt: Time sampling interval [s].
        linewidth_hz: Individual laser linewidth Δν [Hz].
        correlation_rho: Phase correlation coefficient ρ (0.0 <= ρ <= 1.0).
        seed: Random seed (optional).

    Returns:
        phi_tx: Instantaneous phase vector of the carrier at the transmitter [rad].
        phi_lo: Instantaneous phase vector of the local oscillator at the receiver [rad].
    """
    if linewidth_lo_hz is None:
        linewidth_lo_hz = linewidth_hz
    if linewidth_hz <= 0 and linewidth_lo_hz <= 0:
        return np.zeros(num_samples), np.zeros(num_samples)

    rng = np.random.default_rng(seed)
    seed1 = rng.integers(0, 1000000)
    seed2 = rng.integers(0, 1000000)

    # 1. Common Wiener Process (represents the seed laser / single source)
    phi_common = generate_wiener_phase_noise(num_samples, dt, 1.0, seed=seed1)

    # 2. Independent Wiener Process (represents uncorrelated noise)
    phi_indep = generate_wiener_phase_noise(num_samples, dt, 1.0, seed=seed2)

    # 3. Phase mixing with correlation coefficient ρ
    phi_tx = np.sqrt(linewidth_hz) * phi_common
    rho_clamped = np.clip(correlation_rho, 0.0, 1.0)

    # Equation (1): φ_LO = ρ * φ_common + sqrt(1 - ρ²) * φ_indep
    phi_lo =  np.sqrt(linewidth_lo_hz) * (rho_clamped * phi_common + np.sqrt(1.0 - rho_clamped**2) * phi_indep)

    return phi_tx, phi_lo


def add_awgn_noise(
    signal: np.ndarray,
    snr_db: float,
    seed: int = None,
) -> np.ndarray:
    """Adds Complex Additive Gaussian White Noise (AWGN) to the I/Q signal.

    Theory:
        Models thermal noise, shot noise in the photodetectors, and mainly
        the Amplified Spontaneous Emission (ASE) noise generated by EDFA
        optical amplifiers along the link. The complex AWGN is zero-mean and
        follows a bidimensional Gaussian distribution, dividing its total power
        equally between the In-Phase (I) and Quadrature (Q) axes.

    Applied Equations:
        1. Average Signal Power:
            P_signal = E{|s(t)|²} = (1 / N) * ∑ |s[n]|²
        2. Linear Signal-to-Noise Ratio:
            SNR_linear = 10^(SNR_dB / 10)
        3. Total Noise Power:
            P_noise = P_signal / SNR_linear
        4. Standard Deviation per Dimension (I and Q):
            σ_I = σ_Q = sqrt(P_noise / 2)
        5. Noisy Signal:
            s_out[n] = s[n] + (n_I[n] + j * n_Q[n]),  where n_I, n_Q ~ N(0, σ_I)

    References:
        - Darli Mello & Fabio Barbosa (2021), "Digital Coherent Optical Systems", Ch. 2.
        - Le Nguyen Binh (2013), "Digital Processing: Optical Transmission", Ch. 3.

    Args:
        signal: Complex baseband signal vector (I + j*Q).
        snr_db: Desired Signal-to-Noise Ratio [dB].
        seed: Random seed (optional).

    Returns:
        signal_noisy: Complex signal with added AWGN noise.
    """
    if snr_db is None or np.isneginf(snr_db):
        return signal

    rng = np.random.default_rng(seed)

    # Equation (1): Average power of the input signal P_signal
    p_signal = np.mean(np.abs(signal)**2)

    # Equation (2): Conversion of SNR from dB to linear scale
    snr_linear = 10.0 ** (snr_db / 10.0)

    # Equation (3): Total noise power P_noise
    p_noise = p_signal / snr_linear

    # Equation (4): Standard deviation per real (I) and imaginary (Q) component
    # σ = sqrt(P_noise / 2)
    sigma = np.sqrt(p_noise / 2.0)

    # Equation (5): Complex Gaussian noise generator n_I + j*n_Q
    noise_i = rng.normal(loc=0.0, scale=sigma, size=len(signal))
    noise_q = rng.normal(loc=0.0, scale=sigma, size=len(signal))
    noise = noise_i + 1j * noise_q

    return signal + noise


if __name__ == "__main__":
    print("=== Phase Noise and AWGN Module for Coherent Systems ===")
    num_samples = 10000
    f_amostragem = 160e9  # 160 GHz
    dt = 1.0 / f_amostragem
    linewidth_hz = 100e3  # 100 kHz

    # Test 1: Wiener noise generation
    phi = generate_wiener_phase_noise(num_samples, dt, linewidth_hz, seed=42)
    print(f"[OK] Wiener noise generated: {len(phi)} samples, Final deviation = {phi[-1]:.4f} rad")

    # Test 2: Correlated noise (rho = 0.9999 vs rho = 0.0)
    phi_tx, phi_lo_corr = generate_correlated_phases(num_samples, dt, linewidth_hz, correlation_rho=0.9999, seed=42)
    diff_corr = phi_tx - phi_lo_corr
    print(f"[OK] Self-Homodyne Beat (rho=0.9999): Residual variance = {np.var(diff_corr):.2e} rad²")

    phi_tx, phi_lo_indep = generate_correlated_phases(num_samples, dt, linewidth_hz, correlation_rho=0.0, seed=42)
    diff_indep = phi_tx - phi_lo_indep
    print(f"[OK] Intradyne Beat (rho=0.0): Residual variance = {np.var(diff_indep):.2e} rad²")

    # Test 3: AWGN addition
    sinal_dummy = np.ones(num_samples, dtype=complex)
    sinal_ruidoso = add_awgn_noise(sinal_dummy, snr_db=20.0, seed=42)
    snr_medida = 10 * np.log10(np.mean(np.abs(sinal_dummy)**2) / np.mean(np.abs(sinal_ruidoso - sinal_dummy)**2))
    print(f"[OK] AWGN added (target SNR 20 dB): Measured SNR = {snr_medida:.2f} dB")
    print("=== All tests completed successfully! ===")
