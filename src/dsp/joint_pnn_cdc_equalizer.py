"""
===============================================================================
Module: Joint PNN + DSP CDC Equalizer (Joint PNN-CDC Equalizer)
===============================================================================
Description:
    This module implements the joint compensation of Chromatic Dispersion (CD)
    of the fiber and the frequency response of the Photonic Neural Network (PNN)
    in the digital signal processing (DSP) of a coherent receiver.

    While the traditional CDC algorithm only inverts the fiber transfer function
    H_fiber(f), the Joint CDC inverts the cascaded optical-digital response
    [H_fiber(f) * H_PNN(f)], seeking to compensate for linear ISI and the
    distortion introduced by the PNN weights, subject to regularization.

Mathematical Formulation:
    1. Frequency Response of the Combined Optical Channel:
       H_total_channel(f) = H_fiber(f) * H_PNN(f)
       where H_fiber(f) = exp(+j * (1/2) * β2 * L * (2πf)²), as per src.dsp.fiber

    2. Optical Frequency Response of the PNN:
       H_PNN(f) = ∑_{i=0}^{N-1} (a_i / N) * exp(j * φ_i) * exp(-j * 2π * f * i * Δt)

    3. Joint CDC Filter in DSP (Regularized Inverse):
       H_CDC_joint(f) = H_fiber^{-1}(f) * [ H_PNN(f) ]^* / [ |H_PNN(f)|² + ε ]

References:
    - Marciano, P. R. N. (2024). Doctoral Thesis, UFES (PNN and FIR Filters).
    - Gitlin & Weinstein (1981). Fractionally-Spaced Equalization (FSE).
    - Mello & Barbosa (2021). Digital Coherent Optical Systems, Ch. 5 and 6.
===============================================================================
"""

from __future__ import annotations
import numpy as np

from src.models.pnn import PhotonicNeuralNetwork


def simulate_fiber_dispersion(
    optical_field: np.ndarray,
    dt: float,
    length_km: float,
    dispersion_d: float = 16.0,
    wavelength: float = 1550e-9,
    c: float = 3e8,
) -> np.ndarray:
    """Simulates chromatic dispersion with the same phase convention as ``fiber.py``.

    This isolated model does not include attenuation.
    """
    if length_km <= 0:
        return np.asarray(optical_field, dtype=complex).copy()

    optical_field = np.asarray(optical_field, dtype=complex)
    n = len(optical_field)
    length_m = length_km * 1e3
    freq_grid = np.fft.fftfreq(n, d=dt)
    omega = 2.0 * np.pi * freq_grid

    d_val = dispersion_d * 1e-6
    beta2 = - (wavelength**2 * d_val) / (2.0 * np.pi * c)

    h_fiber = np.exp(+ (1j / 2.0) * beta2 * length_m * (omega**2))
    return np.fft.ifft(np.fft.fft(optical_field) * h_fiber)


def cd_compensation_traditional(
    signal_rx: np.ndarray,
    dt: float,
    length_km: float,
    dispersion_d: float = 16.0,
    wavelength: float = 1550e-9,
    c: float = 3e8,
) -> np.ndarray:
    """Traditional CDC equalization, inverse of the ``fiber.py`` convention."""
    if length_km <= 0:
        return np.asarray(signal_rx, dtype=complex).copy()

    signal_rx = np.asarray(signal_rx, dtype=complex)
    n = len(signal_rx)
    length_m = length_km * 1e3
    freq_grid = np.fft.fftfreq(n, d=dt)
    omega = 2.0 * np.pi * freq_grid

    d_val = dispersion_d * 1e-6
    beta2 = - (wavelength**2 * d_val) / (2.0 * np.pi * c)

    h_cdc = np.exp(- (1j / 2.0) * beta2 * length_m * (omega**2))
    return np.fft.ifft(np.fft.fft(signal_rx) * h_cdc)


def cd_compensation_joint_pnn(
    signal_rx: np.ndarray,
    dt: float,
    length_km: float,
    pnn_instance: PhotonicNeuralNetwork,
    dispersion_d: float = 16.0,
    wavelength: float = 1550e-9,
    c: float = 3e8,
    reg_factor: float = 1e-3,
) -> np.ndarray:
    """Joint PNN + DSP CDC Equalizer in the Coherent Receiver.

    Inverts the frequency response of the fiber (convention of ``fiber.py``) and the PNN.
    """
    signal_rx = np.asarray(signal_rx, dtype=complex)
    n = len(signal_rx)
    length_m = length_km * 1e3
    freq_grid = np.fft.fftfreq(n, d=dt)
    omega = 2.0 * np.pi * freq_grid

    # 1. Fiber Inverse
    d_val = dispersion_d * 1e-6
    beta2 = - (wavelength**2 * d_val) / (2.0 * np.pi * c)
    h_fiber_inv = np.exp(- (1j / 2.0) * beta2 * length_m * (omega**2))

    # 2. PNN Frequency Response: H_PNN(f)
    h_pnn = pnn_instance.frequency_response(freq_grid)

    # 3. Regularized Inverse of PNN: H_PNN_inv = H_PNN* / (|H_PNN|² + ε)
    h_pnn_inv = np.conj(h_pnn) / (np.abs(h_pnn)**2 + reg_factor)

    # 4. Joint CDC Filter: H_joint = H_fiber_inv * H_pnn_inv
    h_cdc_joint = h_fiber_inv * h_pnn_inv

    return np.fft.ifft(np.fft.fft(signal_rx) * h_cdc_joint)


def calculate_evm_qpsk(symbols_rx: np.ndarray) -> float:
    """Calculates the RMS EVM (%) for a normalized QPSK constellation."""
    symbols_rx = np.asarray(symbols_rx, dtype=complex)
    decided = np.zeros_like(symbols_rx, dtype=complex)
    decided.real = np.where(symbols_rx.real >= 0, 1.0 / np.sqrt(2), -1.0 / np.sqrt(2))
    decided.imag = np.where(symbols_rx.imag >= 0, 1.0 / np.sqrt(2), -1.0 / np.sqrt(2))

    p_ref = np.mean(np.abs(decided)**2)
    power_rx = np.mean(np.abs(symbols_rx)**2)
    if power_rx <= 0:
        return 100.0

    scale = np.sqrt(p_ref / power_rx)
    norm_symbols = symbols_rx * scale

    error = norm_symbols - decided
    return float(np.sqrt(np.mean(np.abs(error)**2) / p_ref) * 100.0)
