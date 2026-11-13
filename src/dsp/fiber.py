"""Single-Mode Optical Fiber propagation model (SSFM and Linear).

Faithfully ported from Fibra_Monomodo.m (Segatto & Togneri / Agrawal).
Supports linear regime (regime=0) and non-linear Split-Step Fourier (regime=1).
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class FiberParams:
    """Physical parameters of the standard single-mode fiber (SSMF)."""
    alpha_db: float = 0.2          # Attenuation [dB/km]
    d_1550: float = 16.0           # Chromatic dispersion at 1550 nm [ps/(nm*km)]
    slope_s: float = 0.09          # Dispersion slope [ps/(nm^2*km)]
    n2: float = 3.2e-20            # Non-linear refractive index [m^2/W]
    a_eff: float = 70e-12          # Effective area [m^2]
    c: float = 3e8                 # Speed of light [m/s]


def fibra_monomodo(
    t: np.ndarray,
    u: np.ndarray,
    wavelength: float,
    t0: float,
    length_km: float,
    regime: int = 1,
    params: FiberParams = None,
    num_steps: int = 100,
) -> tuple[np.ndarray, float, float]:
    """Calculates the optical field propagation in the single-mode fiber.

    Args:
        t: Time vector [s].
        u: Electric field envelope [sqrt(W)].
        wavelength: Carrier wavelength [m] (e.g.: 1550e-9).
        t0: Temporal width of the pulse/symbol [s].
        length_km: Fiber length [km].
        regime: 0 -> Linear (analytical transfer function),
                1 -> Split-Step Fourier Method (SSFM with non-linearity).
        params: FiberParams instance. If None, uses default SSMF parameters.
        num_steps: Number of steps for SSFM in regime 1.

    Returns:
        uf: Electric field at the fiber output [sqrt(W)].
        l_d: Dispersion length [km].
        l_nl: Non-linear length [km].
    """
    if params is None:
        params = FiberParams()

    length_m = length_km * 1e3
    n = len(t)
    dt = t[1] - t[0]
    total_t = n * dt

    c = params.c
    freq_carrier = c / wavelength
    omega_carrier = 2 * np.pi * freq_carrier

    # Attenuation: conversion from dB/km to Np/m
    alpha_np = (params.alpha_db / (10 * np.log10(np.e))) * 1e-3

    # Dispersion considering the slope (S) as a function of lambda
    d_disp = params.d_1550 + params.slope_s * (wavelength * 1e9 - 1550.0)  # [ps/(nm*km)]
    d_val = d_disp * 1e-6  # [s/(m*m)]
    beta2 = - (wavelength**2 * d_val) / (2 * np.pi * c)  # [s^2/m]
    beta3 = (d_val * wavelength**3) / ((2 * np.pi * c)**2)  # [s^3/m]

    # Non-linear coefficient gamma [1/(W*m)]
    gamma = params.n2 * omega_carrier / (c * params.a_eff)

    # Angular frequency axis omega (rad/s) with fftshift
    freq_grid = np.fft.fftfreq(n, d=dt)
    omega = 2 * np.pi * freq_grid

    # Characteristic lengths
    t0_ps = t0 * 1e12
    l_d = (t0_ps**2) / (abs(beta2 * 1e27) + 1e-30)  # [km]
    peak_power = np.max(np.abs(u)**2)
    if peak_power > 0 and gamma > 0:
        l_nl = (1.0 / (gamma * peak_power)) / 1000.0  # [km]
    else:
        l_nl = np.inf

    if regime == 0:
        # LINEAR REGIME: Linear transfer function in the frequency domain
        u_spec = np.fft.fft(u)
        h_linear = np.exp(
            (1j / 2.0) * beta2 * length_m * (omega**2) - (alpha_np / 2.0) * length_m
        )
        uf = np.fft.ifft(u_spec * h_linear)
        return uf, l_d, l_nl

    elif regime == 1:
        # SPLIT-STEP FOURIER REGIME
        # Dispersion and linear attenuation operator: Dw
        # Dw = -alpha/2 + j/2 * beta2 * omega^2 - j/6 * beta3 * omega^3
        dw = - (alpha_np / 2.0) + (1j / 2.0) * beta2 * (omega**2) - (1j / 6.0) * beta3 * (omega**3)

        dz = length_m / num_steps
        h2 = dz / 2.0
        exp_h2_dw = np.exp(h2 * dw)

        az = u.copy()
        for _ in range(num_steps):
            # 1. Half dispersive step in the frequency domain
            az_w = np.fft.fft(az) * exp_h2_dw
            az_mid = np.fft.ifft(az_w)

            # 2. Full non-linear step in the time domain
            phase_nl = gamma * dz * (np.abs(az_mid)**2)
            az_nl = az_mid * np.exp(1j * phase_nl)

            # 3. Half dispersive step in the frequency domain
            az_w_final = np.fft.fft(az_nl) * exp_h2_dw
            az = np.fft.ifft(az_w_final)

        return az, l_d, l_nl

    else:
        raise ValueError(f"Invalid regime: {regime}. Choose 0 (Linear) or 1 (SSFM).")
