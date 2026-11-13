"""Traditional coherent QPSK receiver.

Includes beating with a local oscillator (LO), chromatic dispersion equalization (FD-CD),
resampling (decimation) and Euclidean bit demapping.
"""

import numpy as np


def coherent_detection(
    optical_field: np.ndarray,
    phase_lo: np.ndarray,
    lo_power_dbm: float = 10.0,
    responsivity: float = 1.0,
) -> np.ndarray:
    """Simulates intradyne/homodyne coherent beating in the optical hybrid and balanced photodiodes.

    The received optical electric field beats with the Local Oscillator:
      E_LO(t) = sqrt(P_LO) * exp(j * phi_LO(t))
      I_out(t) = 2 * R * sqrt(P_LO) * Re/Im { E_rx(t) * exp(-j * phi_LO(t)) }

    Args:
        optical_field: Envelope of the received optical signal [sqrt(W)].
        phase_lo: Instantaneous phase of the local oscillator [rad].
        lo_power_dbm: LO power [dBm].
        responsivity: Responsivity of the photodetectors [A/W].

    Returns:
        r_baseband: Recovered complex electrical baseband signal.
    """
    p_lo_w = 10.0 ** ((lo_power_dbm - 30.0) / 10.0)
    lo_field = np.sqrt(p_lo_w) * np.exp(1j * phase_lo)

    # Balanced I/Q beating
    r_baseband = responsivity * optical_field * np.conj(lo_field)
    return r_baseband


def cd_compensation(
    signal: np.ndarray,
    dt: float,
    length_km: float,
    dispersion_d: float = 16.0,
    wavelength: float = 1550e-9,
    c: float = 3e8,
) -> np.ndarray:
    """Static chromatic dispersion equalizer in the frequency domain (FD-CD).

    Applies the exact inverse transfer function of the single-mode fiber:
      H_CDC(omega) = exp(+j * (D * lambda^2 / (4 * pi * c)) * L * omega^2)
                   = exp(-j * (beta2 / 2) * L * omega^2)

    Args:
        signal: Electrical or optical baseband signal sampled at dt [s].
        dt: Sampling interval [s].
        length_km: Accumulated fiber link length [km].
        dispersion_d: Chromatic dispersion parameter [ps/(nm*km)].
        wavelength: Carrier wavelength [m].
        c: Speed of light [m/s].

    Returns:
        signal_cdc: Compensated signal in the time domain.
    """
    if length_km <= 0:
        return signal.copy()

    n = len(signal)
    length_m = length_km * 1e3
    freq_grid = np.fft.fftfreq(n, d=dt)
    omega = 2 * np.pi * freq_grid

    d_val = dispersion_d * 1e-6  # [s/(m*m)]
    beta2 = - (wavelength**2 * d_val) / (2 * np.pi * c)  # [s^2/m]

    # Exact inverse filter
    h_cdc = np.exp(- (1j / 2.0) * beta2 * length_m * (omega**2))
    spec_out = np.fft.fft(signal) * h_cdc
    return np.fft.ifft(spec_out)


def sample_to_symbols(
    signal: np.ndarray,
    sps: int,
    sample_offset: int = None,
) -> np.ndarray:
    """Decimates the oversampled signal to 1 sample per symbol at the ideal instant."""
    if sample_offset is None:
        sample_offset = sps // 2  # center of the pulse
    return signal[sample_offset::sps]


def demap_qpsk(symbols_rx: np.ndarray) -> np.ndarray:
    """Euclidean demapping of QPSK symbols to bit sequence (Gray).

    Decision criterion by quadrant:
      Re >= 0 -> bit_I = 0; Re < 0 -> bit_I = 1
      Im >= 0 -> bit_Q = 0; Im < 0 -> bit_Q = 1

    Returns:
        bits_rx: 1D binary vector of size 2 * len(symbols_rx).
    """
    n_sym = len(symbols_rx)
    bits_rx = np.zeros(2 * n_sym, dtype=int)

    bit_i = np.where(np.real(symbols_rx) >= 0.0, 0, 1)
    bit_q = np.where(np.imag(symbols_rx) >= 0.0, 0, 1)

    bits_rx[0::2] = bit_i
    bits_rx[1::2] = bit_q
    return bits_rx
