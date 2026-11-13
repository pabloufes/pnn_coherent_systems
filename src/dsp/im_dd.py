"""Simulation and testing module for IM-DD (Intensity Modulation / Direct Detection) components.

Reproduces the MATLAB routines:
- SCM_FechamentoCW.m (RF Power Fading study induced by chromatic dispersion)
- Laser_DFB_Simplificado.m (Static DFB laser PxI model)
- Fotodiodo.m (Square-law photodetector)
"""

from dataclasses import dataclass
import numpy as np
from src.dsp.fiber import fibra_monomodo, FiberParams


@dataclass
class DFBParams:
    """DFB Laser parameters (based on SCM_Input_Data.m)."""
    i_bias: float = 100.0   # Bias current [mA]
    i_th: float = 30.0      # Threshold current [mA]
    k_slope: float = 0.999  # PxI curve slope [mW/mA]
    i_factor: float = 50.0  # Modulating signal amplitude [mA]


def laser_dfb_simplificado(
    i_mod: np.ndarray,
    params: DFBParams = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Models the direct modulation of a DFB laser above threshold.

    Ported from Laser_DFB_Simplificado.m.

    Args:
        i_mod: Modulation current [mA].
        params: DFBParams instance.

    Returns:
        p_watts: Instantaneous optical power [W].
        e_field: Optical electric field envelope [sqrt(W)].
    """
    if params is None:
        params = DFBParams()

    i_total = params.i_bias + i_mod
    if np.any(i_total < params.i_th):
        # Clamp at threshold current to avoid negative values
        i_total = np.maximum(i_total, params.i_th)

    p_mw = params.k_slope * (i_total - params.i_th)
    p_watts = p_mw * 1e-3
    e_field = np.sqrt(np.maximum(p_watts, 0.0))
    return p_watts, e_field


def fotodiodo_pin(
    campo_e: np.ndarray,
    responsivity: float = 0.75,
) -> np.ndarray:
    """Square-law PIN photodetector.

    Ported from Fotodiodo.m.

    Args:
        campo_e: Optical electric field [sqrt(W)].
        responsivity: Photodiode responsivity [A/W].

    Returns:
        photocurrent: Generated photocurrent [A].
    """
    p_in = np.abs(campo_e)**2
    return responsivity * p_in


def calcular_rf_power_fading_teorico(
    freqs_hz: np.ndarray,
    length_km: float,
    wavelength_m: float = 1563e-9,
    d_disp_ps_nm_km: float = 17.17,
    c: float = 3e8,
) -> tuple[np.ndarray, list[float]]:
    """Calculates the theoretical analytical RF Power Fading response in an IM-DD system.

    Formula:
      |H_RF(f)| = | cos( (pi * D * lambda^2 * L / c) * f^2 ) |

    Null frequencies:
      f_null_m = sqrt( (2*m + 1) * c / (2 * D * lambda^2 * L) ),  m = 0, 1, 2, ...

    Returns:
        h_rf_db: Normalized response in dB (20 * log10(|H_RF|)).
        notches_hz: List of null frequencies within the band.
    """
    length_m = length_km * 1e3
    d_si = d_disp_ps_nm_km * 1e-6  # [s/(m*m)]

    arg = (np.pi * d_si * (wavelength_m**2) * length_m / c) * (freqs_hz**2)
    h_rf_lin = np.abs(np.cos(arg))
    h_rf_db = 20.0 * np.log10(np.maximum(h_rf_lin, 1e-4) / (h_rf_lin[0] + 1e-12))

    # Calculate theoretical null frequencies
    notches_hz = []
    m = 0
    while True:
        f_null = np.sqrt(((2 * m + 1) * c) / (2 * d_si * (wavelength_m**2) * length_m))
        if f_null > freqs_hz[-1]:
            break
        if f_null >= freqs_hz[0]:
            notches_hz.append(f_null)
        m += 1

    return h_rf_db, notches_hz


def simular_scm_fechamento_cw(
    freqs_hz: np.ndarray,
    length_km: float = 100.0,
    wavelength_m: float = 1563e-9,
    regime: int = 0,
    np_points: int = 512,
    num_cycles: int = 10,
    dfb_params: DFBParams = None,
    fiber_params: FiberParams = None,
    num_ssfm_steps: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Runs the complete CW subcarrier fading simulation as in SCM_FechamentoCW.m.

    For each subcarrier frequency:
    1. Generates the RF sinusoidal signal: s(t) = cos(2*pi*fc*t).
    2. Modulates the simplified DFB laser: P(t) = k*(Ibias + Imod), E_in = sqrt(P).
    3. Propagates through the fiber via fibra_monomodo (regime 0 or 1).
    4. Detects with the PIN photodiode: I(t) = R * |E_out|^2.
    5. Measures the spectral amplitude via FFT at fc (removing DC).
    6. Normalizes the response relative to the first frequency.

    Returns:
        resposta_db: Response in dB along freqs_hz.
        resposta_linear: Linear response (FFT amplitude at fc).
    """
    if dfb_params is None:
        dfb_params = DFBParams()
    if fiber_params is None:
        fiber_params = FiberParams()

    resposta_lin = []

    for fc in freqs_hz:
        period_t = 1.0 / fc
        t_final = num_cycles * period_t
        t = np.linspace(0.0, t_final, np_points, endpoint=False)

        # 1. CW RF signal
        sinal = np.cos(2.0 * np.pi * fc * t)
        s_min, s_max = np.min(sinal), np.max(sinal)
        i_mod = dfb_params.i_factor * (sinal - s_min) / (s_max - s_min + 1e-30)

        # 2. DFB Laser
        _, ent_fibra = laser_dfb_simplificado(i_mod, params=dfb_params)

        # 3. Fiber Propagation
        sai_fibra, _, _ = fibra_monomodo(
            t=t,
            u=ent_fibra,
            wavelength=wavelength_m,
            t0=period_t,
            length_km=length_km,
            regime=regime,
            params=fiber_params,
            num_steps=num_ssfm_steps,
        )

        # 4. PIN Photodetector
        corrente = fotodiodo_pin(sai_fibra, responsivity=0.75)

        # 5. FFT
        foto_fft = np.fft.fft(corrente)
        foto_fft[0] = 0.0  # Remove DC component (continuous term)

        # Peak around the fundamental carrier frequency
        amp_pico = np.max(np.abs(foto_fft[: np_points // 2]))
        resposta_lin.append(amp_pico)

    resposta_lin = np.array(resposta_lin)
    ref_val = resposta_lin[0] if resposta_lin[0] > 0 else 1.0
    resposta_db = 20.0 * np.log10(np.maximum(resposta_lin, 1e-12) / ref_val)

    return resposta_db, resposta_lin
