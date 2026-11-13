"""Optical QPSK transmitter in ideal and IQ Mach-Zehnder modes.

Generates bit sequences, Gray-coded QPSK symbols,
continuous oversampling with shaped pulses and optical launch power control.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from src.utils.filtros import aplicar_filtro


def dbm_to_watts(power_dbm: float) -> float:
    """Converts power from dBm to Watts."""
    return 10.0 ** ((power_dbm - 30.0) / 10.0)


@dataclass(frozen=True)
class ElectricalFilterParams:
    """Parameters of the electrical filter applied to the MZM drive signals.

    The filter is applied to the I and Q drive currents (real baseband signals)
    before optical modulation. Uses the same kernel as ``calcular_resposta_filtro``
    from ``filtros.py``.

    Attributes
    ----------
    tipo       : ``'gaussiano'`` | ``'cosseno_levantado'`` | ``'butterworth'``
    largura_banda_hz : filter bandwidth [Hz].
                        For gaussian/butterworth: -3 dB bandwidth.
                        For raised cosine: symbol rate Rs.
                        If ``None``, the filter is not applied.
    beta       : roll-off of the raised cosine (0 to 1).
    ordem      : Butterworth order or super-Gaussian exponent.
    """

    tipo: str = "gaussiano"
    largura_banda_hz: Optional[float] = None  # None = filter disabled
    beta: float = 0.5
    ordem: int = 1


@dataclass(frozen=True)
class IQMZMParams:
    """Electrical and quadrature parameters of the IQ modulator.

    The default values for Vπ and the phase between branches follow the
    ``MZ_Input_Data_DP.m`` and ``Mach_Zehnder_Modulator_DP.m`` files. Each sub-MZM
    uses the dual-drive equation from ``Mach_Zehnder_Modulator_simplificado.m``.

    ``null_bias_i/q`` are the differential bias voltages [V] for
    push-pull driving. This mode cancels the chirp term in the simplified
    dual-drive equation. The distributed electrode response from MATLAB
    DP is optional and disabled by default.
    """

    v_pi_i: float = 3.1
    v_pi_q: float = 3.25
    null_bias_i: float = 3.1
    null_bias_q: float = 3.25
    quadrature_phase_rad: float = -np.pi / 2.0
    electrical_response: bool = False
    electrode_length_cm: float = 10.0
    n_optical: float = 2.17
    n_electrical: float = 2.60
    alpha0_linear: float = 10.0 ** (0.55 / 20.0)


def dual_drive_mzm(
    ein: np.ndarray,
    u1: np.ndarray,
    u2: np.ndarray,
    v_pi: float,
    u0: float,
) -> np.ndarray:
    """Applies the field transfer of a dual-drive MZM.

    Port of the equation from ``Mach_Zehnder_Modulator_simplificado.m``:

    ``Eout = Ein*cos(pi/2*(U1-U2-U0)/Vpi)*exp(-j*pi/2*(U1+U2)/Vpi)``.
    """
    if v_pi <= 0:
        raise ValueError("v_pi must be positive.")

    ein = np.asarray(ein, dtype=complex)
    u1 = np.asarray(u1, dtype=float)
    u2 = np.asarray(u2, dtype=float)
    phase_difference = (np.pi / 2.0) * (u1 - u2 - u0) / v_pi
    phase_chirp = -(np.pi / 2.0) * (u1 + u2) / v_pi
    return ein * np.cos(phase_difference) * np.exp(1j * phase_chirp)


def _iq_mzm_electrode_response(
    voltage: np.ndarray,
    dt: float,
    params: IQMZMParams,
) -> np.ndarray:
    """Filters a voltage through the distributed electrode response of MATLAB DP."""
    if dt <= 0:
        raise ValueError("dt must be positive to apply the electrode response.")
    if params.electrode_length_cm <= 0:
        raise ValueError("electrode_length_cm must be positive.")

    freq_ghz = np.fft.fftfreq(len(voltage), d=dt) / 1e9
    alpha_f = 0.5 * params.alpha0_linear * np.sqrt(np.abs(freq_ghz))
    gamma_f = (
        2.0 * np.pi * freq_ghz
        * (params.n_electrical - params.n_optical) / 30.0
    )
    attenuation = (alpha_f + 1j * gamma_f) * params.electrode_length_cm

    response = np.ones_like(attenuation, dtype=complex)
    nonzero = np.abs(attenuation) > 1e-14
    response[nonzero] = -np.expm1(-attenuation[nonzero]) / attenuation[nonzero]
    # The signed frequency in gamma_f ensures H(-f) = conj(H(f)) and,
    # therefore, that the filtered voltage remains real.
    filtered = np.fft.ifft(np.fft.fft(voltage) * response)
    return filtered.real


def _plot_electrical_filter(
    before: list[np.ndarray],
    after: list[np.ndarray],
    dt: float,
    filter_params: ElectricalFilterParams,
    save_path: Optional[str] = None,
    n_plot: Optional[int] = None,
    sps: Optional[int] = None,
) -> None:
    """Compares the drives before/after the filter in the requested window."""
    import matplotlib.pyplot as plt

    names = ("I branch 1", "I branch 2", "Q branch 1", "Q branch 2")
    if n_plot is None:
        n_samples = before[0].size
    else:
        n_plot = int(n_plot)
        if n_plot <= 0:
            raise ValueError("n_plot must be positive.")
        n_samples = n_plot * sps if sps is not None else n_plot
        n_samples = min(n_samples, before[0].size)

    if sps is not None:
        time_axis = np.arange(n_samples) / sps
        xlabel = "Time [symbols]"
    else:
        time_axis = np.arange(n_samples) * dt * 1e9
        xlabel = "Time [ns]"

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for ax, name, raw, filtered in zip(axes.flat, names, before, after):
        ax.plot(time_axis, raw[:n_samples], label="Before filter", alpha=0.8)
        ax.plot(time_axis, filtered[:n_samples], label="After filter", linewidth=1.5)
        ax.set_title(name)
        ax.set_ylabel("Voltage [V]")
        ax.grid(True, alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel(xlabel)
    axes[0, 0].legend()
    fig.suptitle(
        f"Electrical drive: {filter_params.tipo} filter, "
        f"B = {filter_params.largura_banda_hz / 1e9:g} GHz"
    )
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def iq_mzm_field(
    ein: np.ndarray,
    u_i1: np.ndarray,
    u_i2: np.ndarray,
    u_q1: np.ndarray,
    u_q2: np.ndarray,
    dt: float,
    params: IQMZMParams = None,
    apply_electrical_response: bool = None,
    filter_params: ElectricalFilterParams = None,
    plot_filter: bool = False,
    filter_plot_path: str = None,
    n_plot: int = None,
    sps: int = None,
) -> np.ndarray:
    """Combines two sub-MZMs in quadrature and returns the IQ optical field.

    Each branch uses the simplified dual-drive model. The ``1/2`` factor per
    branch and the quadrature phase follow the field combination in the DP-MZM
    MATLAB. The voltages must have the same shape as ``ein``.

    The electrical filter in ``filter_params`` is applied **before** the
    distributed electrode response (when active), replicating the physical order
    of a real driver followed by a modulation electrode.

    ``n_plot`` limits the comparison plot. Since ``sps`` is provided by the
    modulator, the value represents symbols; ``None`` shows the entire sequence.
    """
    if params is None:
        params = IQMZMParams()

    ein = np.asarray(ein, dtype=complex)
    voltages = [np.asarray(v, dtype=float) for v in (u_i1, u_i2, u_q1, u_q2)]
    if any(v.shape != ein.shape for v in voltages):
        raise ValueError("ein and the four branch voltages must have the same shape.")
    if params.v_pi_i <= 0 or params.v_pi_q <= 0:
        raise ValueError("v_pi_i and v_pi_q must be positive.")

    # 1) Bandpass electrical filter (driver / DAC modeling)
    if filter_params is not None and filter_params.largura_banda_hz is not None:
        if dt <= 0:
            raise ValueError("dt must be positive to apply the electrical filter.")
        voltages_before_filter = [v.copy() for v in voltages]
        fs = 1.0 / dt
        voltages = [
            aplicar_filtro(
                v, fs,
                tipo=filter_params.tipo,
                B=filter_params.largura_banda_hz,
                f0=0.0,
                beta=filter_params.beta,
                ordem=filter_params.ordem,
                plot=False,
                n_plot=n_plot,
                sps=sps,
            )[0]
            for v in voltages
        ]
        if plot_filter:
            _plot_electrical_filter(
                voltages_before_filter,
                voltages,
                dt,
                filter_params,
                save_path=filter_plot_path,
                n_plot=n_plot,
                sps=sps,
            )

    # 2) Distributed electrode response (optional)
    use_response = (
        params.electrical_response
        if apply_electrical_response is None
        else apply_electrical_response
    )
    if use_response:
        voltages = [_iq_mzm_electrode_response(v, dt, params) for v in voltages]

    e_i = 0.5 * dual_drive_mzm(
        ein, voltages[0], voltages[1], params.v_pi_i, params.null_bias_i
    )
    e_q = 0.5 * dual_drive_mzm(
        ein, voltages[2], voltages[3], params.v_pi_q, params.null_bias_q
    )
    return e_i + e_q * np.exp(1j * params.quadrature_phase_rad)


def modulate_iq_mzm(
    symbols: np.ndarray,
    sps: int = 8,
    launch_power_dbm: float = 0.0,
    symbol_rate: float = 10e9,
    params: IQMZMParams = None,
    normalize_output_power: bool = True,
    filter_params: ElectricalFilterParams = None,
    plot_filter: bool = False,
    filter_plot_path: str = None,
    n_plot: int = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Generates QPSK via two dual-drive sub-MZMs combined in quadrature.

    The symbols determine the push-pull electrical levels for I and Q. The Q
    branch receives the signal inversion required for the ``-pi/2`` phase used
    in the MATLAB DP-MZM. By default, the output field is normalized so that
    ``launch_power_dbm`` continues to represent the average power launched into
    the fiber, as in ``ideal`` mode.

    With ``normalize_output_power=False``, ``launch_power_dbm`` represents
    the input optical power and the intrinsic division of the IQ-MZM reduces the
    output power.

    The optional parameter ``filter_params`` applies a low-pass filter
    to the electrical drive signals (I and Q) before the MZM, modeling the
    frequency response of the DAC / electrical driver.

    ``n_plot`` defines how many symbols appear in the comparison plot
    when ``plot_filter=True``. If ``None``, the plot shows everything.
    """
    if params is None:
        params = IQMZMParams()
    if not (sps > 0 and (sps & (sps - 1)) == 0):
        raise ValueError(f"sps must be a power of 2. Received: {sps}")
    if symbol_rate <= 0:
        raise ValueError("symbol_rate must be positive.")

    symbols = np.asarray(symbols, dtype=complex)
    if symbols.ndim != 1 or symbols.size == 0:
        raise ValueError("symbols must be a non-empty one-dimensional vector.")

    dt = (1.0 / symbol_rate) / sps
    t = np.arange(symbols.size * sps) * dt
    i_sign = np.where(np.real(symbols) >= 0.0, 1.0, -1.0)
    q_sign = -np.where(np.imag(symbols) >= 0.0, 1.0, -1.0)
    i_drive = np.repeat(i_sign, sps) * (params.v_pi_i / 2.0)
    q_drive = np.repeat(q_sign, sps) * (params.v_pi_q / 2.0)

    target_power_w = dbm_to_watts(launch_power_dbm)
    ein = np.full(t.size, np.sqrt(target_power_w), dtype=complex)
    u_tx = iq_mzm_field(
        ein,
        i_drive,
        -i_drive,
        q_drive,
        -q_drive,
        dt,
        params=params,
        filter_params=filter_params,
        plot_filter=plot_filter,
        filter_plot_path=filter_plot_path,
        n_plot=n_plot,
        sps=sps,
    )

    if normalize_output_power:
        output_power = np.mean(np.abs(u_tx) ** 2)
        if output_power <= 0:
            raise ValueError("The IQ-MZM produced zero output power.")
        u_tx *= np.sqrt(target_power_w / output_power)

    return u_tx, t, dt


def generate_qpsk_symbols(num_symbols: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generates a random bit sequence and Gray-mapped QPSK symbols.

    Ensures that num_symbols is a power of 2.

    Gray QPSK mapping normalized to unit power:
      00 -> ( 1 + 1j) / sqrt(2)
      01 -> ( 1 - 1j) / sqrt(2)
      11 -> (-1 - 1j) / sqrt(2)
      10 -> (-1 + 1j) / sqrt(2)

    Returns:
        symbols: Vector of normalized complex symbols E[|s|^2] = 1.
        bits: Vector of transmitted bits (size: 2 * num_symbols).
    """
    if not (num_symbols > 0 and (num_symbols & (num_symbols - 1)) == 0):
        raise ValueError(f"num_symbols must be a power of 2. Received: {num_symbols}")

    rng = np.random.default_rng(seed)
    # 2 bits per symbol for QPSK
    bits = rng.integers(0, 2, size=2 * num_symbols)

    bit_i = bits[0::2]
    bit_q = bits[1::2]

    # BPSK mapping in each quadrant (0 -> +1, 1 -> -1)
    sym_i = 1.0 - 2.0 * bit_i
    sym_q = 1.0 - 2.0 * bit_q

    symbols = (sym_i + 1j * sym_q) / np.sqrt(2.0)
    return symbols, bits


def modulate_optical_signal(
    symbols: np.ndarray,
    sps: int = 8,
    launch_power_dbm: float = 0.0,
    symbol_rate: float = 10e9,
    pulse_type: str = "nrz",
    modulator: str = "ideal",
    iq_params: IQMZMParams = None,
    normalize_output_power: bool = True,
    filter_params: ElectricalFilterParams = None,
    plot_filter: bool = False,
    filter_plot_path: str = None,
    n_plot: int = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Modulates discrete symbols into a continuous optical field envelope.

    Args:
        symbols: QPSK symbol vector.
        sps: Samples per symbol (must be a power of 2).
        launch_power_dbm: Average optical launch power into the fiber [dBm].
        symbol_rate: Symbol rate [Baud] (e.g.: 10e9 for 10 GBaud).
        pulse_type: Pulse type ('nrz' rectangular, etc.).
        modulator: ``"ideal"`` (default) or ``"iq_mzm"``.
        iq_params: IQ-MZM parameters, used when ``modulator="iq_mzm"``.
        normalize_output_power: Keeps the average power at the IQ-MZM output
            equal to ``launch_power_dbm``.
        filter_params: Optional electrical filter on the MZM drive signals.
            Ignored when ``modulator="ideal"``.
        plot_filter: Plots the electrical drives before/after the filter (only
            when ``modulator="iq_mzm"`` and a filter is active).
        filter_plot_path: If provided, saves the plot to this path instead of
            opening a window. Useful in scripts with a non-interactive backend.
        n_plot: Number of symbols displayed in the filter plot. ``None``
            shows all symbols.

    Returns:
        u_tx: Optical envelope [sqrt(W)].
        t: Time vector [s].
        dt: Sampling interval [s].
    """
    if not (sps > 0 and (sps & (sps - 1)) == 0):
        raise ValueError(f"sps must be a power of 2. Received: {sps}")

    modulator = modulator.lower()
    if modulator == "iq_mzm":
        return modulate_iq_mzm(
            symbols=symbols,
            sps=sps,
            launch_power_dbm=launch_power_dbm,
            symbol_rate=symbol_rate,
            params=iq_params,
            normalize_output_power=normalize_output_power,
            filter_params=filter_params,
            plot_filter=plot_filter,
            filter_plot_path=filter_plot_path,
            n_plot=n_plot,
        )
    if modulator != "ideal":
        raise ValueError("modulator must be 'ideal' or 'iq_mzm'.")

    num_symbols = len(symbols)
    total_samples = num_symbols * sps

    dt = (1.0 / symbol_rate) / sps
    t = np.arange(total_samples) * dt

    if pulse_type == "nrz":
        # NRZ pulse: repetition of the symbol for 'sps' consecutive samples
        u_base = np.repeat(symbols, sps)
    else:
        u_base = np.repeat(symbols, sps)

    power_watts = dbm_to_watts(launch_power_dbm)
    current_power = np.mean(np.abs(u_base)**2)
    u_tx = u_base * np.sqrt(power_watts / (current_power + 1e-30))

    return u_tx, t, dt
