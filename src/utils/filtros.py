import numpy as np
import matplotlib.pyplot as plt


def calcular_resposta_filtro(
    N: int,
    fs: float,
    tipo: str = "gaussiano",
    B: float = None,
    f0: float = 0.0,
    beta: float = 0.1,
    ordem: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the frequency response of an analog filter in the FFT domain.

    Parameters
    ----------
    N     : number of signal samples
    fs    : sampling rate [Hz]
    tipo  : ``'gaussiano'`` | ``'cosseno_levantado'`` | ``'butterworth'``
    B     : bandwidth [Hz]
              - gaussian / butterworth : full width at -3 dB
              - raised cosine          : symbol rate Rs = 1/Ts
    f0    : center frequency [Hz] (0 for baseband low-pass)
    beta  : roll-off do cosseno levantado (0 a 1)
    ordem : ordem do Butterworth ou expoente super-gaussiano

    Returns
    -------
    H : frequency response (in ``np.fft.fftfreq`` order)
    f : frequency vector [Hz]
    """
    if B is None:
        raise ValueError("Set the filter bandwidth B.")

    f = np.fft.fftfreq(N, d=1.0 / fs)

    # For low-pass filtering (f0 == 0), double the bandwidth to cover ±B/2.
    B_efetivo = 2 * B if f0 == 0.0 else B

    # Spectral distance from the center frequency.
    d_abs = np.abs(f - f0)

    tipo = tipo.lower()
    if tipo == "gaussiano":
        n = max(1, ordem)
        H = np.exp(-np.log(np.sqrt(2)) * (d_abs / (B_efetivo / 2)) ** (2 * n))

    elif tipo == "cosseno_levantado":
        Ts = 1.0 / B
        f1 = (1 - beta) / (2 * Ts)
        f2 = (1 + beta) / (2 * Ts)
        H = np.zeros(N, dtype=float)
        H[d_abs <= f1] = 1.0
        zona = (d_abs > f1) & (d_abs < f2)
        H[zona] = 0.5 + 0.5 * np.cos(np.pi * Ts / beta * (d_abs[zona] - f1))

    elif tipo == "butterworth":
        H = 1.0 / np.sqrt(1.0 + (d_abs / (B_efetivo / 2)) ** (2 * ordem))

    else:
        raise ValueError(
            "tipo must be 'gaussiano', 'cosseno_levantado', or 'butterworth'."
        )

    return H, f


def aplicar_filtro(
    x: np.ndarray,
    fs: float,
    tipo: str = "gaussiano",
    B: float = None,
    f0: float = 0.0,
    beta: float = 0.1,
    ordem: int = 2,
    plot: bool = True,
    n_plot: int = None,
    sps: int = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a filter (in the frequency domain) to a modulated signal.

    Parameters
    ----------
    x      : input signal (complex baseband or real passband)
    fs     : sampling rate [Hz]
    tipo   : ``'gaussiano'`` | ``'cosseno_levantado'`` | ``'butterworth'``
    B      : filter bandwidth [Hz]
               - gaussian / butterworth : full width at -3 dB
               - raised cosine          : symbol rate Rs (= 1/Ts)
    f0     : filter center frequency [Hz] (0 for baseband)
    beta   : roll-off do cosseno levantado (0 a 1)
    ordem  : ordem do Butterworth (ou super-gaussiano, se > 1 no gaussiano)
    plot   : enable/disable the time-domain comparison plot
    n_plot : number of points displayed. With ``sps``, this is the number of symbols;
             without ``sps``, it is the number of samples. ``None`` displays all.
    sps    : samples per symbol; when supplied, ``n_plot`` is counted in
             symbols and the X axis is also shown in symbols.

    Returns
    -------
    y : filtered signal
    H : frequency response used (in ``np.fft.fftfreq`` order)
    f : frequency vector [Hz]
    """
    x = np.asarray(x)
    N = len(x)
    eh_real = np.isrealobj(x)

    H, f = calcular_resposta_filtro(N, fs, tipo, B, f0, beta, ordem)

    y = np.fft.ifft(np.fft.fft(x) * H)
    if eh_real:
        y = y.real

    if plot:
        if sps is not None and sps <= 0:
            raise ValueError("sps must be positive.")
        if n_plot is None:
            n = N
        else:
            n_plot = int(n_plot)
            if n_plot <= 0:
                raise ValueError("n_plot must be positive.")
            n_amostras = n_plot * sps if sps is not None else n_plot
            n = min(n_amostras, N)
        t = np.arange(n) / sps if sps else np.arange(n) / fs
        xlabel = "Time [symbols]" if sps else "Time [s]"

        plt.figure(figsize=(10, 4))
        if np.iscomplexobj(x):
            plt.plot(t, x[:n].real, label="Unfiltered (Re)", alpha=0.6)
            plt.plot(t, y[:n].real, label="Filtered (Re)", lw=2)
            plt.plot(t, x[:n].imag, "--", label="Unfiltered (Im)", alpha=0.6)
            plt.plot(t, y[:n].imag, "--", label="Filtered (Im)", lw=2)
        else:
            plt.plot(t, x[:n], label="Unfiltered", alpha=0.6)
            plt.plot(t, y[:n], label="Filtered", lw=2)
        plt.xlabel(xlabel)
        plt.ylabel("Amplitude")
        B_label = B if B else 0.0
        plt.title(f"{tipo} filter | B = {B_label / 1e9:.2f} GHz | f0 = {f0 / 1e9:.2f} GHz")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return y, H, f
