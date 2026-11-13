"""
===============================================================================
Module: Silicon Photonic Neural Network (PNN - Photonic Neural Network)
===============================================================================
Description:
    Theoretical model of a Photonic Neural Network (PNN) based on a
    Tapped-Delay Line (TDL) structure in a silicon photonic integrated
    circuit (PIC).

    This module implements the equivalent behavior of a Finite Impulse Response
    (FIR) filter in the optical domain with N parallel channels, complex
    amplitude (a_i) and phase (phi_i) weights, and a non-linear activation
    function via quadratic detection at the photodetector (|E|^2).

Mathematical Formulation:
    1. Time delay of the i-th channel:
        τ_i = i * Δt ,  for i = 0, 1, ..., N-1

    2. Complex weights of each channel:
        w_i = a_i * exp(j * φ_i)
        where a_i ∈ [0, 1] (transmittance/amplitude) and φ_i ∈ [0, 2π) (phase shift).

    3. Optical Combiner (Interferometric Sum of Fields):
        E_sum(t) = (1 / N) * ∑_{i=0}^{N-1} a_i * exp(j * φ_i) * E_in(t - i * Δt)

    4. Optical Frequency Response H(f):
        H(f) = (1 / N) * ∑_{i=0}^{N-1} a_i * exp(j * φ_i) * exp(-j * 2π * f * i * Δt)

    5. Quadratic Detection (Photonic Activation Function):
        I_out(t) = R * |E_sum(t)|²

References:
    - Marciano, P. R. N. (2024). Doctoral Thesis, UFES (FIR/PNN Filter in Silicon).
    - Staffoli, E., et al. (2024/2025). Journal of Lightwave Tech. & Laser Phot. Rev.
    - Mancinelli, M., et al. (2022). Scientific Reports (Complex Photonic Perceptron).
===============================================================================
"""

from __future__ import annotations
import numpy as np


class PhotonicNeuralNetwork:
    """Class representing the Theoretical Photonic Neural Network (PNN).

    Attributes:
        num_channels (int): Number of parallel channels/paths (N). Default: 8.
        base_delay_ps (float): Incremental base delay per channel Δt [ps]. Default: 25.0 ps.
        amplitude_weights (np.ndarray): Amplitude weight vector a_i ∈ [0, 1].
        phase_weights (np.ndarray): Phase weight vector φ_i [rad].
        normalize_power (bool): If True, normalizes the weights by the channel division (1/N).
        responsivity (float): Photodetector responsivity R [A/W]. Default: 1.0.
    """

    def __init__(
        self,
        num_channels: int = 8,
        base_delay_ps: float = 25.0,
        amplitude_weights: np.ndarray | None = None,
        phase_weights: np.ndarray | None = None,
        normalize_power: bool = True,
        responsivity: float = 1.0,
    ):
        if isinstance(num_channels, (bool, np.bool_)) or not isinstance(
            num_channels, (int, np.integer)
        ) or num_channels <= 0:
            raise ValueError("num_channels must be a positive integer.")
        base_delay_ps = self._validate_nonnegative_scalar(base_delay_ps, "base_delay_ps")
        responsivity = self._validate_nonnegative_scalar(responsivity, "responsivity")
        if not isinstance(normalize_power, (bool, np.bool_)):
            raise ValueError("normalize_power must be boolean.")

        self.num_channels = int(num_channels)
        self.base_delay_ps = float(base_delay_ps)
        self.base_delay_s = self.base_delay_ps * 1e-12  # Converts ps to seconds
        self.normalize_power = normalize_power
        self.responsivity = float(responsivity)

        if amplitude_weights is None:
            amplitude_weights = np.ones(self.num_channels, dtype=float)
        if phase_weights is None:
            phase_weights = np.zeros(self.num_channels, dtype=float)
        self.amplitude_weights, self.phase_weights = self._validate_weights(
            amplitude_weights, phase_weights
        )

    @staticmethod
    def _validate_nonnegative_scalar(value, name: str) -> float:
        """Validates a real, finite and non-negative scalar parameter."""
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
            raise ValueError(f"{name} must be a real, finite and non-negative scalar.")
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{name} must be a real, finite and non-negative scalar."
            ) from exc
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative.")
        return value

    def _validate_weights(self, amplitude_weights, phase_weights):
        """Validates dimension, length, finiteness and range of the weights."""
        try:
            amplitudes = np.asarray(amplitude_weights, dtype=float)
            phases = np.asarray(phase_weights, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("The weights must contain real numeric values.") from exc

        for weights, name in ((amplitudes, "amplitude_weights"), (phases, "phase_weights")):
            if weights.ndim != 1:
                raise ValueError(f"{name} must be a one-dimensional vector.")
            if weights.size != self.num_channels:
                raise ValueError(f"{name} must have size {self.num_channels}.")
            if not np.all(np.isfinite(weights)):
                raise ValueError(f"{name} must contain only finite values.")
        if np.any((amplitudes < 0) | (amplitudes > 1)):
            raise ValueError("amplitude_weights must be in the range [0, 1].")
        return amplitudes.copy(), phases.copy()

    @property
    def complex_weights(self) -> np.ndarray:
        """Returns the complex weights w_i = a_i * exp(j * φ_i)."""
        w = self.amplitude_weights * np.exp(1j * self.phase_weights)
        if self.normalize_power:
            w = w / self.num_channels
        return w

    def set_weights(self, amplitude_weights: np.ndarray, phase_weights: np.ndarray):
        """Simultaneously updates the amplitude and phase weights of the PNN."""
        self.amplitude_weights, self.phase_weights = self._validate_weights(
            amplitude_weights, phase_weights
        )

    def frequency_response(self, frequencies_hz: np.ndarray) -> np.ndarray:
        """Calculates the complex Optical Frequency Response H(f) of the PNN.

        H(f) = ∑_{i=0}^{N-1} w_i * exp(-j * 2π * f * i * Δt)
        """
        frequencies_hz = np.asarray(frequencies_hz, dtype=float)
        if frequencies_hz.ndim != 1 or not np.all(np.isfinite(frequencies_hz)):
            raise ValueError("frequencies_hz must be a 1D vector of finite values.")
        w = self.complex_weights
        H = np.zeros(len(frequencies_hz), dtype=complex)
        for i in range(self.num_channels):
            tau_i = i * self.base_delay_s
            H += w[i] * np.exp(-1j * 2.0 * np.pi * frequencies_hz * tau_i)
        return H

    def forward(
        self,
        optical_field: np.ndarray,
        dt_s: float,
        return_optical_field: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Processes the input optical field E_in(t) through the PNN structure.

        Args:
            optical_field: Complex 1D vector representing the optical field E_in(t).
            dt_s: Signal sampling period in seconds (dt_s = 1 / sampling_frequency).
            return_optical_field: If True, also returns the summed optical field E_sum(t).

        Returns:
            photocurrent: Photodetected electrical signal I_out(t) = R * |E_sum(t)|².
            E_sum (optional): Recombined optical field before photodetection.
        """
        if isinstance(dt_s, (bool, np.bool_)) or not np.isscalar(dt_s):
            raise ValueError("dt_s must be a positive and finite scalar.")
        try:
            dt_s = float(dt_s)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("dt_s must be a positive and finite scalar.") from exc
        if not np.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("dt_s must be positive and finite.")

        try:
            optical_field = np.asarray(optical_field, dtype=complex)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("optical_field must contain finite numeric values.") from exc
        if optical_field.ndim != 1 or optical_field.size == 0:
            raise ValueError("optical_field must be a non-empty 1D vector.")
        if not np.all(np.isfinite(optical_field)):
            raise ValueError("optical_field must contain only finite values.")

        num_samples = optical_field.size
        E_sum = np.zeros(num_samples, dtype=complex)
        w = self.complex_weights

        for i in range(self.num_channels):
            tau_i = i * self.base_delay_s
            shift_samples = tau_i / dt_s

            if np.abs(shift_samples - round(shift_samples)) < 1e-5:
                # Exact delay in integer samples
                k_shift = int(round(shift_samples))
                if k_shift < num_samples:
                    E_delayed = np.zeros(num_samples, dtype=complex)
                    E_delayed[k_shift:] = optical_field[: num_samples - k_shift]
                else:
                    E_delayed = np.zeros(num_samples, dtype=complex)
            else:
                # Fractional delay in the frequency domain
                freqs = np.fft.fftfreq(num_samples, d=dt_s)
                phase_shift = np.exp(-1j * 2.0 * np.pi * freqs * tau_i)
                E_delayed = np.fft.ifft(np.fft.fft(optical_field) * phase_shift)

            E_sum += w[i] * E_delayed

        # Quadratic Detection (Photonic Activation Function)
        photocurrent = self.responsivity * np.real(E_sum * np.conj(E_sum))

        if return_optical_field:
            return photocurrent, E_sum
        return photocurrent
