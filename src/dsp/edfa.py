"""Models for Erbium-Doped Fiber Amplifier (EDFA).

This module keeps the EDFA as an optical device separate from the propagation
algorithm. The ideal model serves as a reference; future analytical models
may implement the same interface ``amplify(optical_field)``.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class EDFAIdeal:
    """Ideal EDFA with fixed gain, no ASE, saturation, or spectral dependence.

    ``gain_db`` is the power gain in dB. The complex optical field is
    multiplied by the square root of the corresponding linear gain.
    """

    gain_db: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.gain_db) or self.gain_db < 0.0:
            raise ValueError("gain_db must be a finite and non-negative value.")

    @property
    def field_gain(self) -> float:
        """Linear factor applied to the optical field amplitude."""
        return 10.0 ** (self.gain_db / 20.0)

    def amplify(self, optical_field: np.ndarray) -> np.ndarray:
        """Amplifies the optical field without modifying it in place."""
        field = np.asarray(optical_field, dtype=complex)
        return field * self.field_gain
