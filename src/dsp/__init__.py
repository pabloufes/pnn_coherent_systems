"""Optical digital signal processing (DSP) modules."""

from src.dsp.edfa import EDFAIdeal
from src.dsp.joint_pnn_cdc_equalizer import (
    cd_compensation_joint_pnn,
    cd_compensation_traditional,
    simulate_fiber_dispersion,
)

__all__ = [
    "EDFAIdeal",
    "cd_compensation_joint_pnn",
    "cd_compensation_traditional",
    "simulate_fiber_dispersion",
]
