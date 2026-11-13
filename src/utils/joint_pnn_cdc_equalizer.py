"""Backward compatibility module for the Joint PNN-CDC Equalizer.

The canonical implementation resides in :mod:`src.dsp.joint_pnn_cdc_equalizer`.
"""

from src.dsp.joint_pnn_cdc_equalizer import (
    PhotonicNeuralNetwork,
    simulate_fiber_dispersion,
    cd_compensation_traditional,
    cd_compensation_joint_pnn,
    calculate_evm_qpsk,
)

__all__ = [
    "PhotonicNeuralNetwork",
    "simulate_fiber_dispersion",
    "cd_compensation_traditional",
    "cd_compensation_joint_pnn",
    "calculate_evm_qpsk",
]
