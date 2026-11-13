"""Backward compatibility module for the Photonic Neural Network (PNN).

The canonical implementation resides in :mod:`src.models.pnn`.
"""

from src.models.pnn import PhotonicNeuralNetwork

__all__ = ["PhotonicNeuralNetwork"]


if __name__ == "__main__":
    import numpy as np

    pnn = PhotonicNeuralNetwork()
    print(f"[Compatibility OK] PNN instantiated with {pnn.num_channels} channels.")
