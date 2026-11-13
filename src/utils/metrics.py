"""Calculation of optical performance metrics: BER and EVM.
"""

import numpy as np


def calculate_ber(bits_tx: np.ndarray, bits_rx: np.ndarray) -> float:
    """Calculates the Bit Error Rate (BER)."""
    min_len = min(len(bits_tx), len(bits_rx))
    errors = np.sum(bits_tx[:min_len] != bits_rx[:min_len])
    return float(errors) / float(min_len)


def calculate_evm_norm(tx_symbols: np.ndarray, rx_symbols: np.ndarray) -> float:
    """Calculates the Error Vector Magnitude (EVM) in percentage (%)."""
    min_len = min(len(tx_symbols), len(rx_symbols))
    s_tx = tx_symbols[:min_len]
    s_rx = rx_symbols[:min_len]

    # Average power normalization of signals
    norm_tx = s_tx / np.sqrt(np.mean(np.abs(s_tx)**2) + 1e-30)
    norm_rx = s_rx / np.sqrt(np.mean(np.abs(s_rx)**2) + 1e-30)

    # Best scale factor and constant phase rotation
    rot_opt = np.angle(np.mean(norm_rx * np.conj(norm_tx)))
    norm_rx_rot = norm_rx * np.exp(-1j * rot_opt)

    error_vector = norm_rx_rot - norm_tx
    evm_rms = np.sqrt(np.mean(np.abs(error_vector)**2))
    return float(evm_rms * 100.0)


def calculate_evm(
    rx_symbols: np.ndarray,
    ideal_constellation: np.ndarray = None,
    correct_phase_rotation: bool = False,
) -> float:
    """Calcula o Error Vector Magnitude (EVM RMS) em porcentagem (%) com base na constelação ideal.

    Args:
        rx_symbols: Vetor 1D de símbolos complexos recebidos.
        ideal_constellation: Array contendo os pontos da constelação ideal.
            Se None, utiliza por padrão a constelação QPSK normalizada com potência unitária.
        correct_phase_rotation: Se True, compensa a rotação constante de fase da
            constelação antes do cálculo do vetor de erro.

    Returns:
        float: EVM RMS em porcentagem (%).
    """
    if len(rx_symbols) == 0:
        return 0.0

    # Constelação ideal padrão (QPSK normalizada com potência média unitária)
    if ideal_constellation is None:
        ideal_constellation = np.array([1.0 + 1.0j, 1.0 - 1.0j, -1.0 + 1.0j, -1.0 - 1.0j]) / np.sqrt(2.0)

    # Normalização de potência média do sinal recebido e da constelação ideal
    p_rx = np.mean(np.abs(rx_symbols) ** 2)
    norm_rx = rx_symbols / np.sqrt(p_rx + 1e-30)

    p_ideal = np.mean(np.abs(ideal_constellation) ** 2)
    norm_ideal = ideal_constellation / np.sqrt(p_ideal + 1e-30)

    # 1. Correção opcional da rotação constante de fase
    if correct_phase_rotation:
        # Decisão preliminar para estimar a rotação média em relação aos pontos ideais
        dists_prelim = np.abs(norm_rx[:, None] - norm_ideal[None, :])
        closest_prelim = norm_ideal[np.argmin(dists_prelim, axis=1)]
        rot_opt = np.angle(np.mean(norm_rx * np.conj(closest_prelim)))
        norm_rx = norm_rx * np.exp(-1j * rot_opt)

    # 2. Decisão do ponto da constelação ideal mais próximo para cada símbolo recebido (slicing)
    distances = np.abs(norm_rx[:, None] - norm_ideal[None, :])
    closest_idx = np.argmin(distances, axis=1)
    s_ref = norm_ideal[closest_idx]

    # 3. Cálculo do vetor de erro e EVM RMS
    error_vector = norm_rx - s_ref
    evm_rms = np.sqrt(np.mean(np.abs(error_vector) ** 2))

    return float(evm_rms * 100.0)


if __name__ == "__main__":
    # Teste de verificação rápida do módulo
    np.random.seed(42)
    # Geração de 1000 símbolos QPSK ideais + ruído Gaussiano complexo
    qpsk_symbols = np.array([1.0 + 1.0j, 1.0 - 1.0j, -1.0 + 1.0j, -1.0 - 1.0j]) / np.sqrt(2.0)
    tx_indices = np.random.randint(0, 4, size=1000)
    s_tx = qpsk_symbols[tx_indices]

    # Adiciona ruído (SNR ~ 17 dB) + rotação de fase constante (15 graus)
    noise = (np.random.randn(1000) + 1j * np.random.randn(1000)) * 0.1
    phase_rot = np.exp(1j * np.radians(15.0))
    s_rx = (s_tx + noise) * phase_rot

    evm_with_corr = calculate_evm(s_rx, correct_phase_rotation=True)
    evm_no_corr = calculate_evm(s_rx, correct_phase_rotation=False)

    print(f"EVM com correção de rotação constante: {evm_with_corr:.2f}%")
    print(f"EVM sem correção de rotação constante: {evm_no_corr:.2f}%")

