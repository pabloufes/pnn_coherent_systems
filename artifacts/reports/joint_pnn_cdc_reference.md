# Documentação de Referência Técnica: Equalizador Conjunto PNN + CDC para Receptores Ópticos Coerentes

---

## 1. Visão Geral e Objetivo

Este documento apresenta a especificação matemática, o embasamento físico e os requisitos de integração do **Equalizador de Dispersão Cromática Conjunto PNN + DSP (`cd_compensation_joint_pnn`)**. 

O objetivo desta implementação é prover a um agente de IA ou projetista de sistemas de comunicação óptica o modelo exato para compensar simultaneamente a **Dispersão Cromática (CD)** acumulada na fibra óptica e a **filtragem de amplitude/fase introduzida por uma Rede Neural Fotônica (PNN)** posicionada no domínio óptico antes da fotodetecção coerente.

---

## 2. Contexto Físico e Problema Teórico

### 2.1 Arquitetura do Sistema

A cadeia de transmissão e recepção coerente com PNN segue a topologia:

```
[ Transmissor QPSK / M-QAM ]
             │
             ▼
[ Enlace de Fibra Óptica (SSMF, L km) ]  ──► H_fibra(f)
             │
             ▼
[ Rede Neural Fotônica em Silício (PNN) ] ──► H_PNN(f)
             │  (Campo Óptico Recombinado E_sum)
             ▼
[ Receptor Coerente (90° Hybrid + BPDs) ] ──► Recovers E_sum(t)
             │  (Sinal Elétrico em Banda Base I + jQ)
             ▼
[ Processamento Digital de Sinais (DSP) ] ──► H_CDC_joint(f)
```

### 2.2 Inadequação do CDC Tradicional

Em receptores coerentes ideais, o sinal elétrico digitalizado $r(t) = I(t) + j Q(t)$ é proporcional ao campo óptico incidente no híbrido de $90^\circ$. A resposta em frequência global vista pelo DSP é o produto em cascata:

$$H_{\text{canal}}(f) = H_{\text{fibra}}(f) \cdot H_{\text{PNN}}(f)$$

O algoritmo de CDC tradicional inverte **apenas** a função de transferência da fibra:

$$H_{\text{CDC\_tradicional}}(f) = H_{\text{fibra}}^{-1}(f) = \exp\left(+j \frac{1}{2} \beta_2 L (2\pi f)^2\right)$$

Ao aplicar $H_{\text{CDC\_tradicional}}(f)$, a saída do DSP permanece distorcida pela resposta da PNN:

$$Y_{\text{tradicional}}(f) = E_{\text{tx}}(f) \cdot H_{\text{PNN}}(f)$$

Essa distorção atua como **Interferência Intersimbólica (ISI) residual**, resultando no fechamento da constelação e estagnação da taxa de erro (EVM $\approx 60\%$).

---

## 3. Formulação Matemática Exata

### 3.1 Resposta da Fibra Óptica ($H_{\text{fibra}}$)

A propagação no regime linear de uma fibra monomodo padrão (SSMF) é governada por:

$$H_{\text{fibra}}(f) = \exp\left(-j \frac{1}{2} \beta_2 L (2\pi f)^2\right)$$

Onde:
- $L$: Comprimento do enlace em metros [m].
- $\beta_2$: Parâmetro de dispersão de segunda ordem $[\text{s}^2/\text{m}]$, calculado a partir do parâmetro $D [\text{ps/(nm}\cdot\text{km)}]$:
  $$\beta_2 = -\frac{\lambda^2 D}{2\pi c}$$

### 3.2 Resposta em Frequência Óptica da PNN ($H_{\text{PNN}}$)

A PNN é modelada como um filtro FIR complexo de $N$ canais paralelos com linhas de atraso temporal ($\Delta t$):

$$H_{\text{PNN}}(f) = \sum_{i=0}^{N-1} w_i \cdot e^{-j 2\pi f \cdot i \cdot \Delta t}$$

Com os pesos complexos $w_i$ normalizados para conservação de energia em circuitos fotônicos integrados (PICs) passivos sem amplificação:

$$w_i = \frac{a_i}{N} e^{j \phi_i}, \quad a_i \in [0, 1], \quad \phi_i \in [0, 2\pi)$$

### 3.3 Filtro Equalizador Conjunto Inverso ($H_{\text{CDC\_conjunto}}$)

Para reverter perfeitamente o canal combinado, a resposta inversa ideal é:

$$H_{\text{CDC\_conjunto\_ideal}}(f) = H_{\text{fibra}}^{-1}(f) \cdot H_{\text{PNN}}^{-1}(f) = \exp\left(+j \frac{1}{2} \beta_2 L (2\pi f)^2\right) \cdot \frac{1}{H_{\text{PNN}}(f)}$$

Para evitar instabilidade numérica e amplificação de ruído em frequências onde a PNN apresenta atenuação acentuada ($|H_{\text{PNN}}(f)| \to 0$), aplica-se a **Inversa Regularizada de Tikhonov**:

$$H_{\text{PNN}}^{-1}(f) = \frac{H_{\text{PNN}}^*(f)}{|H_{\text{PNN}}(f)|^2 + \varepsilon}$$

Portanto, a equação completa executada no DSP é:

$$H_{\text{CDC\_conjunto}}(f) = \exp\left(+j \frac{1}{2} \beta_2 L (2\pi f)^2\right) \cdot \frac{H_{\text{PNN}}^*(f)}{|H_{\text{PNN}}(f)|^2 + \varepsilon}$$

Onde $\varepsilon > 0$ é o fator de regularização (valor padrão típico: $10^{-3}$).

---

## 4. Algoritmo e Especificação de Código

### 4.1 Assinatura da Função `cd_compensation_joint_pnn`

```python
import numpy as np

def cd_compensation_joint_pnn(
    signal_rx: np.ndarray,
    dt: float,
    length_km: float,
    pnn_instance: PhotonicNeuralNetwork,
    dispersion_d: float = 16.0,
    wavelength: float = 1550e-9,
    c: float = 299792458.0,
    reg_factor: float = 1e-3,
) -> np.ndarray:
    """Equalizador CDC Conjunto PNN + DSP no Receptor Coerente.

    Args:
        signal_rx (np.ndarray): Vetor 1D complexo do sinal elétrico (I + jQ) amostrado a dt.
        dt (float): Intervalo de amostragem em segundos (dt = 1 / f_amostragem).
        length_km (float): Distância da fibra em km.
        pnn_instance (PhotonicNeuralNetwork): Instância da PNN contendo os pesos (a_i, phi_i) e Δt.
        dispersion_d (float): Parâmetro de dispersão D em ps/(nm*km). Padrão: 16.0.
        wavelength (float): Comprimento de onda da portadora em metros. Padrão: 1550e-9.
        c (float): Velocidade da luz em m/s. Padrão: 299792458.0.
        reg_factor (float): Fator de regularização ε para a inversa da PNN. Padrão: 1e-3.

    Returns:
        np.ndarray: Sinal complexo equalizado no domínio do tempo (tamanho igual a signal_rx).
    """
    n = len(signal_rx)
    length_m = length_km * 1e3
    freq_grid = np.fft.fftfreq(n, d=dt)
    omega = 2.0 * np.pi * freq_grid

    # 1. Inversa da Dispersão Cromática da Fibra
    d_val = dispersion_d * 1e-6  # Converte ps/(nm*km) para s/m²
    beta2 = - (wavelength**2 * d_val) / (2.0 * np.pi * c)
    h_fiber_inv = np.exp(+ (1j / 2.0) * beta2 * length_m * (omega**2))

    # 2. Resposta em Frequência Óptica da PNN
    h_pnn = pnn_instance.frequency_response(freq_grid)

    # 3. Inversa Regularizada da PNN
    h_pnn_inv = np.conj(h_pnn) / (np.abs(h_pnn)**2 + reg_factor)

    # 4. Resposta Conjunta do Equalizador no DSP
    h_cdc_joint = h_fiber_inv * h_pnn_inv

    # 5. Filtragem por FFT e IFFT
    spec_rx = np.fft.fft(signal_rx)
    spec_equalized = spec_rx * h_cdc_joint
    return np.fft.ifft(spec_equalized)
```

---

## 5. Resultados de Validação e Benchmarks

Em simulação com sinal **10 GBaud QPSK** a **16 amostras por símbolo**, propagado por **40 km de SSMF** com uma PNN de **8 canais** ($\Delta t = 25\text{ ps}$):

| Métrica / Parâmetro | 1. Sem CDC (Sinal Bruto) | 2. CDC Tradicional (Apenas Fibra) | 3. CDC Conjunto PNN + DSP |
| :--- | :--- | :--- | :--- |
| **EVM RMS (%)** | **60,79%** | **59,97%** | **14,72%** |
| **Estado da Constelação** | Destruída / Dispersa | Dispersa (ISI da PNN mantida) | **Totalmente Aberta e Decidível** |
| **Melhoria Absoluta EVM** | — | 0,82 pp | **45,25 pp** |

---

## 6. Guia de Integração para Agentes de IA e Engenheiros

Ao aplicar essa função em uma nova simulação ou pipeline de DSP:

1. **Ordem de Execução**:
   - Gere o sinal de entrada $E_{\text{in}}(t)$.
   - Propague o campo óptico pela fibra (`simulate_fiber_dispersion`).
   - Passe o campo resultante pela PNN (`pnn.forward(..., return_optical_field=True)`).
   - Simule o batimento no receptor coerente para obter $r(t) = I + jQ$.
   - Aplique o `cd_compensation_joint_pnn(r, dt, length_km, pnn)`.
2. **Sincronização do Grid de Frequências**:
   - O argumento `freq_grid` gerado via `np.fft.fftfreq(n, d=dt)` deve utilizar exatamente a mesma taxa de amostragem ($dt = 1/f_s$) e tamanho de vetor ($N$) do sinal elétrico digitalizado.
3. **Ajuste Fino da Regularização ($\varepsilon$)**:
   - Se o nível de ruído AWGN/ASE do canal for elevado, aumente `reg_factor` para $10^{-2}$ para evitar amplificação de ruído nas frequências de atenuação da PNN.
   - Em cenários com elevado OSNR (> 25 dB), diminua `reg_factor` para $10^{-4}$ para maximizar a precisão da inversão.

---
