# Algoritmos implementados — trechos de código comentados

Este documento acompanha [`relatorio_algoritmos_implementados.md`](relatorio_algoritmos_implementados.md) e reúne os trechos centrais usados para explicar a implementação. Os blocos são excertos representativos, não cópias integrais dos módulos. Para código completo, atualizado e executável, use os caminhos-fonte indicados em cada seção. Essa separação evita que uma cópia em Markdown se torne desatualizada.

## 1. Propagação linear e SSFM

**Fonte:** [`src/dsp/fiber.py`](../../src/dsp/fiber.py), função `fibra_monomodo`.

A implementação usa frequência em Hz convertida para frequência angular e, no regime não linear, repete meio passo linear, passo Kerr e meio passo linear em cada subdivisão `dz`:

```python
freq_grid = np.fft.fftfreq(n, d=dt)
omega = 2 * np.pi * freq_grid
dz = length_m / num_steps
h2 = dz / 2.0
exp_h2_dw = np.exp(h2 * dw)

az = u.copy()
for _ in range(num_steps):
    # Meio passo linear: atenuação e dispersão no domínio da frequência.
    az_w = np.fft.fft(az) * exp_h2_dw
    az_mid = np.fft.ifft(az_w)

    # Passo não linear Kerr/SPM no domínio do tempo.
    phase_nl = gamma * dz * (np.abs(az_mid) ** 2)
    az_nl = az_mid * np.exp(1j * phase_nl)

    # Segundo meio passo linear completa o passo simétrico.
    az_w_final = np.fft.fft(az_nl) * exp_h2_dw
    az = np.fft.ifft(az_w_final)
```

O regime linear (`regime=0`) aplica diretamente a resposta em frequência, incluindo atenuação. O SSFM acima usa o termo de dispersão de segunda e terceira ordem definido no próprio módulo. Os arrays FFT representam blocos periódicos; efeitos de contorno devem ser considerados ao interpretar resultados.

## 2. Modelo de campo da PNN

**Fonte:** [`src/models/pnn.py`](../../src/models/pnn.py), classe `PhotonicNeuralNetwork`, métodos `complex_weights`, `frequency_response` e `forward`.

Pesos complexos normalizados e resposta em frequência:

```python
w = self.amplitude_weights * np.exp(1j * self.phase_weights)
if self.normalize_power:
    w = w / self.num_channels

H = np.zeros(len(frequencies_hz), dtype=complex)
for i in range(self.num_channels):
    tau_i = i * self.base_delay_s
    H += w[i] * np.exp(-1j * 2.0 * np.pi * frequencies_hz * tau_i)
```

No processamento temporal, atrasos que correspondem a número inteiro de amostras são implementados com preenchimento de zeros à esquerda. Atrasos fracionários usam deslocamento de fase no domínio FFT. O fim do processamento e a saída alternativa de detecção direta são:

```python
E_sum += w[i] * E_delayed
photocurrent = self.responsivity * np.real(E_sum * np.conj(E_sum))

if return_optical_field:
    return photocurrent, E_sum
return photocurrent
```

Na cadeia coerente, o notebook passa adiante `E_sum`, o campo complexo antes da detecção quadrática. A fotocorrente real `R|E_sum|²` serve para a interface de detecção direta; ela não substitui o campo de entrada do híbrido coerente.

## 3. Ruído de fase de Wiener e correlação TX/LO

**Fonte:** [`src/dsp/noise_models.py`](../../src/dsp/noise_models.py), `generate_wiener_phase_noise` e `generate_correlated_phases`.

```python
var_delta_phi = 2.0 * np.pi * linewidth_hz * dt
sigma_delta_phi = np.sqrt(var_delta_phi)
delta_phi = rng.normal(loc=0.0, scale=sigma_delta_phi, size=num_samples)
delta_phi[0] = 0.0
phi = np.cumsum(delta_phi)
```

A mistura de fase correlacionada gera um processo comum e um processo independente, ambos com incrementos de Wiener:

```python
phi_common = generate_wiener_phase_noise(
    num_samples, dt, linewidth_hz, seed=seed1
)
phi_indep = generate_wiener_phase_noise(
    num_samples, dt, linewidth_hz, seed=seed2
)

phi_tx = phi_common
rho_clamped = np.clip(correlation_rho, 0.0, 1.0)
phi_lo = (
    rho_clamped * phi_common
    + np.sqrt(1.0 - rho_clamped**2) * phi_indep
)
```

Aqui `rho` controla a combinação dos dois caminhos no modelo. Para `rho=1`, TX e LO compartilham a mesma trajetória de fase; para `rho<1`, permanece ruído residual. Não há neste modelo offset de frequência entre lasers.

## 4. Equalizador conjunto PNN + CDC

**Fonte canônica:** [`src/dsp/joint_pnn_cdc_equalizer.py`](../../src/dsp/joint_pnn_cdc_equalizer.py), função `cd_compensation_joint_pnn`.

O equalizador combina a inversa da dispersão da fibra com uma inversa regularizada da resposta de campo da PNN:

```python
freq_grid = np.fft.fftfreq(len(signal_rx), d=dt)
omega = 2.0 * np.pi * freq_grid
beta2 = -(wavelength**2 * (dispersion_d * 1e-6)) / (2 * np.pi * c)

h_fiber_inv = np.exp(-1j * beta2 * length_m * omega**2 / 2.0)
h_pnn = pnn_instance.frequency_response(freq_grid)
h_pnn_inv = np.conj(h_pnn) / (np.abs(h_pnn)**2 + reg_factor)
h_cdc_joint = h_fiber_inv * h_pnn_inv
return np.fft.ifft(np.fft.fft(signal_rx) * h_cdc_joint)
```

A regularização reduz o ganho em vales espectrais, mas não recupera informação eliminada por nulos. Este equalizador pressupõe um canal linear em campo; não inverte Kerr nem detecção quadrática. `reg_factor` é absoluto na função. O notebook pode convertê-lo em valor relativo à potência média de `|H_PNN|²`.

## 5. PNN integrada ao CDC no treinamento PSO

**Fonte do fluxo de treinamento:** [`notebooks/ber_distancia_pnn_cdc_joint_cdc.ipynb`](../../notebooks/ber_distancia_pnn_cdc_joint_cdc.ipynb), função `optimize_pnn`. **Otimizador:** [`scripts/optimize_pnn_pso.py`](../../scripts/optimize_pnn_pso.py), função `particle_swarm_optimize`.

Cada posição candidata contém `N` amplitudes e `N` fases. Para avaliar a posição, o notebook atualiza a PNN, processa o campo de treino, aplica detecção coerente e escolhe a compensação correspondente ao modo de objetivo:

```python
model.set_weights(
    position[:PNN_NUM_CHANNELS], position[PNN_NUM_CHANNELS:]
)
_, field_pnn = model.forward(
    field_train, dt_s=DT, return_optical_field=True
)
bb = coherent_detection(field_pnn, phase_lo=phase_lo_train)

if objective_mode == "cdc":
    bb = cd_compensation(
        bb, DT, distance_km, dispersion_d=FIBER_PARAMS.d_1550
    )
elif objective_mode == "joint_cdc":
    h = model.frequency_response(np.fft.fftfreq(len(bb), d=DT))
    eps = JOINT_CDC_REG_REL * np.mean(np.abs(h) ** 2)
    bb = cd_compensation_joint_pnn(
        bb, DT, distance_km, model,
        dispersion_d=FIBER_PARAMS.d_1550,
        wavelength=WAVELENGTH, reg_factor=eps
    )

bits_rx = demap_qpsk(sample_to_symbols(bb, sps=SPS))
n = min(len(train_bits), len(bits_rx))
return calculate_ber(train_bits[:n], bits_rx[:n])
```

O PSO avalia essa função objetivo e trata as coordenadas de fase como periódicas no intervalo configurado:

```python
result = particle_swarm_optimize(
    objective,
    bounds,
    swarm_size=PSO_SWARM_SIZE,
    max_iterations=PSO_ITERATIONS,
    periodic_dimensions=range(PNN_NUM_CHANNELS, 2 * PNN_NUM_CHANNELS),
    seed=seed,
)
```

Assim, no modo `joint_cdc`, cada candidato à PNN também altera `H_PNN` e, portanto, o equalizador calculado dentro da própria avaliação. O trecho de treinamento é específico desse notebook; outros notebooks podem usar outra função objetivo, tamanho de quadro, métrica, limites ou estratégia de regularização.

## 6. Referências de leitura

- Visão geral, equações e limitações: [`relatorio_algoritmos_implementados.md`](relatorio_algoritmos_implementados.md).
- Resultados e ressalvas do estudo BER por distância: [`relatorio_ber_distancia_pnn_cdc_joint_cdc.md`](relatorio_ber_distancia_pnn_cdc_joint_cdc.md).
- Configuração de referência do projeto: [`config/system_params.yaml`](../../config/system_params.yaml) e [`config/pnn_config.yaml`](../../config/pnn_config.yaml). Os notebooks podem sobrescrever estes valores; consulte sempre a configuração executada.
