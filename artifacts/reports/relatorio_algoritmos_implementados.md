# Relatório Técnico dos Algoritmos Implementados: Simulador Óptico Coerente e Rede Neural Fotônica (PNN)

**Projeto:** PNN Coerente — Equalização Óptico-Digital e Redes Neurais Fotônicas  
**Data:** Novembro de 2026  
**Ambiente:** Python 3.10+ / NumPy / SciPy / Matplotlib  
**Referências Centrais:**
- Mello, D. A. A., & Barbosa, F. A. (2021). *Digital Coherent Optical Systems: Architecture and Algorithms*. Springer.
- Marciano, P. R. N. (2024). *Modelagem e Otimização de Filtros FIR e Redes Neurais Fotônicas em Silício*. Tese de Doutorado, UFES.
- Agrawal, G. P. (2012). *Nonlinear Fiber Optics*. Academic Press.
- Lundberg, L., et al. (2020). *Phase-correlated frequency comb lines*. Nature Communications.

---

**Objetivo e escopo:** documentar os modelos e algoritmos implementados no simulador Python para revisão técnica e preparação de apresentações. O documento descreve modelos numéricos e simulações; não constitui validação experimental de hardware fotônico.

**Limitações de escopo:** notebooks diferentes usam configurações distintas (por exemplo, 10 GBaud/8 amostras por símbolo nos estudos coerentes e 64 GBaud/2 amostras por símbolo na configuração de referência YAML). Leia parâmetros e conclusões junto do notebook/relatório de origem. O EDFA ideal não inclui ASE ou saturação; cenários lineares não incluem Kerr. Esta implementação não representa uma cadeia completa de sincronização, recuperação de portadora ou CFO.
## 1. Visão Geral da Arquitetura do Sistema

O simulador modela a cadeia completa de transmissão, propagação e recepção óptica coerente com modulação QPSK a 10 GBaud em 1550 nm, integrando uma **Rede Neural Fotônica (PNN)** em circuito integrado fotônico (PIC) e algoritmos de **Processamento Digital de Sinais (DSP)**.

```
+-------------------------------------------------------------------------------------------------+
|                                        TRANSMISSOR (TX)                                         |
|  Bits PRBS ---> Mapeamento Gray QPSK ---> Filtro de Pulso/DAC ---> Modulador Dual-Drive IQ-MZM  |
+-------------------------------------------------------------------------------------------------+
                                                |
                                        [Campo Óptico E(t)]
                                                v
+-------------------------------------------------------------------------------------------------+
|                                        CANAL DE FIBRA (SSMF)                                    |
|         Propagação linear com CD ou SSFM com CD + Kerr, conforme regime selecionado             |
+-------------------------------------------------------------------------------------------------+
                                                |
                                                v
+-------------------------------------------------------------------------------------------------+
|                                       AMPLIFICAÇÃO ÓPTICA                                       |
|                       EDFA (Erbium-Doped Fiber Amplifier) com Ganho Configurável                |
+-------------------------------------------------------------------------------------------------+
                                                |
                                                v
+-------------------------------------------------------------------------------------------------+
|                                REDE NEURAL FOTÔNICA (PNN) EM SILÍCIO                            |
| Filtro FIR óptico TDL, N=8 taps; o campo complexo E_sum segue ao RX coerente              |
+-------------------------------------------------------------------------------------------------+
                                                |
                                                v
+-------------------------------------------------------------------------------------------------+
|                                         RECEPTOR (RX)                                           |
| Híbrido Óptico 90° + LO (com Ruído de Fase Wiener Correlacionado rho) ---> Fotodetecção Balanceada|
+-------------------------------------------------------------------------------------------------+
                                                |
                                                v
+-------------------------------------------------------------------------------------------------+
|                                          DSP RECEPTOR                                           |
|       CDC DSP (convencional ou joint) ---> Decimação ---> Demapeamento ---> BER/EVM            |
+-------------------------------------------------------------------------------------------------+
```

---

## 2. Módulos e Algoritmos Detalhados

### 2.1. Transmissor Óptico (`src/dsp/transmitter.py`)

#### A. Mapeamento de Símbolos QPSK
- **Função:** `generate_qpsk_symbols(num_symbols, seed)`
- **Formula:** Para cada par de bits $(b_I, b_Q) \in \{0, 1\}^2$, calcula os símbolos normalizados em potência média unitária:
  $$s = \frac{1}{\sqrt{2}} \left[ (1 - 2b_I) + j(1 - 2b_Q) \right]$$
  Mapeamento implementado em quadratura: `00 -> (+1+1j)/sqrt(2)`, `01 -> (+1-1j)/sqrt(2)`, `11 -> (-1-1j)/sqrt(2)`, `10 -> (-1+1j)/sqrt(2)`.

#### B. Modulador Analítico IQ Mach-Zehnder (IQ-MZM)
- **Funções:** `dual_drive_mzm(...)`, `iq_mzm_field(...)`, `modulate_iq_mzm(...)`
- **Fundamentação:** Port analítico de alta fidelidade das equações do MATLAB (`Mach_Zehnder_Modulator_simplificado.m`).
- **Transferência do sub-MZM Dual-Drive:**
  $$E_{\text{out}}(t) = E_{\text{in}}(t) \cdot \cos\left( \frac{\pi}{2} \frac{U_1(t) - U_2(t) - U_0}{V_\pi} \right) \cdot \exp\left( -j \frac{\pi}{2} \frac{U_1(t) + U_2(t)}{V_\pi} \right)$$
  - $U_1, U_2$: Tensões aplicadas nos eletrodos dos braços superiores e inferiores.
  - $U_0 = V_\pi$: Tensão de polarização nula (bias de transmissão nula com portadora suprimida).
  - **Operação Push-Pull:** Ao aplicar $U_1(t) = -U_2(t)$, o termo de *chirp* $\exp(-j \frac{\pi}{2} \frac{U_1+U_2}{V_\pi}) = 1$, eliminando completamente o chirp residual do transmissor.
- **Estrutura em Quadratura:**
  $$E_{\text{TX}}(t) = 0.5 \cdot \left[ E_I(t) + E_Q(t) \cdot e^{j \phi_{\text{quad}}} \right]$$
  onde $\phi_{\text{quad}} = -\pi/2$ (ou $+\pi/2$) insere a defasagem estática necessária para sintetizar a constelação no plano complexo.

#### C. Filtragem Elétrica de Drive (DAC/Driver)
- **Classe/Funções:** `ElectricalFilterParams`, `aplicar_filtro` (`src/utils/filtros.py`)
- **Implementação:** Filtro passa-baixas em banda base aplicado aos sinais analógicos de tensão $U_I(t)$ e $U_Q(t)$ antes da modulação eletro-óptica.
- **Tipos disponíveis:** Gaussiano de ordem $n$, Cosseno Levantado com roll-off $\beta$, e Butterworth de ordem $m$. Modela a largura de banda analógica limitada dos conversores digital-analógicos e amplificadores de RF.

---

### 2.2. Canal de Propagação em Fibra Monomodo (`src/dsp/fiber.py`)

#### A. Equação Não-Linear de Schrödinger (NLSE)
$$\frac{\partial A}{\partial z} + \frac{\alpha}{2} A + j \frac{\beta_2}{2} \frac{\partial^2 A}{\partial T^2} - \frac{\beta_3}{6} \frac{\partial^3 A}{\partial T^3} = j \gamma |A|^2 A$$

#### B. Algoritmo Split-Step Fourier Method (SSFM)
- **Função:** `fibra_monomodo(t, u, wavelength, t0, length_km, regime=1, params=FiberParams, num_steps=100)`
- **Mecanismo:** Divide o enlace de comprimento $L$ em passos $\Delta z = L / \text{num\_steps}$. Em cada passo, repete a sequência simétrica abaixo:
  1. **Meio passo linear no domínio da frequência:** Aplica atenuação e dispersão cromática:
     $$\hat{D} = -\frac{\alpha}{2} - j \frac{\beta_2}{2} (2\pi f)^2 + j \frac{\beta_3}{6} (2\pi f)^3$$
     $$A(z + \Delta z/2, t) = \mathcal{F}^{-1} \left\{ \mathcal{F}\{A(z, t)\} \cdot \exp\left( \hat{D} \frac{\Delta z}{2} \right) \right\}$$
  2. **Passo Não-Linear no domínio do tempo:** Aplica auto-modulação de fase (SPM via efeito Kerr):
     $$\hat{N} = j \gamma |A|^2$$
     $$A(z + \Delta z/2, t) = A(z + \Delta z/2, t) \cdot \exp\left( j \gamma |A|^2 \Delta z \right)$$
  3. **Segundo meio passo linear ($\\Delta z / 2$):** Completa o passo simétrico. A sequência inteira é repetida `num_steps` vezes; sob as hipóteses usuais, o splitting simétrico tem erro global de segunda ordem.
- **Regime Linear Analítico (`regime=0`):** Aplica diretamente a função de transferência contínua em um único passo FFT:
  $$H_{\text{fibra}}(f) = \exp\left( -\frac{\alpha}{2} L \right) \exp\left( +j \frac{\beta_2}{2} L (2\pi f)^2 \right)$$
  com $\beta_2 = -\frac{\lambda^2 D}{2\pi c}$. O CDC aplica a fase inversa, com sinal $-j$.

---

### 2.3. Modelos de Ruído Estocástico e Laser (`src/dsp/noise_models.py`)

#### A. Ruído de Fase de Wiener (Passeio Aleatório)
- **Função:** `generate_wiener_phase_noise(num_samples, dt, linewidth_hz, seed)`
- **Física:** Modela o ruído Lorentziano de emissão espontânea na cavidade do laser semicondutor.
- **Variância por amostra temporal:**
  $$\sigma_{\Delta\phi}^2 = 2 \pi \Delta\nu \Delta t$$
  $$\phi[n] = \sum_{k=0}^{n} \Delta\phi[k], \quad \Delta\phi[k] \sim \mathcal{N}(0, \sigma_{\Delta\phi}^2)$$

#### B. Mistura de Ruído de Fase Correlacionado ($\rho$)
- **Função:** `generate_correlated_phases(num_samples, dt, linewidth_hz, correlation_rho, seed)`
- **Mecanismo:** Modela sistemas com acoplamento de fase ajustável entre TX e LO:
  $$\phi_{\text{TX}}(t) = \phi_{\text{common}}(t)$$
  $$\phi_{\text{LO}}(t) = \rho \cdot \phi_{\text{common}}(t) + \sqrt{1 - \rho^2} \cdot \phi_{\text{indep}}(t)$$
- **Comportamento Limite:**
  - $\rho = 0.0$: os processos de fase TX e LO são independentes no modelo; para larguras de linha iguais, a variância residual por amostra é $4\pi\Delta\nu\Delta t$.
  - $\rho = 1.0$: as duas fases são idênticas no modelo e o ruído de fase residual cancela exatamente. Para qualquer $\rho<1$ permanece fase residual; isso não demonstra que CPE/BPS seja dispensável em um sistema real.

---

### 2.4. Amplificador Óptico EDFA (`src/dsp/edfa.py`)

- **Classe:** `EDFAIdeal(gain_db)`
- **Operação:** Amplificação óptica de ganho plano:
  $$E_{\text{amp}}(t) = E_{\text{in}}(t) \cdot 10^{\frac{G_{\text{dB}}}{20}}$$
  Atua compensando perdas de campo conforme o ganho configurado. O modelo é ideal: não inclui ASE, saturação ou dependência espectral; nas simulações, o ganho por span repõe a atenuação média da fibra.

---

### 2.5. Rede Neural Fotônica em Silício (`src/models/pnn.py`)

- **Classe:** `PhotonicNeuralNetwork(num_channels, base_delay_ps, amplitude_weights, phase_weights, normalize_power, responsivity)`
- **Arquitetura Física:** Linha de Atraso Temporal (*Tapped-Delay Line - TDL*) integrada em guia de onda de silício (PIC).
- **Parâmetros Base:**
  - $N = 8$ vias paralelas (canais/taps).
  - $\Delta t = 25\text{ ps}$ (atraso incremental base), resultando em $\text{FSR} = 40\text{ GHz}$.
  - Em $10\text{ GBaud}$ ($T_s = 100\text{ ps}$), o filtro atua como um **FSE fracionário $T/4$**.
- **Equação de Campo Recombinado:**
  $$E_{\text{sum}}(t) = \frac{1}{N} \sum_{i=0}^{N-1} a_i e^{j\phi_i} E_{\text{in}}(t - i\Delta t)$$
  - $a_i \in [0, 1]$: Pesos de atenuação/transmitância (modulados por MZI termo-ópticos ou eletro-ópticos).
  - $\phi_i \in [0, 2\pi)$: Deslocamentos de fase ópticos.
- **Resposta em Frequência Analítica:**
  $$H_{\text{PNN}}(f) = \frac{1}{N} \sum_{i=0}^{N-1} a_i e^{j\phi_i} \exp\left( -j 2\pi f i \Delta t \right)$$
- **Detecção Quadrática (Não Linearidade Fotônica):**
  $$I_{\text{out}}(t) = R \cdot |E_{\text{sum}}(t)|^2$$
- **Separação de interfaces:** `forward(..., return_optical_field=True)` retorna a corrente real calculada por $R|E_{sum}|^2$ e também o campo complexo pré-fotodetecção $E_{sum}$. No caminho coerente dos notebooks, o receptor usa o campo complexo; a corrente quadrática é uma saída alternativa de detecção direta, não a entrada do híbrido coerente.

---

### 2.6. Equalização Conjunta Óptico-Digital (`src/dsp/joint_pnn_cdc_equalizer.py`)

- **Função:** `cd_compensation_joint_pnn(signal_rx, dt, length_km, pnn_instance, dispersion_d, reg_factor=1e-3)`
- **Motivação:** Quando a PNN opera no domínio óptico antes do receptor, a resposta linear total sofrida pelo sinal é a cascata $H_{\text{total}}(f) = H_{\text{fibra}}(f) \cdot H_{\text{PNN}}(f)$. Aplicar apenas o CDC convencional da fibra deixa a distorção introduzida pelos pesos da PNN sem correção.
- **Filtro Inverso Regularizado no DSP:**
  $$H_{\text{joint}}(f) = H_{\text{fibra}}^{-1}(f) \cdot \frac{H_{\text{PNN}}^*(f)}{|H_{\text{PNN}}(f)|^2 + \varepsilon}$$
  - A implementação canônica usa $H_{\text{fibra}}(f)=\exp(+j\beta_2L(2\pi f)^2/2)$, compatível com `src/dsp/fiber.py`; portanto, $H_{\text{fibra}}^{-1}(f)=\exp(-j\beta_2L(2\pi f)^2/2)$.
  - $\varepsilon$ regulariza a inversa da PNN. O valor absoluto padrão da função é $10^{-3}$; no notebook BER por distância, o valor usado é relativo: $10^{-3}$ vezes a média de $|H_{PNN}|^2$. A regularização limita o ganho da inversa, mas não garante recuperação perfeita, especialmente em nulos espectrais.
  - Essa expressão é para um canal linear em campo. Não demonstra inversão de Kerr ou de detecção quadrática. A simulação específica de dispersão deste módulo não inclui atenuação; a cadeia de enlace principal está em `src/dsp/fiber.py`.

---

### 2.7. Otimização dos Pesos da PNN via PSO (`scripts/optimize_pnn_pso.py`)

- **Algoritmo:** *Particle Swarm Optimization (PSO)* com espaço contínuo limitado.
- **Vetor de Estados (Dimensão $2N = 16$):**
  $$\mathbf{x} = [a_0, a_1, \dots, a_{N-1}, \, \phi_0, \phi_1, \dots, \phi_{N-1}]$$
- **Condições de Contorno:**
  - Amplitudes: $a_i \in [0.0, 1.0]$ com saturação rígida (*clamping*).
  - Fases: os notebooks atuais usam limites $[-\pi,\pi]$ e marcam essas coordenadas como periódicas no PSO. O otimizador aceita qualquer intervalo periódico; o envolvimento evita que fases equivalentes em lados opostos do intervalo sejam tratadas como distantes. O PSO não usa gradientes.
- **Funções objetivo:** o PSO é genérico e minimiza o escalar retornado por uma função fornecida pelo chamador. Os notebooks podem retornar BER ou EVM; no notebook `evm_pnn_cdc_treinamento_conjunto.ipynb`, os dois objetivos são comparados em treinamentos separados.

---

### 2.8. Receptor Coerente e Métricas (`src/dsp/receiver.py` & `src/utils/metrics.py`)

- **Detecção coerente (`coherent_detection`):** forma o sinal complexo em banda base $r_{bb}=R E_s E_{LO}^*$. Sua parte real representa I e sua parte imaginária representa Q. O código usa uma representação equivalente idealizada e não calcula explicitamente as quatro correntes dos fotodiodos do híbrido. A escala implementada não inclui um fator 2 explícito; correntes absolutas exigem declarar a convenção de escala do híbrido/balanceamento.
- **Compensação digital de dispersão (`cd_compensation`):** aplica $H_{CDC}(f)=\exp(-j\beta_2L(2\pi f)^2/2)$, inversa do filtro de propagação linear de `src/dsp/fiber.py`.
- **Demapeamento (`demap_qpsk`):** decisão por quadrante no plano complexo I/Q e conversão para bits conforme o mapeamento gerado pelo transmissor.
- **Métricas:** BER compara bits; EVM RMS normaliza potência e remove uma rotação de fase constante ótima. Não faz recuperação de fase variável nem busca de temporização.

---

## 3. Resumo das Capacidades de Simulação e Notebooks

| Caderno | Descrição do Fluxo | Foco de Pesquisa |
|---|---|---|
| `simulacao_coerente_qpsk.ipynb` | Modulação ideal, canal SSMF, detecção coerente B2B e CDC | Linha de base teórica tradicional |
| `simulacao_coerente_qpsk_iq_mzm.ipynb` | Modulador IQ-MZM analítico, varredura de distância e ruído de fase | Efeito de Vpi, chirp nulo e linewidth |
| `teste_modulador_iq_mzm.ipynb` | Curvas estáticas de transferência óptica vs. tensão de drive | Caracterização analítica do MZM |
| `teste_modelo_fibra_scm_fechamentoCW.ipynb` | Validação do algoritmo SSFM contra o código original MATLAB | Fechamento de olho e dispersão |
| `ook_mzm_pushpull_ber_distancia.ipynb` | Modulação OOK push-pull com detecção direta PIN e PNN | Mitigação de dispersão em enlaces curtos |
| `exploracao_pnn_dispersao_edfa.ipynb` | PNN com atenuação de fibra e compensação de ganho por EDFA | Impacto da cascata óptica com PNN |
| `exploracao_pnn_dispersao_edfa_correlacao_fase.ipynb` | PNN com coeficiente TX/LO configurável (valor atual $\rho=0.99$) | Efeito da correlação e da largura de linha |
| `avaliacao_equalizacao_conjunta_pnn_cdc.ipynb` | Comparativo CDC tradicional vs. Equalizador Conjunto | Eliminação da ISI induzida pela PNN |
| `ber_distancia_pnn_cdc_joint_cdc.ipynb` | BER vs. distância (50 a 1000 km) para sete estratégias e três larguras de linha | Comparação CDC tradicional e joint CDC; ver `artifacts/reports/relatorio_ber_distancia_pnn_cdc_joint_cdc.md` |
| `evm_pnn_cdc_treinamento_conjunto.ipynb` | Compara treinamento por BER e por EVM; curvas de EVM e constelações | Síntese dos pesos $a_i$ e $\phi_i$ |

---

### 3.1 Reprodutibilidade e leitura dos resultados

Para cada figura ou conclusão, registre notebook de origem, configuração efetiva, sementes, tamanho dos quadros de treino/teste e número de erros. `config/system_params.yaml` documenta uma configuração de referência, mas não substitui parâmetros locais dos notebooks. O relatório `artifacts/reports/relatorio_ber_distancia_pnn_cdc_joint_cdc.md` alerta que a leitura da figura é qualitativa: o piso de plotagem pode ocultar BER zero versus BER não nula, e cada ponto pode usar pesos PNN treinados separadamente. Não generalize resultados de simulação para desempenho de dispositivo físico.

Os trechos comentados de código estão em `artifacts/reports/algoritmos_codigo_fonte.md`. Para a implementação integral e atualizada, consulte os arquivos `.py` indicados nesse documento.

## 4. Consistência de sinais e convenções

A propagação linear canônica (`src/dsp/fiber.py`) usa $H_{fibra}=\exp(+j\beta_2L\omega^2/2)$, incluindo atenuação. O CDC tradicional usa a fase inversa. O joint CDC e a função de propagação auxiliar foram alinhados a essa convenção. O notebook `ber_distancia_pnn_cdc_joint_cdc.ipynb` contém uma verificação controlada da cascata linear antes da comparação por PSO.

Na recepção coerente, o código representa as duas correntes reais I/Q como um único vetor complexo. Na PNN, a saída de fotocorrente $R|E|^2$ é real; quando a cadeia segue para recepção coerente, o sinal usado é o campo complexo pré-detecção retornado por `return_optical_field=True`.

---

## 5. Orientações para Geração de Apresentações e Análises Executivas

Ao sintetizar este material para apresentações e artigos, recomenda-se enfatizar três conclusões principais:
1. **Correlação de fase:** no modelo implementado, o componente comum cancela exatamente no batimento quando $\rho=1$. Para $\rho<1$ resta ruído de fase; o modelo não representa offset de frequência nem prova, por si só, que CPE/BPS seja dispensável em um sistema real.
2. **Equalização conjunta:** a inversão da cascata linear PNN-fibra é uma hipótese a avaliar. A inversa regularizada não garante recuperação plena em nulos da PNN e o desempenho depende da consistência de sinais, convenções, ruído e contornos FFT. Os resultados precisam ser validados por uma cadeia sintética controlada.
3. **Espaçamento fracionário:** com $T_s=100$ ps, o passo de atraso de 25 ps corresponde a $T_s/4$. Isso descreve o espaçamento entre taps; não estabelece equivalência completa de desempenho, consumo ou implementação com um FSE eletrônico.

4. **Evidência dos resultados:** separe fatos implementados, resultados observados e hipóteses. Relatórios devem registrar resolução estatística, sementes e limitações; BER estimada em quadros finitos não distingue automaticamente zero erros de uma taxa muito pequena.


