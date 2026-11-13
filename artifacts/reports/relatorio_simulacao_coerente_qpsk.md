# Coherent QPSK Optical System Simulation Report at 10 GBaud (1550 nm)

**Execution date:** Outubro de 2026  
**Environment:** Python / Jupyter / PyCharm  
**Fiber model:** Split-Step Fourier Method (SSFM) e Modelo Linear baseados em `Fibra_Monomodo.m`.

---

## 1. General Simulation Parameters
- **Symbol rate:** 10 GBaud ($T_s = 100$ ps)
- **Modulation:** QPSK (Mapeamento Gray, 2 bits/símbolo)
- **Central wavelength:** $\lambda = 1550$ nm
- **Number of symbols:** $N_{sym} = 16384 = 2^{14}$ (potência de 2)
- **Samples per symbol (SpS):** $8 = 2^3$ (potência de 2)
- **SSMF parameters:**
  - Attenuation $\alpha = 0.2$ dB/km
  - Dispersion $D_{1550} = 16.0$ ps/(nm$\cdot$km)
  - Slope $S = 0.09$ ps/(nm$^2\cdot$km)
  - Nonlinearity $n_2 = 3.2 \times 10^{-20}$ m$^2$/W
  - Effective area $A_{eff} = 70\,\mu$m$^2$

---

## 2. Analysis Results

### Analysis 1: BER vs. Distance (100 km to 1200 km)
- **Phase condition:** Fase ideal entre portadora e oscilador local (ruído de fase nulo, $\Delta \phi = 0$).
- **Scenarios:** Sem compensação de dispersão cromática (Sem CDC) vs. Com compensação (Com CDC) para potências de lançamento de $-10$ dBm, $0$ dBm e $+10$ dBm.
- **Generated figure:** `artifacts/figures/analise_1_ber_vs_distancia.png`

**Findings:**
1. **Without CDC:** O espalhamento temporal induzido pela dispersão cromática ($\Delta T \approx 128$ ps em 100 km até $1536$ ps em 1200 km) fecha completamente o diagrama de olho já a partir de $100$ km. A BER se fixa rapidamente em torno de $0.15$ a $0.35$, tornando a transmissão inviável sem equalização.
2. **With CDC:** A aplicação do filtro inverso exato $H_{CDC}(\omega) = \exp(-j \frac{\beta_2}{2} L \omega^2)$ restaura perfeitamente a integridade do sinal em potências lineares ($-10$ dBm e $0$ dBm), mantendo a BER no piso ótimo estabelecido pelo ruído térmico/AWGN.
3. **Nonlinear effect (+10 dBm):** Sob alta potência ($+10$ dBm), o efeito da automodulação de fase (SPM) passa a interagir com a dispersão cromática durante o Split-Step Fourier, causando uma penalidade gradual na BER para distâncias superiores a $800$ km, demonstrando os limites do equalizador linear puramente cromático.

---

### Analysis 2: BER vs. Independent Phase Noise at 100 km
- **Condition:** Distância fixa em 100 km com CDC estático, lasers independentes ($\rho = 0$, sistema intrágeno tradicional), sem bloco de recuperação de fase no DSP.
- **Sweep:** Largura de linha Lorentziana dos lasers variando de $0$ kHz a $10$ MHz.
- **Generated figure:** `artifacts/figures/analise_2_ber_vs_phase_noise.png`

**Findings:**
- Para larguras de linha ultra-estreitas ($< 50$ kHz), o passeio aleatório de Wiener permanece contido dentro dos limites de decisão de $\pm \pi/4$ do QPSK durante curtos intervalos.
- Conforme a largura de linha aumenta para a faixa de centenas de kHz e MHz (comum em lasers semicondutores DFB), a variância do incremento de fase por símbolo $\sigma_{d\phi}^2 = 2\pi \Delta\nu T_s$ acumula rotações contínuas da constelação.
- Sem um bloco de Carrier Phase Estimation (CPE, como o BPS), a BER colapsa rapidamente para o limite assintótico de decisão puramente aleatória ($\text{BER} \to 0.5$).

---

### Analysis 3: BER vs. Linewidth with Highly Correlated Phase at 100 km
- **Condition:** Distância fixa em 100 km, comparando três regimes de correlação de fase entre portadora e LO:
  - $\rho = 0.0$: Lasers independentes (Receptor intrágeno tradicional).
  - $\rho = 0.99$: Correlação moderada/alta.
  - $\rho = 0.9999$: Alta correlação característica de fontes com pente de frequência óptico (OFC) ou receptores auto-homódinos (*Coherent-Lite* de Coelho & Segatto).
- **Generated figure:** `artifacts/figures/analise_3_ber_vs_correlacao_linewidth.png`

**Findings:**
- Enquanto o sistema tradicional com lasers independentes ($\rho = 0$) degrada catastroficamente com o aumento da largura de linha, o sistema com alta correlação de fase ($\rho = 0.9999$) **mantém a BER rigorosamente estável e nula/mínima** em toda a faixa de $10$ kHz até $10$ MHz.
- Isso comprova experimentalmente no modelo a dedução teórica do NotebookLM: no batimento óptico balanceado, a fase do laser semente $\phi_{seed}(t)$ entra com sinais opostos no campo do sinal e no LO ($E_s \cdot E_{LO}^*$), promovendo o **cancelamento óptico em modo comum**.
- Dessa forma, sistemas auto-homódinos ou baseados em pentes de frequência eliminam a necessidade de algoritmos complexos de recuperação de fase no DSP, permitindo o uso de lasers semente com linewidths elevados sem qualquer penalidade de BER.
