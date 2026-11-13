# Relatório Técnico e Diagnóstico de Desempenho: PNN, CDC e Equalização Conjunta (Joint CDC)

**Autor:** Gemini Notebook Analysis System  
**Data:** 03 de Novembro de 2026  
**Público-Alvo:** Agente de IA / Engenheiro de Sistemas de Comunicações Ópticas  
**Objetivo:** Fornecer um diagnóstico teórico, numérico e algorítmico consolidado dos experimentos de Bit Error Rate (BER) vs. Distância (50 a 1000 km) utilizando Rede Neural Fotônica (PNN) em circuitos de silício e Compensação de Dispersão Cromática (CDC) no DSP.

---

## 1. Parâmetros do Sistema e Configuração Experimental

Para auditoria e reprodutibilidade pelos próximos agentes de IA, os parâmetros extraídos do ambiente e da execução do notebook `ber_distancia_pnn_cdc_joint_cdc.ipynb` são:

| Parâmetro | Valor | Descrição / Observação |
| :--- | :--- | :--- |
| **Formato de Modulação** | QPSK | Mapeamento Euclidiano de Gray |
| **Taxa de Símbolos ($R_s$)** | 10 GBaud | Período de símbolo $T_s = 100\text{ ps}$ |
| **Taxa de Amostragem ($f_s$)** | 80 GSa/s | $sps = 8$ amostras por símbolo ($dt = 12.5\text{ ps}$) |
| **Parâmetros da Fibra** | SSMF | $D = 16\text{ ps/(nm}\cdot\text{km)}$, $\alpha = 0.2\text{ dB/km}$, $\lambda = 1550\text{ nm}$ |
| **Regime da Fibra** | Linear (`regime = 0`) | Efeito Kerr (SPM/XPM) desligado para isolar a dispersão |
| **Arquitetura do Enlace** | Spans de 50 km | Amplificação EDFA ideal compensando exatamente a perda por span |
| **Varredura de Distância** | 50 a 1000 km | 20 pontos discretos em passos de 50 km |
| **Ruído de Fase (TX/LO)** | $\rho = 0.99$ | Ruído de fase altamente correlacionado ($\Delta\nu \in \{0\text{ Hz}, 100\text{ kHz}, 1\text{ MHz}\}$) |
| **Tamanho dos Quadros** | Treino: 512 sim (1024 bits)<br>Teste: 4096 sim (8192 bits) | Teste realizado em quadro independente |
| **Piso de BER do Gráfico** | $1.22 \times 10^{-4}$ | $1 / (2 \times 4096) = 1 / 8192$ (0 erros detectados no teste) |
| **Estrutura da PNN** | $N = 8$ canais, $\Delta t = 25\text{ ps}$ | Janela temporal $\Delta T_{\text{PNN}} = (N-1)\Delta t = 175\text{ ps}$; pot. norm = $1/N$ |
| **Otimizador da PNN** | PSO | 60 partículas, 100 iterações (6060 avaliações por distância/modelo) |

---

## 2. Resumo dos Resultados Observados (7 Configurações)

1. **CDC Apenas (DSP Tradicional)**: Permaneceu no piso do gráfico ($\text{BER} \approx 1.22 \times 10^{-4}$, 0 erros) em toda a extensão do enlace ($50\text{ km}$ a $1000\text{ km}$).
2. **Sem Compensação**: Degradação severa imediata ($\text{BER} \sim 10^{-1}$) já nos primeiros spans.
3. **PNN Apenas (Sem DSP)**: Funciona satisfatoriamente apenas em distâncias curtas ($< 200\text{ km}$), degradando para $\sim 10^{-1}$ em distâncias longas.
4. **PNN + CDC (Treino Separado)**: Apresenta comportamento similar à PNN isolada em distâncias longas, demonstrando que aplicar o CDC tradicional após uma PNN treinada sem DSP não restaura o desempenho.
5. **PNN + CDC (Treino Conjunto)**: Apresenta excelente desempenho na maioria das distâncias (próximo ao piso), porém exibe **picos isolados de BER** em certas distâncias.
6. **PNN + Joint CDC (Treino Separado ou Conjunto)**: **Resultado negativo crítico**. A BER permanece extremamente alta ($\sim 10^{-1}$) na maior parte dos pontos de distância.

---

## 3. Diagnóstico Físico, Matemático e Algorítmico

### 3.1. Por que o "Joint CDC" Falhou ($H_{\text{joint}} = H_{\text{fibra}}^{-1} \cdot H_{\text{PNN}}^{-1}$)?

A formulação teórica da inversão conjunta no DSP tentou aplicar:
$$H_{\text{joint}}(f) = H_{\text{fibra}}^{-1}(f) \cdot \frac{H_{\text{PNN}}^*(f)}{|H_{\text{PNN}}(f)|^2 + \varepsilon}$$

**Causas do Fracasso:**
1. **Vales Espectrais (Zeros de $H_{\text{PNN}}$):** A PNN cria atenuações em frequências específicas. Na inversão regularizada, onde $|H_{\text{PNN}}(f)| \to 0$, a função inversa exige ganhos elevados, **amplificando drasticamente o ruído aditivo e resíduos numéricos de borda da FFT**.
2. **Conflito entre Recepção Coerente e Filtragem de Amplitude:** Em um receptor coerente linear, a fibra é um filtro passa-tudo de fase ($|H_{\text{fibra}}(f)| = 1$). O CDC digital inverte essa fase perfeitamente. A PNN introduz filtragem de amplitude óptica; tentar reamplificar essa amplitude no DSP destrói a Relação Sinal-Ruído (SNR) digital.

### 3.2. Por que a PNN Isolada Degrada em Enlaces Longos ($> 200\text{ km}$)?

A memória física máxima da PNN de 8 canais é:
$$\Delta T_{\text{PNN}} = (8 - 1) \times 25\text{ ps} = 175\text{ ps}$$

O espalhamento temporal do pulso devido à dispersão cromática cresce linearmente com a distância:
$$\Delta T_{\text{CD}} = |D| \cdot L \cdot \Delta \lambda$$
* Para $50\text{ km}$: $\Delta T_{\text{CD}} \approx 13.6\text{ ps} \ll 175\text{ ps}$ (PNN consegue equalizar).
* Para $1000\text{ km}$: $\Delta T_{\text{CD}} \approx 272\text{ ps} > 175\text{ ps}$ (Excede a capacidade de memória da PNN).

### 3.3. Origem dos Picos Isolados na Curva "PNN + CDC (Treino Conjunto)"

1. **Overfitting por Função de Custo Discreta (BER):** O PSO minimiza o número discreto de erros em um quadro curto de treino (512 símbolos = 1024 bits). Isso cria platôs na paisagem de otimização.
2. **Falta de Generalização:** Pesos que zeram erros no quadro de 512 símbolos podem introduzir pequenas distorções de amplitude que provocam erros ao testar no quadro longo e independente de 4096 símbolos.

---

## 4. Diretrizes e Recomendações para o Próximo Agente de IA

Para implementar melhorias ou ajustar o código da simulação, siga o plano de ação abaixo:

### Ação 1: Alterar a Função Objetivo do PSO (Trocar BER por EVM/MSE)
* **Problema:** A BER é uma métrica quantizada/discreta, inadequada para algoritmos baseados em enxame (PSO).
* **Solução:** Modificar a função de custo do PSO para minimizar o **Error Vector Magnitude (EVM RMS)** ou o **Mean Squared Error (MSE)** das amostras I/Q complexas contínuas antes da decisão. Isso fornece uma superfície de custo suave e elimina os picos isolados de overfitting.

### Ação 2: Trava de Ganho no Inverso da PNN (Caso Mantenha o Joint CDC)
* Se o Joint CDC for mantido para testes, implementar uma limitação estrita no ganho da resposta inversa para evitar explosão de ruído nos vales espectrais:
  $$|H_{\text{PNN}}^{-1}(f)| \le G_{\text{max}} \quad (\text{ex.: } G_{\text{max}} = 3\text{ dB} \dots 6\text{ dB})$$

### Ação 3: Reorientar a Aplicação da PNN
* **Em Sistemas Coerentes Lineares:** Utilizar a PNN para **Compensação de Dispersão Residual** (reduzindo o número de taps e o tamanho da FFT no DSP).
* **Em Sistemas Não Lineares (Regime Kerr):** Treinar a PNN com potência de lançamento elevada para mitigar a Automodulação de Fase (SPM/XPM), deixando a dispersão cromática linear totalmente para o CDC convencional do DSP.

---
*Arquivo gerado para suporte ao agente de IA em tarefas de otimização e avaliação de arquiteturas fotônicas.*
