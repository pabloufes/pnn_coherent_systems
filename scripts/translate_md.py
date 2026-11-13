# -*- coding: utf-8 -*-
import re

replacements = [
    (r"# Exploração do PNN em transmissão coerente QPSK com correlação de fase", 
     "# Exploration of PNN in Coherent QPSK Transmission with Phase Correlation"),
    (r"São avaliados \(1\) dispersão cromática sem não linearidade, em função da distância, e \(2\) dispersão \+ efeito Kerr, em função da potência de lançamento para um enlace de 1000 km\. Para cada análise e largura de linha, uma PNN é treinada separadamente e depois avaliada com pesos fixos\.",
     "The notebook evaluates (1) chromatic dispersion without nonlinearity as a function of distance, and (2) chromatic dispersion + Kerr effect as a function of launch power for a 1000 km link. For each analysis and linewidth, a PNN is trained separately and evaluated with fixed weights."),
    (r"As fases de TX e LO são realizações independentes\. Não há compensação digital de dispersão nem recuperação de fase nesta linha de base\.",
     "TX and LO phases are independent realizations. There is no digital chromatic dispersion compensation or carrier phase recovery in this baseline."),
    (r"O coeficiente de correlação `PHASE_CORRELATION_TX_RX` é configurável entre 0 \(processos independentes\) e 1 \(fase comum ideal\)\.",
     "The correlation coefficient `PHASE_CORRELATION_TX_RX` is configurable between 0 (independent processes) and 1 (ideal common phase)."),
    (r"Raiz do projeto:", "Project root:"),
    (r"## 1\. Parâmetros e configuração do enlace", "## 1. Parameters and Link Configuration"),
    (r"`SPAN_LENGTH_KM` é o único tamanho de trecho de fibra usado em todo o enlace\. O comprimento total precisa ser um múltiplo inteiro dele; o caderno interrompe a execução caso essa condição não seja atendida, evitando spans finais menores\. Para comparar trechos de 50, 100 ou 200 km, altere esse parâmetro\.",
     "`SPAN_LENGTH_KM` is the single fiber span length used across the entire link. The total link distance must be an integer multiple of it; the notebook halts execution if this condition is not met, preventing fractional terminal spans. To compare spans of 50, 100, or 200 km, adjust this parameter."),
    (r"Nesta etapa as EDFAs são ideais: não incluem ASE, figura de ruído ou saturação\.",
     "At this stage, EDFAs are ideal: they do not include ASE noise, noise figure, or saturation."),
    (r"# EDFA de linha compensa alpha_db \* SPAN_LENGTH_KM em cada trecho\.",
     "# In-line EDFA compensates alpha_db * SPAN_LENGTH_KM in each span."),
    (r"# Correlação entre a fase do transmissor e a fase do oscilador local do receptor\.",
     "# Correlation between transmitter phase and receiver local oscillator phase."),
    (r"# rho=0: lasers independentes; rho=1: fase comum ideal\.",
     "# rho=0: independent lasers; rho=1: ideal common phase."),
    (r"PHASE_CORRELATION_TX_RX deve estar no intervalo \[0, 1\]\.",
     "PHASE_CORRELATION_TX_RX must be in range [0, 1]."),
    (r"SPAN_LENGTH_KM deve ser positivo\.", "SPAN_LENGTH_KM must be positive."),
    (r"DISTANCE_SWEEP_MAX_KM deve ser múltiplo de SPAN_LENGTH_KM\.",
     "DISTANCE_SWEEP_MAX_KM must be an integer multiple of SPAN_LENGTH_KM."),
    (r"DISTANCE_NONLINEAR_KM deve ser múltiplo de SPAN_LENGTH_KM\.",
     "DISTANCE_NONLINEAR_KM must be an integer multiple of SPAN_LENGTH_KM."),
    (r"Potência-alvo no receptor:", "Target power at receiver:"),
    (r"Larguras de linha TX = LO:", "TX = LO linewidths:"),
    (r"Correlação de fase TX-RX:", "TX-RX phase correlation:"),
    (r"## 2\. Funções de propagação e potência", "## 2. Propagation and Power Functions"),
    (r"Cada span propaga o campo pela fibra e aplica uma EDFA ideal com ganho igual à perda daquele trecho\. O estágio terminal aplica ganho fixo e calcula a atenuação necessária para obter a potência-alvo no receptor\.",
     "Each span propagates the field through fiber and applies an ideal EDFA with gain equal to the span loss. The terminal stage applies fixed gain and computes the required attenuation to reach the target receiver power."),
    (r"Potência óptica média do envelope, em dBm\.", "Mean optical envelope power in dBm."),
    (r"Propaga spans uniformes, cada um seguido por EDFA de ganho compensador\.",
     "Propagates uniform spans, each followed by an EDFA with compensating gain."),
    (r"Aplica EDFA terminal fixa e VOA ideal para fixar a potência recebida\.",
     "Applies fixed terminal EDFA and ideal VOA to set received power."),
    (r"## 3\. Sinal de referência e ruído de fase", "## 3. Reference Signal and Phase Noise"),
    (r"# Referência de 0 dBm; a varredura de potência escala linearmente o campo\.",
     "# 0 dBm reference; the power sweep linearly scales the field."),
    (r"## 4\. Análise 1 — BER versus distância: dispersão cromática",
     "## 4. Analysis 1 — BER vs. Distance: Chromatic Dispersion"),
    (r"## 5\. Análise 2 — BER versus potência: dispersão \+ não linearidade",
     "## 5. Analysis 2 — BER vs. Launch Power: Dispersion + Nonlinearity"),
    (r"## 6\. Treinamento independente da PNN para cada análise",
     "## 6. Independent PNN Training for Each Analysis"),
    (r"## 7\. Avaliacao com PNN treinada", "## 7. Evaluation with Trained PNN"),
    (r"## Observações e próximas etapas", "## Observations and Next Steps"),
]

def main():
    path = "notebooks/exploracao_pnn_dispersao_edfa_correlacao_fase.md"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)

    out_path = r"C:\Users\Pablo\.gemini\antigravity\brain\c5489970-8f41-439f-91e0-e32e5923a6cc\scratch\translated.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)

    print("Successfully translated to scratch/translated.md")

if __name__ == "__main__":
    main()
