# Relatório de resultados — BER por distância com PNN, CDC e joint CDC

## Objetivo

Documentar a execução do notebook `ber_distancia_pnn_cdc_joint_cdc.ipynb` para avaliação independente. O experimento compara BER em distâncias de 50 a 1000 km, em passos de 50 km, para três larguras de linha TX/LO: 0 Hz, 100 kHz e 1 MHz.

A figura gerada está em `artifacts/figures/ber_distancia_pnn_cdc_joint_cdc_tres_larguras_linha.png`.

## Configuração observada na execução

- Modulação: QPSK; taxa de símbolo 10 GBaud; 8 amostras por símbolo (80 GSa/s).
- Fibra: SSMF, dispersão D = 16 ps/(nm·km), atenuação 0.2 dB/km.
- Propagação em spans uniformes de 50 km com EDFA ideal compensando a perda de cada span.
- Fibra em regime linear (`regime=0`); efeito Kerr desligado.
- Potência óptica recebida normalizada para 0 dBm após o enlace.
- Ruído de fase TX/LO correlacionado com ρ = 0.99; larguras de linha 0, 100 kHz e 1 MHz.
- Quadro de treino: 512 símbolos; quadro de teste independente: 4096 símbolos.
- PNN: 8 canais, atraso base de 25 ps, normalização de potência habilitada.
- Otimizador: PSO com 60 partículas e 100 iterações (6060 avaliações por treinamento, conforme a implementação do otimizador).
- Regularização do joint CDC: ε = 10⁻³ × média(|H_PNN|²), recalculada para os pesos candidatos.
- Para cada distância e largura de linha, foram executados três treinamentos independentes: objetivo BER sem DSP, objetivo BER após CDC tradicional e objetivo BER após joint CDC. Total nominal: 20 distâncias × 3 larguras de linha × 3 treinamentos = 180 execuções PSO.
- Os mesmos pesos PNN treinados sem DSP alimentam as curvas de PNN isolada, PNN + CDC tradicional com treino separado e PNN + joint CDC com treino separado. As outras duas curvas usam os pesos treinados com o DSP correspondente no objetivo.

## Definição das sete curvas

1. **Sem compensação:** sem PNN e sem equalização CDC.
2. **CDC apenas:** sem PNN; CDC convencional aplicado no DSP.
3. **PNN apenas:** PNN treinada sem DSP e avaliada sem CDC.
4. **PNN + CDC treino separado:** PNN treinada sem CDC; CDC convencional aplicado apenas na avaliação.
5. **PNN + CDC treino conjunto:** PNN treinada para minimizar BER depois da CDC convencional; CDC também aplicado na avaliação.
6. **PNN + joint CDC treino separado:** PNN treinada sem DSP; equalizador conjunto, que inverte a dispersão e tenta inverter a resposta de campo da PNN, aplicado na avaliação.
7. **PNN + joint CDC treino conjunto:** PNN treinada para minimizar BER após o joint CDC; durante a otimização, a resposta em frequência e o equalizador são recalculados com os pesos candidatos.

## Resultados observados na figura

Leitura qualitativa da figura `artifacts/figures/ber_distancia_pnn_cdc_joint_cdc_tres_larguras_linha.png`, em escala logarítmica. Não há séries BER numéricas exportadas no artefato disponível; portanto, não se inferem valores pontuais dos traços do gráfico. O piso visual é aproximadamente `1/(2 × 4096) = 1.22×10⁻⁴`.

- **CDC apenas** permanece no piso visual ao longo das distâncias e nas três larguras de linha. Como os valores no piso podem representar BER abaixo do limite de exibição, inclusive zero erros, a figura não permite distingui-los.
- **Sem compensação** cresce rapidamente com a distância e fica na faixa de décimos nos enlaces longos. A largura de linha muda pouco a tendência geral nesta figura.
- **PNN apenas** começa no piso em distâncias curtas, depois cresce em transição e chega à ordem de 10⁻¹ nos enlaces longos.
- **PNN + CDC treino separado** também degrada rapidamente: depois de distâncias curtas aparece em geral na ordem de 10⁻¹, sem melhoria consistente sobre CDC apenas.
- **PNN + CDC treino conjunto** permanece no piso visual na maior parte dos pontos, mas apresenta picos esparsos, mais visíveis no painel de 1 MHz e também presentes nos painéis de 0 e 100 kHz. Isso indica desempenho irregular em alguns pontos, que precisam de auditoria quantitativa.
- **PNN + joint CDC treino separado** aparece no piso visual ao longo de toda a faixa nos três painéis. O gráfico não mostra a BER real abaixo do piso nem permite confirmar recuperação perfeita.
- **PNN + joint CDC treino conjunto** também aparece no piso visual ao longo de toda a faixa nos três painéis. Assim, a figura atual contradiz a interpretação anterior de BER alta para o joint CDC; qualquer conclusão sobre sua efetividade deve aguardar séries numéricas e validação da cadeia.

Essas observações descrevem somente o gráfico. Como os dados subjacentes não foram salvos em CSV/JSON, não é possível calcular médias, intervalos de confiança, número exato de erros ou diferenças entre estratégias que se sobrepõem no piso. A execução do notebook deixou mensagens de progresso e a figura, mas não persistiu `ber_curves` e `training_records` como dados tabulares.

## Interpretação preliminar

1. No cenário representado, CDC convencional isolado e as duas variantes de joint CDC ficam no piso visual em toda a faixa de distância. A resolução do gráfico impede ordenar essas estratégias ou atribuir BER zero.
2. O treino conjunto da PNN com CDC tradicional geralmente chega ao piso, mas há picos isolados na BER de teste. A discrepância sugere avaliar variabilidade do PSO, generalização do quadro de treino e quantidade exata de erros em cada ponto.
3. O treino separado da PNN não compensa adequadamente a dispersão neste cenário: PNN isolada e PNN + CDC treino separado pioram fortemente com a distância.
4. A pequena separação visual entre larguras de linha é compatível com o valor de correlação TX/LO configurado (ρ = 0.99), mas não demonstra insensibilidade geral ao ruído de fase.
5. O resultado aparentemente favorável de joint CDC requer validação antes de uma conclusão física. A inversa regularizada pode produzir resultados sensíveis a convenção de sinal, alinhamento temporal, bordas do bloco FFT e valor de regularização; verifique também que os pesos apresentados no treino e avaliação são os efetivamente usados.
6. A figura não permite avaliar convergência do PSO. BER em quadro curto é discreta e pode ter platôs, empates e sensibilidade à semente. Além disso, cada ponto usa uma PNN especializada para distância/largura de linha, não uma PNN fixa para todo o enlace.

## Próximos passos para validação

- Exportar as sete séries BER exatas por distância e largura de linha, contagem de erros e número efetivo de bits comparados; também salvar BER de treino, pesos finais e históricos de convergência das 180 otimizações.
- Investigar os picos isolados da curva PNN + CDC treino conjunto e confirmar os valores no piso das curvas joint CDC: verificar BER de treino/teste, pesos finais, contagem de erros e reprodutibilidade com múltiplas sementes.
- Verificar se `cd_compensation_joint_pnn` e o modelo de campo da PNN usam exatamente a mesma convenção de normalização, grid de frequência, atraso e condições de contorno. O equalizador é aplicado em bloco por FFT, enquanto `forward` implementa atrasos com preenchimento de zeros; avaliar o efeito dessas bordas.
- Medir \\(|H_{PNN}(f)|\\), os vales espectrais e o ganho máximo do inverso regularizado para cada PNN treinada com joint CDC. Confirmar que ε relativo à potência média é apropriado para todas as respostas aprendidas.
- Comparar em uma cadeia sintética sem ruído: TX → fibra linear → PNN de pesos fixos → joint CDC. Essa checagem separa erro de implementação do efeito de treinamento e do ruído de fase.
- Confirmar a estimativa de BER mínima: com 4096 símbolos QPSK há 8192 bits de teste, então uma única ocorrência de erro corresponde a 1/8192 ≈ 1.22×10⁻⁴. O floor da figura é justamente esse valor e mascara a distinção entre zero erro e um erro.
- Verificar consistência entre comprimento dos bits TX e símbolos decimados, temporização de amostragem e número de bits efetivamente comparados em `calculate_ber`.

## Artefatos de origem

- Notebook executado: `notebooks/ber_distancia_pnn_cdc_joint_cdc.ipynb`.
- Figura dos resultados: `artifacts/figures/ber_distancia_pnn_cdc_joint_cdc_tres_larguras_linha.png`.
- Implementação joint CDC: `src/utils/joint_pnn_cdc_equalizer.py`.


## Estudo ampliado de BER/EVM e joint CDC

A análise estendida, com varreduras por distância e potência, objetivos PSO BER/EVM e figuras de constelação, está documentada em [`relatorio_analise_pnn_cdc_ber_evm.md`](relatorio_analise_pnn_cdc_ber_evm.md). Ela usa PNN com espaçamento de 25 ps, PSO com 30 partículas e 60 iterações, fibra linear na varredura de distância e SSFM com Kerr na varredura de potência; portanto, seus resultados complementam este experimento e não substituem os valores desta execução anterior.
