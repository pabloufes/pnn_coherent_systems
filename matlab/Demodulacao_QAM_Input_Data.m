%
%         Dados de entrada para a função Demodulacao_QAM. Contém
%         as informações sobre o tipo de filtro que deve ser usdo
%         no processo de demodulação.
%
%                                        by M.Segatto
%                                          16/12/2003

filtro_tipo = 'g'; % 'c' ou 'g' otherwise sem filtro
%
kk_demod     = 1;
m            = 5;       % Ordem da super-gaussiana
alfa         = 0.95;    % Fator de rolloff do filtro cosseno-levantado
