%
%    Arquivo de entrada do programa SCM.
%
%addpath('Optico','Filtros')
%
B             = 6E9;     % Taxa de bits [bit/s]
mod_type      = 'QPSK';  % Tipo de modulacão usado nas subportadoras
M             = 4;       % M = 2^m � a ordem do QAM utilizado
NB            = 4^4;     % Número de bits
NPPB          = 64;     % Número de pontos por bits
%
bb_eletrico     = 0;     %   Define simulacão no dom�nio el�trico
                         % ou optico (0) �ptico (1) el�trico
bb_optico       = 0;     %   Define se sinal propaga na fibra ou n�o
                         % (0) propaga, (1) o sinal de sa�da � igual ao entrada
%
SNR          = 15;       % Relacao sinal-ruido no canal AWGN
%
%   Par�metros do sistema �ptico
%
L      = 100;      % Comprimento da fibra [km]
lamb0  = 1563;    % Comprimento de onda de operação [nm]
regime = 0;       % Regime linear (0) ou n�o linear (1) da fibra
Ibias  = 100;     % Corrente de polarizaćão do laser [mA]
Ith    = 30;      % Corrente de limiar [mA]
k      = 0.999;   % Fator de linearidade do modelo Laser_DFB_Simplificado
Ifator = 50;      % Amplitude do sinal modulante  [mA]
supres = 8;       % Fator de supress�o da portadora �ptica [dB]
%
%   Par�metros do sistema eletrico
%
espacamento  = 'naouniforme';
switch espacamento
case 'uniforme'
  n_portadoras = 1;      % # de portadoras
  Fc1          = 2E9;    % Freq��ncia da primeira subportadora [Hz]
  Fespacamento = 1E9;    % Espacamento entre as subportadoras [Hz]
  for ii = 1:n_portadoras,
    Fc(ii) = Fc1 + (ii - 1)*Fespacamento;
  end
%
case 'naouniforme'
  Fc           = (4:6)*1E9;        % Freq. das sub-portadoras [Hz]
  n_portadoras = length(Fc);      % # de portadoras
otherwise
  error('Tipo de espacamento em frequencia invalido!!!')
end
%
%%%%%%%%%%%%%%%%%%%%%%%%
%     Convers��es      %
%%%%%%%%%%%%%%%%%%%%%%%%
%
T             = 1/B;
%
% Declara��o de vari�veis globais
%
global kk_demod
global kk_demux
%

%%%%%%%%%%%%%%%%%%%%%%%%
%      Graficos        %
%%%%%%%%%%%%%%%%%%%%%%%%
%
graficos     = 1;                        % Mostra ou não os gráficos na saída
ncanal       = 1;%%n_portadoras;       % # do canal a ser mostrado (caso grafico = 1)
