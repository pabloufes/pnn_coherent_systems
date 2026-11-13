function [qam,t,I,Q,BQAM] = Modulacao_QAM(vet,mod_type,Fc,B,NPPB);
%
% function [qam,t,I,Q,BQAM] = Modulacao_QAM(vet,mod_type,Fc,B,NPPB);
%
%     Retorna o sinal modulado QAM.
%
%      Entrada:
%               vet        : Vetor contendo a informação binária a ser modulada
%               mod_type   : Tipo da modulação que será utilizada (m-ASK, m-PSK, m-QAM,...)
%               Fc         : Freqüência da portadora [Hz]
%               B          : Taxa de bits [bit/s]
%               NPPB       : Número de pontos por bit
%
%      Saída:
%               qam        : Vetor contendo o sinal QAM
%               t          : Vetor tempo [s]
%               I,Q        : Sinais I e Q para gerar a constelação
%               BQAM       : Taxa de símbolos [símbolo/s]
%
%      Usa: QAM_Tabela, Clock_Recovery, Filtros
%
%                                      by M.Segatto
%                                        25/08/2003

%    Gera vetores I e Q
%
global ipulso qpulso
[I,Q,T,BQAM] = QAM_Tabela(vet,mod_type,B);
%
%   Gera pulsos I e Q usando um ZOH
%
NB2  = length(I);
cont = 1;

t    = linspace(0,NB2*T,NB2*NPPB);    % [s]
%
for ii = 1:NB2,
  ipulso(cont:ii*NPPB) = I(ii);
  qpulso(cont:ii*NPPB) = Q(ii);
  cont                 = cont + NPPB;
end
%
%   Filtro
%
filtro_tipo = 'g'; % 'r', 'c' ou 'g'
%
f           = fftshift(time2freq(t));
%
%% IPP1 =abs(fft(ipulso));
%
switch filtro_tipo
  case 'g'  % Gaussiano
    kk    = 2;
    fFWHM = kk*BQAM;    % Largura de banda
    m     = 1;       % Ordem da super-gaussiana
    H     = FiltroGaussiano(f,fFWHM,0,m);
  case 'c'  % Cosseno levantado
    alfa = 10;
    H    = FiltroCossenoLevantado(f,BQAM,alfa);
    H    = H/max(abs(H));
  case 'r'  % RC
    k        = 1;
    [freq,H] = FiltroRC(t,BQAM,k,1);
    H        = H/max(abs(H));

  end
%
ipulso = real(ifft(H.*fft(ipulso)));
qpulso = real(ifft(H.*fft(qpulso)));
%
%figure(5),plot(f,700*abs(H),f,abs(fft(ipulso)),'g.',f,IPP1,'r'),grid
%legend('Filtro','Após o filtro','Antes'),zoom on
%
%   Modulação QAM
%
    imod  = ipulso*sqrt(2).*cos(2*pi*Fc*t + 1/Fc);
    qmod  = -1*qpulso*sqrt(2).*sin(2*pi*Fc*t);
    qam   = imod + qmod;
%%
iclock_out = Clock_Recovery(t,ipulso,BQAM);
qclock_out = Clock_Recovery(t,qpulso,BQAM);
%
diclock    = diff(iclock_out);
dqclock    = diff(qclock_out);
iamo       = find(diclock > 0);
qamo       = find(dqclock > 0);
if sum(I) ~= 0,
  I        = ipulso(iamo);
end
if sum(Q) ~= 0,
  Q        = qpulso(qamo);
end
