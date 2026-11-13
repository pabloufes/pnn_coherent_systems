function [Irec,Qrec,iqam,qqam,iclock_out,qclock_out] = Demodulacao_QAM(qam,t,Fc,B,Demodulacao_QAM_Input_File);
%
%   function [Irec,Qrec,iqam,qqam,iclock_out,qclock_out] = Demodulacao_QAM(qam,t,Fc,B);
%
%     Retorna o sinal demodulado QAM.
%
%      Entrada:
%               qam          : Sinal QAM recebido [u.a]
%               t            : Vetor tempo [s]
%               Fc           : Freqüência da portadora
%               B            : Taxa de bits [bit/s]
%               Demodulacao_QAM_Input_File
%                            : Arquivo com os parâmetros do demodulador
%
%      Saida:
%               Irec         : Sinal I recebido
%               Qrec         : Sinal Q recebido
%               iqam         : Sinal em fase (I) QAM em função do tempo
%               qqam         : Sinal em (Q)uadratura QAM em função do tempo
%               iclock_out   : Relógio recuperado para o sinal I
%               qclock_out   : Relógio recuperado para o sinal Q
%
%      Usa: time2freq, Clock_Recovery, Filtros
%
%                                      by M.Segatto
%                                        25/08/2003
if (nargin <= 4),
  Demodulacao_QAM_Input_File = 'Demodulacao_QAM_Input_Data';
end
eval(Demodulacao_QAM_Input_File);
%
% global kk_demod
global diclock dqclock iamo qamo
idem = qam*sqrt(2).*cos(2*pi*Fc*t);
qdem = -1*qam*sqrt(2).*sin(2*pi*Fc*t);
IDEM = fft(idem);
QDEM = fft(qdem);
%
%   Filtro
%

%
f     = fftshift(time2freq(t));
%
switch filtro_tipo
  case 'g'             % Gaussiano
     fFWHM = kk_demod*B;    % Largura de banda
     H     = FiltroGaussiano(f,fFWHM,0,m);
  case 'c'             % Cosseno levantado
    H    = FiltroCossenoLevantado(f,B,alfa);
    H    = H/max(abs(H));
  otherwise
    H    = ones(1,length(f));
  end
%
iqam = real(ifft(IDEM.*H));
qqam = real(ifft(QDEM.*H));
%
%figure(10),plot(f,abs(H),f,abs(IDEM)/max(abs(IDEM))),grid
%
%    recuperaÃ§ao de relógio e amostragem do sinal recebido.
%
iclock_out = Clock_Recovery(t,iqam,B);
qclock_out = Clock_Recovery(t,qqam,B);
%
diclock    = diff(iclock_out);
dqclock    = diff(qclock_out);
iamo       = find(diclock > 0);
qamo       = find(dqclock > 0);
Irec       = iqam(iamo);
Qrec       = qqam(qamo);

