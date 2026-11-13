function [demux,H] = Demux_FDM(sinal,t,Fc,B,Demux_FDM_Input_File);
%
%   function demux = Demux_FDM(sinal,t,Fc,B);
%
%     Faz a demultiplexaÄ‡Ã£o FDM de sinal.
%
%      Entrada:
%               sinal  : Sinal FDM [u.a]
%               t      : Vetor tempo [s]
%               Fc     : Freqüência da portadora
%               B      : Taxa de bits [bit/s]
%               Demux_FDM_Input_File
%                      : Arquivo com os parâmetros do demultiplexer
%
%      Saida:
%               demux  : Sinal demultiplexado em FDM
%               H      : Função de transferência do filtro utilizado
%
%      Usa: time2freq, Filtros
%
%                                      by M.Segatto
%                                        31/08/2003
if (nargin <= 4),
  Demux_FDM_Input_File = 'Demux_FDM_Input_Data';
end
eval(Demux_FDM_Input_File);
%
f        = fftshift(time2freq(t));
SINAL    = fft(sinal);
%
switch filtro_tipo
  case 'g'                     % Gaussiano
    fFWHM  = kk_demux*B;       % Largura de banda
    H      = FiltroGaussiano(f,fFWHM,Fc,m) + FiltroGaussiano(f,fFWHM,-Fc,m);
  otherwise
    error('Esse filtro só pode ser Gaussiano. Por enquanto!')
  end
%
DEMUX = H.*SINAL;
demux = real(ifft(DEMUX));
%

%% HdB = 10*(abs(H)/max(abs(H)));
%% SdB = 10*(abs(SINAL)/max(abs(SINAL)));
%% DdB = 10*(abs(DEMUX)/max(abs(DEMUX)));
%% figure(6),plot(f,HdB,'k',f,SdB,'g.',f,DdB,'r'),grid
%% v = axis; v(4) = 5; v(3) = -50;%axis(v),xlabel('freq. [GHz]'),ylabel('Amplitude [dB]')
