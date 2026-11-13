% 	SCM_FechamentoL
%                 Estuda o efeito de "RF power fading" em fun��o do
%          comprimento L da fibra.
%
%   Usa: SCM_Input_Data, time2freq,
%        Laser_DFB_Simplificado, Fibra_Linear, Fotodiodo
%
%                                by M.Segatto
%                                  30/08/2003
close all; clear; clc;
SCM_Input_Data;
%
Fibra_Monomodo_Input_Data;
%
global ipulso qpulso
contL = 0;
for L = 0:1:100, %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  contL = contL + 1;
  LL(contL) = L;
% L = input('L ? [km] ');
TelaInfo2;
%
%  Modulação para cada canal  %%% separar a modulacao da multiplexacao
%
% modulado = 0;
if ~exist('vet'),
  vet             = randi(n_portadoras,NB);  % bits
end
for ii = 1:n_portadoras,
  [qam(ii,:),t,I(ii,:),Q(ii,:),BQAM]  = Modulacao_QAM(vet,mod_type,Fc(ii),B,NPPB);
end
%
 mux = Mux_FDM(qam);
%
f   = fftshift( time2freq(t)); % Frequencia [Hz]
tps = t*1E12;                      % [ps]
%
if bb_eletrico == 0,
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  laser DFB (modelo simplificado   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
  ent_laser = mux;
  Ilaser    = Ifator*(ent_laser-min(ent_laser))/(max(ent_laser)-min(ent_laser));
%
  P         =  Laser_DFB_Simplificado(Ilaser,Ibias,Ith,k);
  ent_fibra = sqrt(P);     % Campo elétrico
%
  if bb_optico == 0,
%%%%%%%%%%%%%%%%%%%%%%%%%
%  Fibra �ptica         %
%%%%%%%%%%%%%%%%%%%%%%%%%
%
    [Po,PdBm]   = Potencia_Sinal(ent_fibra);
    sai_fibra   = Fibra_Monomodo(t,ent_fibra,lamb0*1E-9,1/BQAM,L,regime);
%
  elseif bb_optico == 1,
    sai_fibra = ent_fibra;
  end
%%%%%%%%%%%%%%%%%%%%%%%%%
%    Fotodetector       %
%%%%%%%%%%%%%%%%%%%%%%%%%
%
  foto = Fotodiodo(sai_fibra);
%
elseif bb_eletrico == 1,
  foto = mux;
end
%
%%%%%%%%%%%%%%%%%%%%%%%%%
%  Demultiplex FDM
%%%%%%%%%%%%%%%%%%%%%%%%%%
%
for ii = 1:n_portadoras,
  demux(ii,:) = Demux_FDM(foto,t,Fc(ii),B);
end
%
%%%%%%%%%%%%%%%%%%%%%%%%%
%     Demodulação       %
%%%%%%%%%%%%%%%%%%%%%%%%%
%
for ii = 1:n_portadoras,
  [Irec(ii,:),Qrec(ii,:),iqam(ii,:),qqam(ii,:),...
   iclock_out(ii,:),qclock_out(ii,:)] = Demodulacao_QAM(demux(ii,:),t,Fc(ii),BQAM);
end
%%%%%%%%%%%%%%%%%%%%%%%%%
%       Gráficos        %
%%%%%%%%%%%%%%%%%%%%%%%%%
%
if graficos,
  ii = ncanal;
  figure(2),Diagrama_Olho(tps,iqam(ii,:),1,0,BQAM,NPPB,length(I(ii,:)));
  title(sprintf('Canal %d, componente em fase',ii));
%
  figure(1),plot(I,Q,'b.',Irec(ii,:),Qrec(ii,:),'r.'),...
  axis(1.15*max(Irec(ii,:))*[-1 1 -1 1]),grid,legend('Sinal transmitido','Sinal recebido'),...
  xlabel('I'),ylabel('Q'),title(sprintf('Canal %d',ii));
%
end
%%%%%%%%%%%%%%%%%%%%%%%%%
%   Abertura do olho   %
%%%%%%%%%%%%%%%%%%%%%%%%%
ii = ncanal;
Abertura(contL) = Diagrama_Olho(tps,iqam(ii,:),1,0,B,NPPB,length(I(ii,:)));
%
% pause(1)
end
AberturadB = 21*log10(Abertura/Abertura(1));
figure(3),plot(LL,AberturadB,'r'),grid,...
xlabel('Comprimento da fibra [km]'),ylabel('Fechamento do olho [dB]')
