%%%%%%%%%%%%%      %%%%      %%%%  %%%%%  %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%  %%%%%%%%  %%%%%%%%   %%%   %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%  %%%%%%  %%%%%%%%  %   %  %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%  %%%%  %%%%%%%%  %% %%  %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%      %%%%      %%%%  %%%%%  %%%%%%%%%%%%%%%%%
%
%      Sistema de sub-portadoras ópticas.
%      Implementa e analisa sistemas de sub-portadoras ópticas. Os
%   parâmetros do sistema estão definidos no arquivo : SCM_Input_Data.
%
%
%   Usa: SCM_Input_Data, time2freq, Diagrama_Olho
%        Modulacao_QAM, Demodulacao_QAM,
%        Mux_FDM, Demux_FDM,
%        Laser_DFB_Simplificado, Fibra_Multimodo, Fotodiodo
%
%                                      by M.Segatto
%                                        30/08/2003
clear all;
SCM_Input_Data;
%
Fibra_Monomodo_Input_Data;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    Verificacao da freq. maxima representada     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
Fcmax      = max(Fc);
Fmax       = NPPB*B/(2*log2(M));
if (Fmax <= Fcmax)
    error('A máxima freqüência representada no seu sistema é inferior a maior frequencia das sub-portadoras.')
end
%
%%   global f    % Calculo feito dentro da rotina Modulacao_QAM
global ipulso qpulso
global diclock dqclock iamo qamo
TelaInfo2;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           Modulacao QAM           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
t0              = clock;
if ~exist('vet'),
  vet             = randi(n_portadoras,NB);  % bits
  vet(:,NB)       = 0;
  vet(:,1)        = 0;
%
  for ii = 1:n_portadoras,
    [qam(ii,:),t,I(ii,:),Q(ii,:),BQAM]  = Modulacao_QAM(vet(ii,:),mod_type,Fc(ii),B,NPPB);
  end
end
t1 = clock;time = etime(t1,t0);
disp(strcat('Modulacao_QAM :   ',num2str(time),' [s]'))

%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%               F D M               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
 mux = Mux_FDM(qam);
 t2 = clock;time = etime(t2,t1);
 disp(strcat('Mux_FDM :   ',num2str(time),' [s]'))
%
 f   = fftshift( time2freq(t)); % Frequencia [Hz]
%
tps = t*1E12;                      % [ps]
%
if bb_eletrico == 0,
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  laser DFB (modelo simplificado)  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
  ent_laser = mux;
  Ilaser    = Ifator*(ent_laser-min(ent_laser))/(max(ent_laser)-min(ent_laser));
%
  P         = Laser_DFB_Simplificado(Ilaser,Ibias,Ith,k);
  P         = P*1E-3;      % Potencia [W]
  ent_notch = sqrt(P);     % Campo eletrico
  t3 = clock;time = etime(t3,t2);
  disp(strcat('Laser_DFB :   ',num2str(time),' [s]'))
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Supressao da portadora óptica    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
ENT_NOTCH    = fft(ent_notch);
tmp          = 10^(supres/20);
ENT_NOTCH(1) = ENT_NOTCH(1)/tmp;
ent_fibra    = real(ifft(ENT_NOTCH));
t4 = clock;time = etime(t4,t3);
disp(strcat('Filtro Notch :   ',num2str(time),' [s]'))
%

  if bb_optico == 0,
%%%%%%%%%%%%%%%%%%%%%%%%%
%  Fibra Óptica         %
%%%%%%%%%%%%%%%%%%%%%%%%%
%
    fator       = 5;
    [Po,PdBm]   = Potencia_Sinal(ent_fibra/fator);
    sai_fibra   = Fibra_Monomodo(t,ent_fibra/fator,lamb0*1E-9,1/BQAM,L,regime);
%
  elseif bb_optico == 1,
    sai_fibra = ent_fibra;
  end
  t5 = clock;time = etime(t5,t4);
  disp(strcat('Fibra óptica :   ',num2str(time),' [s]'))
%
%%%%%%%%%%%%%%%%%%%%%%%%%
%    Fotodetector       %
%%%%%%%%%%%%%%%%%%%%%%%%%
%
  foto = Fotodiodo(sai_fibra);
  FOTO = abs(fft(foto));
  t6 = clock;time = etime(t6,t5);
  disp(strcat('Fotodiodo :   ',num2str(time),' [s]'))

%
elseif bb_eletrico == 1,
  foto = mux;
  FOTO = abs(fft(foto));
end
%%%%%%%%%%%%%%%%%%%%%%%%%
%    Canal AWGN         %
%%%%%%%%%%%%%%%%%%%%%%%%%
awgn = CanalAWGN(foto,SNR);
t7 = clock;time = etime(t7,t6);
disp(strcat('AWGN :   ',num2str(time),' [s]'))

%
%%%%%%%%%%%%%%%%%%%%%%%%%
%  Demultiplex FDM      %
%%%%%%%%%%%%%%%%%%%%%%%%%
%
for ii = 1:n_portadoras,
  [demux(ii,:),H] = Demux_FDM(awgn,t,Fc(ii),BQAM);
end
  t8 = clock;time = etime(t8,t7);
  disp(strcat('Demux_FDM :   ',num2str(time),' [s]'))
%
%%%%%%%%%%%%%%%%%%%%%%%%%
%     Demodulacao       %
%%%%%%%%%%%%%%%%%%%%%%%%%
%
for ii = 1:n_portadoras,
  [Irec(ii,:),Qrec(ii,:),iqam(ii,:),qqam(ii,:),...
   iclock_out(ii,:),qclock_out(ii,:)] = Demodulacao_QAM(demux(ii,:),t,Fc(ii),BQAM);
end
t9 = clock;time = etime(t9,t8);
disp(strcat('Demodulacao_QAM :   ',num2str(time),' [s]'))

%%%%%%%%%%%%%%%%%%%%%%%%%
%       Graficos        %
%%%%%%%%%%%%%%%%%%%%%%%%%
%
if graficos,
  tt0 = clock;
  ii = ncanal;
  figure(2),Diagrama_Olho(tps,iqam(ii,:),1,0,BQAM,NPPB,length(I(ii,:)));
  title(sprintf('Canal %d, componente em fase',ii));
%
  figure(1),plot(I/max(I(ii,:)),Q/max(Q(ii,:)),'b.',Irec(ii,:)/max(Irec(ii,:)),Qrec(ii,:)/max(Qrec(ii,:)),'r.'),...
axis(1.5*max(max([Irec(ii,:)/max(Irec(ii,:)) I(ii,:)/max(I(ii,:))]))*[-1 1 -1 1]),grid,legend('Sinal transmitido','Sinal recebido'),...
  xlabel('I'),ylabel('Q'),title(sprintf('Canal %d',ii));%
  time = etime(clock,tt0);
disp(strcat('Graficos :   ',num2str(time),' [s]'))

end
%
% Tempo
%
t10 = clock;time = etime(t10,t0);
disp(strcat('Tempo total :   ',num2str(time),' [s]'))
