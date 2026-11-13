%
% 	SCM_FechamentoCW
%                 Estuda o efeito de "RF power fading" utilizando sub-
%          portadora sem modula��o.
%
%   Usa: SCM_Input_Data, time2freq,
%        Laser_DFB_Simplificado, Fibra_Linear, Fotodiodo
%
%                                by M.Segatto
%                                  30/08/2003

SCM_Input_Data;
% L = input('L = ')
L = 100;
%
Fibra_Monomodo_Input_Data;
%
NP    =  256;
contF = 0;
for Fc1 = (0.1:0.01:60)*1E9, %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  Fc        = Fc1;
  T         = 1/Fc;
  Tf        = 10*T;
  contF     = contF + 1;
  FF(contF) = Fc1/1E9;
%
  t         = linspace(0,Tf,NP);    % [s]
  f         = fftshift( time2freq(t)); % Frequencia [Hz]
  tps       = t*1E12;                      % [ps]
  sinal     = cos(2*pi*Fc*t);
  SINAL     = fft(sinal);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  laser DFB (modelo simplificado   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
  ent_laser = sinal;
  Ilaser    = Ifator*(ent_laser-min(ent_laser))/(max(ent_laser)-min(ent_laser));
%
  P         = Laser_DFB_Simplificado(Ilaser,Ibias,Ith,k);
  P         = P*1E-3;      % Potencia [W]
  ent_fibra = sqrt(P);     % Campo eletrico
%
  if bb_optico == 0,
%%%%%%%%%%%%%%%%%%%%%%%%%
%     Fibra Otica       %
%%%%%%%%%%%%%%%%%%%%%%%%%
%
[Po,PdBm] = Potencia_Sinal(ent_fibra);
    [Po,PdBm]   = Potencia_Sinal(ent_fibra);
    sai_fibra   = Fibra_Monomodo(t,(ent_fibra),lamb0*1E-9,1,L,regime);
%
  elseif bb_optico == 1,
    sai_fibra = ent_fibra;
  end
%%%%%%%%%%%%%%%%%%%%%%%%%
%    Fotodetector       %
%%%%%%%%%%%%%%%%%%%%%%%%%
%
  foto    = Fotodiodo(sai_fibra);
  FOTO    = fft(foto);
  FOTO(1) = 0;
%
  Resposta1(contF) = max(abs(FOTO(1:NP/2)));
  Resposta2(contF) = max(abs(FOTO(NP/2+1:NP)));
end
%%%%%%%%%%%%%%%%%%%%%%%%%
%       Gráficos        %
%%%%%%%%%%%%%%%%%%%%%%%%%
Resposta1dB = 20*log10(Resposta1/Resposta1(1));
Resposta2dB = 20*log10(Resposta2/Resposta2(1));
%
if graficos,
%
figure(),plot(FF,Resposta1dB,'LineWidth',1),grid on,xlabel('Frequency [GHz]'),...
ylabel('Normalized frequency response [dB]') %,zoom on,legend('Parte positiva','Parte negativa')
%title(sprintf('L =  %f [km]',L))
a=1;
end

