% 	SCM_FechamentoCW3dB
%                 Calcula a largura de 3 dB do sistema SCM. Utiliza sub-
%          portadoras sem modulação.
%
%   Usa: SCM_Input_Data, time2freq,
%        Laser_DFB_Simplificado, Fibra_Linear, Fotodiodo
%
%                                by M.Segatto
%                                  30/08/2003



SCM_Input_Data;
%
Fibra_Monomodo_Input_Data;
%
NP    =  1024;
contL = 0;
for L = 5:5:100,
  contF = 0;
  for Fc1 = (2:0.1:20)*1E9, %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
    P         =  Laser_DFB_Simplificado(Ilaser,Ibias,Ith,k);
    ent_fibra = sqrt(P);     % Campo elÃ©trico
%
    if bb_optico == 0,
%%%%%%%%%%%%%%%%%%%%%%%%%
%  Fibra Ã³ptica linear  %
%%%%%%%%%%%%%%%%%%%%%%%%%
%
%    t deve estar em [ps]
%       sai_fibra = Fibra_Linear(tps,ent_fibra,lamb0,L);
      sai_fibra = Fibra_Monomodo(t,(ent_fibra),lamb0*1E-9,1,L,regime);
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
%       GrÃ¡ficos        %
%%%%%%%%%%%%%%%%%%%%%%%%%
  Resposta1dB = 20*log10(Resposta1/Resposta1(1));
  Resposta2dB = 20*log10(Resposta2/Resposta2(1));
%
  contL     = contL + 1;
  LL(contL) = L;
  zz        = find(Resposta1dB > -3.1 & Resposta1dB < -2.9);
  BW(contL) = FF(zz(1));
end
if graficos,
%
  figure(3),plot(LL,BW,'r'),grid on,xlabel('L [km]'),...
  ylabel('Largura de banda de 3 dB [GHz]'),zoom on
title(sprintf('D =  %f [ps/nm.km]',D))

end
