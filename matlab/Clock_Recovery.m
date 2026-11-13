function clock_out = Clock_Recovery(t,a,B);
%
%  function clock_out = Clock_Recovery(t,a,B);
%
%    Recupera o relógio em uma sequência de pontos de um sinal digital
%
%      Entrada :
%                  t : Vetor tempo;
%                  a : Sinal digital;
%                  B : Taxa de bits.
%      Saida :
%                  clock_out : Sinal do relógio recuperado.
%
%      Usa:
%                  FiltroGaussiano
%
%                                  by M.Segatto
%                                  08/08/2003

NPontos = length(a);
T       = 1/B;
%
%   Fator de escala do sinal de entrada
%
A    = fft(a);
A(1) = 0;
a    = real(ifft(A));

%
%  Comparador
%
b  = zeros(1,NPontos);
for ii = 1:NPontos,
  if (a(ii) > 0),
    b(ii) = 1;
  end
end
%
% Detector de subida e descida
%
c = [0 diff(b)];
%
% Retificador de onda completa
%
d = abs(c);
D = fft(d);
%
%     Circuito tanque (neste caso um capacitor em paralelo com
%  um indutor de resistência interna R)
%
dt   = t(2) - t(1);
df   = 1/(dt*NPontos);
freq = fftshift(df*(-NPontos/2:NPontos/2-1));
w    = 2*pi*freq;
%
%   Filtro gaussiano
%
    f           = fftshift( time2freq(t));
    fFWHM       = B/1000;    % Largura de banda
    fc          = 1/T; % Freq. central
    m           = 5;  % Ordem da super-gaussiana
%
    H           = FiltroGaussiano(f,fFWHM,fc,m);
%
%% f0 = 1/T;
%% w0 = 2*pi*f0;
%% R  = 0.001;
%% C  = 1;
%% L  = 1/(C*w0^2);
%% %
%% H  = (R + i*w*L)./(1 + i*w*R*C - L*C*w.^2);
%
E  = D.*H;
e  = real(ifft(E));

%
%  Comparador final
%
clock_out  = zeros(1,NPontos);
for ii = 1:NPontos,
  if (e(ii) > 0),
    clock_out(ii) = 1;
  end
end
