function [P,PdBm,Ppeak] = Potencia_Sinal(u);
%
% function [P,PdBm,Ppeak] = Potencia_Sinal(u);
%
%     Calcula a potência do sinal u em Watts e dBm.
%  O pico de potência também é calculado.
%
%                                             by M.Segatto
%                                            10-11/02/2004


P  = sum(abs(u).^2)/length(u); % Potência do sinal [W]

if nargout >= 2,
  PdBm = 30 + 10*log10(P);      % Potência do sinal [dBm]
end
if nargout == 3,
  umax = max(u);
  Ppeak = umax^2;      % Potência do pico do sinal [W]
end

