function P = Laser_DFB_Simplificado(Imod,Ibias,Ith,k);
%
%          function sai_laser = Laser_DFB_Simplificado(Imod,Ibias,Ith,k);
%
%    Implementa um laser DFB utilizando somente a curva PxI acima do limiar
%
%  Entrada:
%                        Imod : Corrente de modulação do laser [mA]
%                        Ibias : Corrente de polarização do laser [mA]
%                        Ith : Corrente de limiar do laser [mA]
%                        k : Inclinação da curva PxI, i.e. P = k*(Ibias + Imod)   [mW/mA]
%
%  Saída:
%                        P : Potênciaóptica de saída do laser  [mW]
%                                             by M.Segatto
%                                               14/08/2003
%
if (min(Imod + Ibias) < Ith),
  error('Esse modelo não funciona para valores de corrente abaixo do limiar de corrente');
end
if (min(Imod + Ibias) < 1.2*Ith),
  warning('ATENÇÃO!!! Valores de corrente muito próximos ao limiar de corrente');
end
%
 P = k*(Ibias + Imod);
 % P = P*1E-3;     % Potencia em [mW]
