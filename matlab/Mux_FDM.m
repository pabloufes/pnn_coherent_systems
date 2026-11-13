function mux = Mux_FDM(sinal);
%
%   function mux = Mux_FDM(sinal);
%
%     Faz a multiplexação FDM de sinal.
%
%      Entrada:
%                sinal   : Sinal FDM [u.a]
%
%      Saida:
%                mux     : Sinal multiplexado em FDM
%
%                                      by M.Segatto
%                                        31/08/2003

n_portadoras = min(size(sinal));
%
mux          = 0;
for ii = 1: n_portadoras,
  mux  = mux + sinal(ii,:);
end
