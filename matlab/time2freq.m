function freq = time2freq(t);
%
%           function freq = time2freq(t);
%
%      Gera o vetor de freqüência para um dado vetor
%      tempo para ser usado junto com a FFT.
%
%   Entrada:
%       T            : Vetor tempo    [s]
%
%   Saida:
%       freq         : Vetor freqüência  [Hz]
%
%                                 by M.Segatto
%                                    2002/2003

dt    = mean(diff(t));
nt    = length(t);
nt2   = nt/2;
df    = inv(dt*nt);
%
freq   = df*(-nt2:nt2 - 1);
