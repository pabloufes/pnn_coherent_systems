


ii = 1;

%    function [Tolho,olho] = Diagrama_Olho(T,pulse,n_bits,offset,Bit_Rate,Sample_Rate,Sequence_Length);

pulse = iqam(ii,:);

[Aertura,Tolho,olho] = Diagrama_Olho(tps,pulse,1,0,B,NPPB,length(I(ii,:)));


nbins    = 200;
tmp      = olho(NPPB/2,:);
x        = linspace(min(tmp),max(tmp),nbins);
y        = hist(tmp,x);
[m,pos0] = max(y(1:nbins/2));         % maior incidência do bit 0
[m,pos1] = max(y(nbins/2+1:nbins));   % maior incidência do bit 1
pos1     = pos1 + nbins/2;
%
Abertura = x(pos1) - x(pos0)