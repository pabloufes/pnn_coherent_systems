function [Abertura,Tolho,olho]=Diagrama_Olho(T,pulse,n_bits,offset,Bit_Rate,Sample_Rate,Sequence_Length);
%
%    function [Abertura,Tolho,olho]=Diagrama_Olho(T,pulse,n_bits,offset,Bit_Rate,Sample_Rate,Sequence_Length);
%
%       Calcula o diagrama e a abertura de olho para uma dada seqüência de bits. Se o número de saídas
%  for nulo, plota o diagrama e o histograma vertical da seqüência.
%
%   ENTRADA:
%    T                  : Vetor tempo [ps]
%    pulse              : Vetor contendo o sinal [u.a]
%    n_bits             : Número de bits a serem mostrados no diagrama
%    offset             : Offset [ps]
%    Bit_Rate           : Taxa de Transmissão [Gbit/s]
%    Sample_Rate        : Número de pontos por bit
%    Sequence_Length    : Número de bits da seqüência
%
%   SAÍDA:
%    Tolho              : Vetor tempo para o diagrama de olho [ps]
%    olho               : Matriz contendo o diagrama de olho
%    Abertura           : Abertura do olho
%
%                                       by M.Segatto
%                                         18/02/2002
%                                       Atualizado por
%                                       Arnaldo P. Togneri 06/09/03   <- Inclusão de 'Abertura'
%
if (offset < 0),
    offset = -1*offset;
end
%
n_bits    = round(n_bits);
TB        = 1000*inv(Bit_Rate);                     % [ps]
noffset   = round((offset/TB)*Sample_Rate);
if (noffset == 0),
    noffset = 1;
end
%
N                 = length(pulse);
pulse1(1:noffset+1) = pulse(N-noffset:N);
for ii = noffset + 2:N,
    pulse1(ii) = pulse(ii - noffset - 1);
end
%
Tolho = T(1:Sample_Rate*n_bits);
olho  = reshape(pulse1,Sample_Rate*n_bits,Sequence_Length/n_bits);
%
%     Rotina para calculo da abertura do diagrama de olho
%
	lv=length(olho(:,1));               % tamanho da janela do diagrama de olho
	point=lv/(2*n_bits);                % centro no olho do primeiro bit de n_bits
	range=8;                            % Num de pontos ao redor do centro do bit >VALOR Par<
    vvalue=olho(point-range/2:point+range/2,:);     % matriz centrada em point
	mvalue=mean(mean(vvalue));          % valor medio flutuante
%
% Calculo da abertura do diagrama de olho
%
	hi=[];
	lo=[];
	for zz= 1:range+1
		for jj=1:Sequence_Length
            if vvalue(zz,jj)>mvalue
                hi=[hi vvalue(zz,jj)];
            else
                lo=[lo vvalue(zz,jj)];
            end
		end
	end
    Lhigh    = mean(hi)-std(hi);   % media dos maximos - Desvio padrao
    Llow     = mean(lo)+std(lo);   % media dos minimos + Desvio padrao
    Abertura = Lhigh-Llow;         % abertura do diagrama de olho no centro com largura "range"
%
if (nargout == 0),
  maxolho = max(max(pulse));
  minolho = min(min(pulse));
  x       = linspace(minolho,maxolho,50);
  y       = hist(pulse,x);
  subplot('position',[0.10 0.09 0.15 0.85]),barh(x,y),grid,ylabel('Amplitude [u.a]')
  subplot('position',[0.33 0.09 0.60 0.85]),plot(Tolho,olho,'g'),grid,xlabel('Tempo [ps]'),...
  ,ylabel(sprintf('Abertura do diagrama de olho:  %8.3e [u.a]',Abertura))
end

