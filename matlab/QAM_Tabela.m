function [I,Q,T,BQAM] = QAM_Tabela_Adolpho2(vet,mod_type,B);
%
%                function [I,Q,T,BQAM] = QAM_Tabela(vet,mod_type,B);
%
%          Dado um vetor de números binários e o tipo de modulação desejada,
%       gera os vetores I e Q da modulação baseado no código "Grey".
%
%      Entrada:
%               vet      : Vetor contendo a sequência de bits
%               B        : Taxa de bits [bit/s]
%               mod_type : Tipo de modulação a ser utilizada, pode ser
%
%                         'ASK'    -> ASK
%                         '4ASK'   -> ASK 4 níveis
%                         'PSK'    -> PSK
%                         'QPSK'   -> PSK 4 níveis
%                         '16PSK'  -> PSK 16 níveis
%                         '16QAM'  -> QAM 16 níveis
%                         '64QAM'  -> QAM 64 níveis
%      Saída:
%               I,Q        : Sinais I e Q para gerar a constelação
%               T          : Tempo do símbolo  [s]
%               BQAM       : Taxa de símbolos [símbolo/s]
%
%                                                       by M.Segatto
%                                                         12/08/2003
%                                         Implementação 16QAM: Adolpho Bonella
%
s22 = sqrt(2)/2;
NB  = length(vet);
%
switch mod_type
%
  case 'ASK',
%
     T    = 1/B;
     BQAM = B;
     I    = vet;
     Q    = zeros(1,length(vet));
%
  case '4ASK',
%
     T    = 2*(1/B);
     BQAM = 1/T;
     cont = 1;
    for ii = 1:2:NB-1,
      switch num2str([vet(ii) vet(ii+1)])
         case '0  0'
          I(cont) = 0;
          Q(cont) =0;
        case '0  1'
          I(cont) = 1;
          Q(cont) = 0;
        case '1  1'
          I(cont) = 2;
          Q(cont) =0;
        case '1  0'
          I(cont) = 3;
          Q(cont) =0;
      end
      cont = cont + 1;
    end
%
  case 'PSK',
%
    T    = 1/B;
    BQAM = 1/T;
    for ii = 1:NB,
      if vet(ii) == 0,
        I(ii) = 1;
        Q(ii) =0;
      elseif vet(ii) == 1,
        I(ii) = -1;
        Q(ii) = 0;
       end
     end
%
  case 'QPSK',
%
     T    = 2*(1/B);
     BQAM = 1/T;
     cont = 1;
    for ii = 1:2:NB-1,
      switch num2str([vet(ii) vet(ii+1)])
         case '0  0'
          I(cont) = 1;
          Q(cont) = 1;
        case '0  1'
          I(cont) = -1;
          Q(cont) = 1;
        case '1  1'
          I(cont) = -1;
          Q(cont) =-1;
        case '1  0'
          I(cont) = 1;
          Q(cont) =-1;
      end
      cont = cont + 1;
    end
%
  case '16PSK',
%
    T    = 4*(1/B);
    BQAM = 1/T;
    ang  = linspace(0,360-360/16,16);
    II   = cos(ang*pi/180);
    QQ   = sin(ang*pi/180);
%
    cont = 1;
    for ii = 1:4:NB-1,
      switch num2str([vet(ii) vet(ii+1) vet(ii+2) vet(ii+3)])
         case '0  0  0  0'
          I(cont) = II(1);
          Q(cont) = QQ(1);
         case '0  0  0  1'
          I(cont) = II(2);
          Q(cont) = QQ(2);
         case '0  0  1  1'
          I(cont) = II(3);
          Q(cont) = QQ(3);
         case '0  0  1  0'
          I(cont) = II(4);
          Q(cont) = QQ(4);
         case '0  1  1  0'
          I(cont) = II(5);
          Q(cont) = QQ(5);
         case '0  1  1  1'
          I(cont) = II(6);
          Q(cont) = QQ(6);
         case '0  1  0  1'
          I(cont) = II(7);
          Q(cont) = QQ(7);
         case '0  1  0  0'
          I(cont) = II(8);
          Q(cont) = QQ(8);
         case '1  1  0  0'
          I(cont) = II(9);
          Q(cont) = QQ(9);
         case '1  1  0  1'
          I(cont) = II(10);
          Q(cont) = QQ(10);
         case '1  1  1  1'
          I(cont) = II(11);
          Q(cont) = QQ(11);
         case '1  1  1  0'
          I(cont) = II(12);
          Q(cont) = QQ(12);
         case '1  0  1  0'
          I(cont) = II(13);
          Q(cont) = QQ(13);
         case '1  0  1  1'
          I(cont) = II(14);
          Q(cont) = QQ(14);
         case '1  0  0  1'
          I(cont) = II(15);
          Q(cont) = QQ(15);
         case '1  0  0  0'
          I(cont) = II(16);
          Q(cont) = QQ(16);
      end
      cont = cont + 1;
    end
%
case '16QAM',
    A    = 1;
    cont = 1;
    T    = 4*(1/B);
    BQAM = 1/T;

    for ii = 1:4:NB-1,
      switch num2str([vet(ii) vet(ii+1) vet(ii+2) vet(ii+3)])
         case '0  0  0  0'
          I(cont) = -3*A;
          Q(cont) = -3*A;
         case '0  0  0  1'
          I(cont) = -3*A;
          Q(cont) = -A  ;
         case '0  0  1  1'
          I(cont) = -3*A;
          Q(cont) = A;
         case '0  0  1  0'
          I(cont) = -3*A;
          Q(cont) =  3*A;
         case '0  1  1  0'
          I(cont) = -A;
          Q(cont) = 3*A;
         case '0  1  1  1'
          I(cont) = -A;
          Q(cont) =  A;
         case '0  1  0  1'
          I(cont) = -A;
          Q(cont) = -A;
         case '0  1  0  0'
          I(cont) = -A;
          Q(cont) = -3*A;
         case '1  1  0  0'
          I(cont) = A;
          Q(cont) = -3*A;
         case '1  1  0  1'
          I(cont) = A;
          Q(cont) = -A;
         case '1  1  1  1'
          I(cont) = A;
          Q(cont) = A;
         case '1  1  1  0'
          I(cont) = A;
          Q(cont) = 3*A;
         case '1  0  1  0'
          I(cont) = 3*A;
          Q(cont) = 3*A;
         case '1  0  1  1'
          I(cont) = 3*A;
          Q(cont) = A;
         case '1  0  0  1'
          I(cont) = 3*A;
          Q(cont) = -A;
         case '1  0  0  0'
          I(cont) =  3*A;
          Q(cont) = -3*A;
         end
      cont = cont + 1;
    end
%
  otherwise
    error('Modulação inválida, por favor verifique a modulação utilizada...');
%
end
