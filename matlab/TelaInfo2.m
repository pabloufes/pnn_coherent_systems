%
%                TelaInfo2
%
%         Usado pela família de  programas SCM para gerar
%         a tela de informações do sistema.
%
%                                 by M.Segatto
%                                         2003
clc
disp('*****************************');
disp('*    Programas SCM          *');
disp('*****************************');
disp('Dados de entrada: ')
disp(' ')
disp(strcat('Modulação tipo:  ',mod_type))
disp(strcat('Taxa de bits (B):  ',num2str(B/1e9),' [Gbit/s]'))
disp(strcat('Freqüência da portadora elétrica:  ',num2str(Fc/1e9),' [GHz]'))
%disp(strcat('Máxima freqüência representada na simulação:  ',num2str(Fmax/1e9),' [GHz]'))
disp(strcat('Comprimento de onda:  ',num2str(lamb0),' [nm]'))
disp(strcat('Comprimento da fibra (L):  ',num2str(L),' [km]'))
disp(strcat('Dispersão da fibra (D):  ',num2str(D1550*1e6),'  [ps/(km.nm)]'))
disp(strcat('Atenuação da fibra (alfa):  ',num2str(alfadB),' [dB/km]'))
disp(' ')
disp(strcat('Número de bits:  ',num2str(NB)))
disp(strcat('Número de pontos por bit:  ',num2str(NPPB)))
if bb_eletrico == 1,
  str = 'Back-to-back';
  disp(strcat('Simulacao tipo : ',str))
else
  if bb_optico == 1,
    str = 'Back-to-back Óptico';
    disp(strcat('Simulacao tipo : ',str))
  else
    str = 'Óptico';
    disp(strcat('Simulacao tipo : ',str))
    if regime == 1,
      str1 = 'Usando o Split-Step Fourier';
      disp(strcat('Fibra : ',str1))
    elseif regime == 0,
      str1 = 'Usando Função de Transferência da fibra';
      disp(strcat('Fibra : ',str1))
    end
  end
end


disp('*****************************');
