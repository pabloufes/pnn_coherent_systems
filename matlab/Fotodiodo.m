function I = Fotodiodo(CampoE,Fotodiodo_Input_File);
%
%    function I = Fotodiodo(CampoE,Fotodiodo_Input_File);
%
%    Implementa um fotodiodo PIN ou APD
%
%  Entrada:
%             CampoE : Campo elétrico  [V/m]
%
%  Saída:
%             I      : Corrente de saída [A]
%
%                                             by M.Segatto
%                                               14/08/2003
%
if (nargin <= 2),
  Fotodiodo_Input_File = 'Fotodiodo_Input_Data';
end
eval(Fotodiodo_Input_File);
%
switch tipo
%
  case 'PIN'
    Pin=abs(CampoE).^2;  
    I = R*Pin;
  case 'APD'
    error('Não implementado ainda...');
  otherwise
    error('tipos válidos são PIN e APD somente.')
end
