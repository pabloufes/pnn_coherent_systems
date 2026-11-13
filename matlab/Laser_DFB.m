function [Elaser,Plaser] = Laser_DFB(Imod,Ith,Ibias,Ifator);


%
%
k      = 0.9;      % Fator de linearizaćão da curva do laser
%
Ilaser   = Ifator*(ent_laser-min(ent_laser))/(max(ent_laser)-min(ent_laser));
%
Plaser         =  Laser_DFB_Simplificado(Ilaser,Ibias,Ith,k);  % Potênica óptica do laser
Elaser         = sqrt(Plaser);                                 % Campo elétrico
