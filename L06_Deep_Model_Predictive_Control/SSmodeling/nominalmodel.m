clc
clear all
close all

%% Identify nominal model

%% Loading data

load('ML_Data.mat');




mf = out.mf_main.Data(:);
SOI = out.SOI.Data(:);
VGT = out.Rack.Data(:);

Tout = out.T_out.Data(:);



%% 
y1 = Tout;

u1 = mf;
u2 = SOI;
u3 = VGT;
Z = iddata([y1],[u1, u2, u3],0.08,'Tstart',0);
t = Z.SamplingInstants;




subplot(411)
plot(t,Z.y(:,1)), ylabel('load [N.m]') 
title('Logged Input-Output Data')


subplot(412)
plot(t,Z.u(:,1)), ylabel('mf [mg]') 
title('Logged Input-Output Data')

subplot(413)
plot(t,Z.u(:,2)), ylabel('SOI [CAD]')

subplot(414)
plot(t,Z.u(:,3)), ylabel('VGT [\%]')
xlabel('Engine Cycle')



%% 
na = [1];
nb = [1 1 1];
nk = 0*[1 1 1];
sys = arx(Z,[na nb nk])


%%
A = -[sys.A(2)];
B = [sys.B{1, 1} sys.B{1, 2} sys.B{1, 3}];

  %%
  xk = [y1(1)];
  uk = [u1(1); u2(1);u3(1)];
  k = 1;
  for i = 1:3751
      
      xk1 = A*xk + B*[u1(k); u2(k);u3(k)];
      
      y_hat(i) = xk1(:);
      
      xk = xk1;
      k = k+1;
  end
  
  
%% Plotting
figure
  
subplot(411)
plot(y1)
hold on
plot(y_hat,'r--')


subplot(412)
plot(t,Z.u(:,1)), ylabel('mf [mg]') 
title('Logged Input-Output Data')

subplot(413)
plot(t,Z.u(:,2)), ylabel('SOI [CAD]')

subplot(414)
plot(t,Z.u(:,3)), ylabel('VGT [\%]')
xlabel('Engine Cycle')

