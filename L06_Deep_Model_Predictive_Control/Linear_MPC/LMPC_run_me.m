% clc
% clear all
% close all

%% sampling time
Ts = 0.08;
p = 10;
m = 1;

%% Nominal plant

A=[0.761122945545910];
B=[1.17010437190343,-0.833338507228864,7.61663544655574];

C = [1];
D = 0;

sys = ss(A,B,C,D,Ts)
%% linear MPC
nx = 1;
ny = 1;
nu = 3;

mpcobj = mpc(sys,Ts,p,m);

x0 = zeros(size(sys.B));

% SolverOptions.Algorithm = 'sqp'
%% Weights

% 
mpcobj.Weights = struct('MV',[0.1, 0.1, 0.1],'Output',[1]);
%% Constrain inputs
mpcobj.MV(1).Min = 10;
mpcobj.MV(1).Max = 80;

mpcobj.MV(2).Min = -10;
mpcobj.MV(2).Max = 3;

mpcobj.MV(3).Min = 0.8;
mpcobj.MV(3).Max = 1;
%% Constrain outputs
mpcobj.OV(1).Min = 0;
mpcobj.OV(1).Max = 600;



%% contraint softenning

mpcobj.Weights.ECR = 0.01;



%% load ref
load_ref = RandomArray(130, 310, 100, 50);

%% Simulink

open('Load_control_LMPC.slx')
sim('Load_control_LMPC.slx')


%% ploting outputs

t = out.t.Data(:);
mf = out.mf_main.Data(:);
SOI = out.SOI.Data(:);
VGT = out.VGT.Data(:);

Tout = out.T_out.Data(:);
ref = out.ref.Data(:);



  figure
  
subplot(411)
plot(t,Tout)
hold on
plot(t,ref,'r')
ylabel('load')
legend('Load','reference')

subplot(412)
plot(t,mf)
ylabel('mf [mg]')


subplot(413)
plot(t,SOI) 
ylabel('SOI [CAD]')

subplot(414)
plot(t,VGT) 
ylabel('VGT [\%]')
