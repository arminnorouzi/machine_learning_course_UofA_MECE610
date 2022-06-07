clc
clear all
close all

%% Nonlinear MPC
nx = 1;
ny = 1;
nu = 2;

nlobj = nlmpc(nx,ny,nu);

%% sampling time
Ts = 0.08;
p = 1;
m = 1;


%% Weights

nlobj.Weights.OutputVariables = [0.1]; % because range tout is not normalized in modeling
nlobj.Weights.ManipulatedVariables = [0,1];
nlobj.Weights.ManipulatedVariablesRate = [0.1, 0.1];

%% Sover options 

nlobj.Optimization.SolverOptions.Display = 'iter';
nlobj.Optimization.SolverOptions.OptimalityTolerance = 1e-10;
nlobj.Optimization.SolverOptions.StepTolerance = 1e-10;
nlobj.Optimization.SolverOptions.Algorithm = 'sqp';
nlobj.Optimization.SolverOptions.SpecifyObjectiveGradient = true;
nlobj.Optimization.SolverOptions.SpecifyConstraintGradient = true;
nlobj.Optimization.SolverOptions.MaxIterations = 5000;
%% horizon.

nlobj.Ts = Ts;
nlobj.PredictionHorizon = p; 
nlobj.ControlHorizon = m;

%% min_max of data


load('ML_Data.mat')

% creating enigne cycle vector
eng_cycle = linspace(1,size(Data(2:end,1), 1),size(Data(2:end,1), 1))';

% Start of injection [CAD]
SOI = Data(2:end,2); 

% Injection fuel amount [mg]
mf = Data(2:end,3);

% Output Tourque- Load [N.m]
Tout = Data(2:end,4);

% normalized = data-min(data) / max(data)-min(data)

[mf_n, min_mf, max_mf] = dataTrainStandardized(mf);
[SOI_n, min_SOI, max_SOI] = dataTrainStandardized(SOI);

%% Constrain inputs


nlobj.MV(1).Min = (-10-min_SOI)/(max_SOI-min_SOI);
nlobj.MV(1).Max = (3-min_SOI)/(max_SOI-min_SOI);

nlobj.MV(2).Min = (10-min_mf)/(max_mf-min_mf);
nlobj.MV(2).Max = (90-min_mf)/(max_mf-min_mf);


%% contraint softenning

nlobj.Weights.ECR = 0.001;


%% StateFcn 

% LoadModel = importKerasNetwork('DNNmodel_MPC.h5');
% % Define function- we can use it later for MPC
% % We have 128 hidden states and 128 cell states
% 
% % Fully connected layers
% WFc1 = LoadModel.Layers(2, 1).Weights;
% bFc1 = LoadModel.Layers(2, 1).Bias;
% 
% WFc2 = LoadModel.Layers(4, 1).Weights;
% bFc2 = LoadModel.Layers(4, 1).Bias;
% 
% WFc3 = LoadModel.Layers(6, 1).Weights;
% bFc3 = LoadModel.Layers(6, 1).Bias;

% Easiest way is transfering these paramter directly yo dnn function
% let's update dnn function



nlobj.Model.StateFcn = @(x,u) dyn_dnn(x,u);

nlobj.Model.IsContinuousTime = false;


%% validation
x0 = [214.683300000000];
u0 = [-3.8000;24.6000];

validateFcns(nlobj,x0,u0)

%% loadref
load_ref = RandomArray(130, 310, 100, 50);


%% Simulink

open('Load_control_NLMPC.slx')
sim('Load_control_NLMPC.slx')


