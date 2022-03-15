clear all
close all
clc

%% Train DDPG Agent for Adaptive Cruise Control
% This example shows how to train a deep deterministic policy gradient (DDPG) 
% agent for adaptive cruise control (ACC) in Simulink®. 


%% Simulink Model
% The reinforcement learning environment for this example is the simple longitudinal 
% dynamics for an ego car and lead car. The training goal is to make the ego car 
% travel at a set velocity while maintaining a safe distance from lead car by 
% controlling longitudinal acceleration and braking. 


% Specify the initial position and velocity for the two vehicles.

x0_lead = 50;   % initial position for lead car (m)
v0_lead = 25;   % initial velocity for lead car (m/s)
x0_ego = 10;    % initial position for ego car (m)
v0_ego = 20;    % initial velocity for ego car (m/s)
%% Specify standstill default spacing (m), time gap (s) and driver-set velocity (m/s).

D_default = 10;
t_gap = 1.4;
v_set = 30;
%% 
% To simulate the physical limitations of the vehicle dynamics, constraint the 
% acceleration to the range |[–3,2]| m/s^2.

amin_ego = -3;
amax_ego = 2;
%% 
% Define the sample time |Ts| and simulation duration |Tf| in seconds.

Ts = 0.1;
Tf = 60;
%% 
% Open the model.

mdl = 'rlACCMdl';
open_system(mdl)
agentblk = [mdl '/RL Agent'];

%% Create Environment Interface
% Create a reinforcement learning environment interface for the model.
% Create the observation specification.
% defined lower and upper limits of observation

observationInfo = rlNumericSpec([3 1],'LowerLimit',-inf*ones(3,1),'UpperLimit',inf*ones(3,1));
observationInfo.Name = 'observations';
observationInfo.Description = 'information on velocity error and ego velocity';
%% 
% Create the action specification.
% Define lower and upper limits of actions

actionInfo = rlNumericSpec([1 1],'LowerLimit',-3,'UpperLimit',2);
actionInfo.Name = 'acceleration';
%% 
% Create the environment interface.

env = rlSimulinkEnv(mdl,agentblk,observationInfo,actionInfo);
%% 
% To define the initial condition for the position of the lead car, specify 
% an environment reset function using an anonymous function handle. The reset 
% function |localResetFcn|, which is defined at the end of the example, randomizes 
% the initial position of the lead car.

env.ResetFcn = @(in)localResetFcn(in);
%% 
% Fix the random generator seed for reproducibility.

rng('default')
%% Create DDPG agent
% A DDPG agent approximates the long-term reward given observations and actions 
% using a critic value function representation. To create the critic, first create 
% a deep neural network with two inputs, the state and action, and one output. 

L = 48; % number of neurons
statePath = [
    featureInputLayer(3,'Normalization','none','Name','observation')
    fullyConnectedLayer(L,'Name','fc1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(L,'Name','fc2')
    additionLayer(2,'Name','add')
    reluLayer('Name','relu2')
    fullyConnectedLayer(L,'Name','fc3')
    reluLayer('Name','relu3')
    fullyConnectedLayer(1,'Name','fc4')];

actionPath = [
    featureInputLayer(1,'Normalization','none','Name','action')
    fullyConnectedLayer(L, 'Name', 'fc5')];

criticNetwork = layerGraph(statePath);
criticNetwork = addLayers(criticNetwork, actionPath);
    
criticNetwork = connectLayers(criticNetwork,'fc5','add/in2');
%% 
% View the critic network configuration.

plot(criticNetwork)
%% 
% Specify options for the critic representation using <docid:rl_ref#mw_45ccf57d-64f0-4822-8000-3f0f44f2572e 
% |rlRepresentationOptions|>.

criticOptions = rlRepresentationOptions('LearnRate',1e-3,'GradientThreshold',1,'L2RegularizationFactor',1e-4);
%% 
% Create the critic representation using the specified neural network and options. 
% You must also specify the action and observation info for the critic, which 
% you obtain from the environment interface. For more information, see <docid:rl_ref#mw_6e9cf856-c679-4834-97e1-1349c4a21e43 
% |rlQValueRepresentation|>.

critic = rlQValueRepresentation(criticNetwork,observationInfo,actionInfo,...
    'Observation',{'observation'},'Action',{'action'},criticOptions);
%% 
% A DDPG agent decides which action to take given observations by using an actor 
% representation. To create the actor, first create a deep neural network with 
% one input, the observation, and one output, the action.


actorNetwork = [
    featureInputLayer(3,'Normalization','none','Name','observation')
    fullyConnectedLayer(L,'Name','fc1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(L,'Name','fc2')
    reluLayer('Name','relu2')
    fullyConnectedLayer(L,'Name','fc3')
    reluLayer('Name','relu3')
    fullyConnectedLayer(1,'Name','fc4')
    tanhLayer('Name','tanh1')
    scalingLayer('Name','ActorScaling1','Scale',2.5,'Bias',-0.5)];

actorOptions = rlRepresentationOptions('LearnRate',1e-4,'GradientThreshold',1,'L2RegularizationFactor',1e-4);
actor = rlDeterministicActorRepresentation(actorNetwork,observationInfo,actionInfo,...
    'Observation',{'observation'},'Action',{'ActorScaling1'},actorOptions);
%% 
% To create the DDPG agent, first specify the DDPG agent options using <docid:rl_ref#mw_5e9a4c5d-03d5-48d9-a85b-3c2d25fde43c 
% |rlDDPGAgentOptions|>.

agentOptions = rlDDPGAgentOptions(...
    'SampleTime',Ts,...
    'TargetSmoothFactor',1e-3,...
    'ExperienceBufferLength',1e6,...
    'DiscountFactor',0.99,...
    'MiniBatchSize',64);
agentOptions.NoiseOptions.StandardDeviation = 0.6;
agentOptions.NoiseOptions.StandardDeviationDecayRate = 1e-5;
%% 
% Then, create the DDPG agent using the specified actor representation, critic 

agent = rlDDPGAgent(actor,critic,agentOptions);
%% Train Agent

maxepisodes = 5000;
maxsteps = ceil(Tf/Ts);
trainingOpts = rlTrainingOptions(...
    'MaxEpisodes',maxepisodes,...
    'MaxStepsPerEpisode',maxsteps,...
    'Verbose',false,...
    'Plots','training-progress',...
    'StopTrainingCriteria','EpisodeReward',...
    'StopTrainingValue',260);
%% 
doTraining = false; %only showing the results

if doTraining    
    % Train the agent.
    trainingStats = train(agent,env,trainingOpts);
else
    % Load a pretrained agent for the example.
    load('SimulinkACCDDPG.mat','agent')       
end
%% Simulate DDPG Agent

x0_lead = 80;
sim(mdl)
