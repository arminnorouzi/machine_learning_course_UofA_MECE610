clc
clear all
close all



%% Train DQN Agent to Swing Up and Balance Pendulum
% This example shows how to train a deep Q-learning network (DQN) agent to swing 
% up and balance a pendulum modeled in Simulink®.

%% Pendulum Swing-up Model
% Open the model.

mdl = 'rlSimplePendulumModel';
open_system(mdl)

%% Create Environment Interface
% Create a predefined environment interface for the pendulum.

env = rlPredefinedEnv('SimplePendulumModel-Discrete')
%% 
% The interface has a discrete action space where the agent can apply one of 
% three possible torque values to the pendulum: –2, 0, or 2 N·m.
%% 
% To define the initial condition of the pendulum as hanging downward, specify 
% an environment reset function using an anonymous function handle. This reset 
% function sets the model workspace variable |theta0| to |pi|.

env.ResetFcn = @(in)setVariable(in,'theta0',pi,'Workspace',mdl);
%% 
% Get the observation and action specification information from the environment 

obsInfo = getObservationInfo(env)
actInfo = getActionInfo(env)
%% 
% Specify the simulation time |Tf| and the agent sample time |Ts| in seconds.

Ts = 0.05;
Tf = 20;
%% 
% Fix the random generator seed for reproducibility.

rng(0)
%% Create DQN Agent

dnn = [
    featureInputLayer(3,'Normalization','none','Name','state')
    fullyConnectedLayer(24,'Name','CriticStateFC1')
    reluLayer('Name','CriticRelu1')
    fullyConnectedLayer(48,'Name','CriticStateFC2')
    reluLayer('Name','CriticCommonRelu')
    fullyConnectedLayer(3,'Name','output')];
dnn = dlnetwork(dnn);
%% 
% View the critic network configuration.

figure
plot(layerGraph(dnn))
%% 
% Specify options for the critic optimizer using |rlOptimizerOptions|.

criticOpts = rlOptimizerOptions('LearnRate',0.001,'GradientThreshold',1);
%% 
% Create the critic representation using the specified deep neural network and 
% options. You must also specify observation and action info for the critic. 

critic = rlVectorQValueFunction(dnn,obsInfo,actInfo);
%% 
% To create the DQN agent, first specify the DQN agent options using rlDQNAgentOptions.

agentOptions = rlDQNAgentOptions(...
    'SampleTime',Ts,...
    'CriticOptimizerOptions',criticOpts,...
    'ExperienceBufferLength',3000,... 
    'UseDoubleDQN',false);
%% 
% Then, create the DQN agent using the specified critic representation and agent 
% options

agent = rlDQNAgent(critic,agentOptions);
%% Train Agent
% To train the agent, first specify the training options. For this example, 
% use the following options.
% For more information rlTrainingOptions.

trainingOptions = rlTrainingOptions(...
    'MaxEpisodes',1000,...
    'MaxStepsPerEpisode',500,...
    'ScoreAveragingWindowLength',5,...
    'Verbose',false,...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',-1100,...
    'SaveAgentCriteria','EpisodeReward',...
    'SaveAgentValue',-1100);
%% 
% Train the agent using the train function. Training this agent is a computationally intensive process 
% that takes several minutes to complete. To save time while running this example, 
% load a pretrained agent by setting doTraining to false. To train the agent 
% yourself, set doTraining to true.

doTraining = false;

if doTraining
    % Train the agent.
    trainingStats = train(agent,env,trainingOptions);
else
    % Load the pretrained agent for the example.
    load('SimulinkPendulumDQNMulti.mat','agent');
end

%% Simulate DQN Agent

simOptions = rlSimulationOptions('MaxSteps',500);
experience = sim(env,agent,simOptions);
