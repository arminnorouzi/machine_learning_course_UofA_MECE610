clear all
close all
clc

%% Loading data

load('ML_Data.mat')

% creating enigne cycle vector
eng_cycle = linspace(1,size(Data(1002:end,1), 1),size(Data(1002:end,1), 1))';

% Start of injection [CAD]
SOI = Data(1002:end,2); 

% Injection fuel amount [mg]
mf = Data(1002:end,3);

% Output Tourque- Load [N.m]
Tout = Data(1002:end,4);

% goal is modeling Load for given SOI and mf


%% Standardize Data 
% normalized = data-mean(data) / standard deviation(data)

[mf_n, mu_mf, sig_mf] = dataTrainStandardized(mf);
[SOI_n, mu_SOI, sig_SOI] = dataTrainStandardized(SOI);
[Tout_n, mu_Tout, sig_Tout] = dataTrainStandardized(Tout);


% first 100000 engine cycle as training
mf_n_tr = mf_n(1:100000);
SOI_n_tr = SOI_n(1:100000);
Tout_n_tr = Tout_n(1:100000);

% from 100000 engine cycle to 110000 as validation
mf_val_n = mf_n(100000:110000);
SOI_val_n = SOI_n(100000:110000);
Tout_val_n = Tout_n(100000:110000);

% from 110000 engine cycle to end as validation
mf_ts_n = mf_n(110000:end);
SOI_ts_n = SOI_n(110000:end);
Tout_ts_n = Tout_n(110000:end);


%% ploting dataset


figure(1)
set(gcf, 'Position', [100, 100, 1600, 700]);
set(gcf,'color','w');

subplot(311)
plot(eng_cycle(1:100000)', Tout(1:100000)', 'b','LineWidth',1.5)
hold on
plot(eng_cycle(100000:110000)', Tout(100000:110000)', 'r','LineWidth',1.5)
hold on
plot(eng_cycle(110000:end)', Tout(110000:end)', 'g','LineWidth',1.5)
legend('train data', 'Validation data' ,'test data','Interpreter','latex')
ylabel('Load [N.m]','Interpreter','latex')
xlabel('Engine Cycle','Interpreter','latex')
set(gca,'FontSize',12)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;

subplot(312)
plot(eng_cycle(1:100000)', mf(1:100000)', 'b','LineWidth',1.5)
hold on
plot(eng_cycle(100000:110000)', mf(100000:110000)', 'r','LineWidth',1.5)
hold on
plot(eng_cycle(110000:end)', mf(110000:end)', 'g','LineWidth',1.5)
ylabel('$m_{f,main}$ [mg]','Interpreter','latex')
xlabel('Engine Cycle','Interpreter','latex')
set(gca,'FontSize',12)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;

subplot(313)
plot(eng_cycle(1:100000)', SOI(1:100000)', 'b','LineWidth',1.5)
hold on
plot(eng_cycle(100000:110000)', SOI(100000:110000)', 'r','LineWidth',1.5)
hold on
plot(eng_cycle(110000:end)', SOI(110000:end)', 'g','LineWidth',1.5)
ylabel('SOI [CAD]','Interpreter','latex')
xlabel('Engine Cycle','Interpreter','latex')
set(gca,'FontSize',12)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;

%% Creating input-out for training

utrain = [mf_n_tr'; SOI_n_tr'];
ytrain = Tout_n_tr';


uval = [mf_val_n'; SOI_val_n'];
yval = Tout_val_n';

uts = [mf_ts_n'; SOI_ts_n'];
yts = Tout_ts_n';


%% Create and Train Network- Load

% number of outputs - we only trying to model load 
numResponses = 1;

% number of inputs - mf and SOI
featureDimension = 2;

% number of hidden units (neurons)
numHiddenUnits1 = 26;

% number number of Epoch
maxEpochs = 120;

% size of minibatch
miniBatchSize = 512;

Networklayers = [...
    sequenceInputLayer(featureDimension)
    fullyConnectedLayer(numHiddenUnits1)
    lstmLayer(numHiddenUnits1,'OutputMode','sequence')
    fullyConnectedLayer(numHiddenUnits1)
    fullyConnectedLayer(numResponses)
    regressionLayer];


options = trainingOptions('adam', ... %optimizer for training network
    'MaxEpochs',maxEpochs, ... % Maximum number of epochs
    'MiniBatchSize',miniBatchSize, ... %Size of mini-batch
    'GradientThreshold',1, ... % Gradient threshold, specified as the comma-separated pair consisting of 'GradientThreshold' and Inf or a positive scalar. If the gradient exceeds the value of GradientThreshold, then the gradient is clipped according to GradientThresholdMethod.
    'Shuffle','once', ... %Option for data shuffling
    'Plots','training-progress',... % Plots to display during network training
    'Verbose',1, ... % Indicator to display training progress information
    'VerboseFrequency',1,... % Frequency of verbose printing
    'LearnRateSchedule','piecewise',... %Option for dropping learning rate during training
    'LearnRateDropPeriod',150,... % Number of epochs for dropping the learning rate
    'LearnRateDropFactor',0.5,... %Factor for dropping the learning rate InitialLearnRate*(this value)
    'L2Regularization',0.1,... % Factor for L2 regularization- lambda in out notation
    'ValidationFrequency',1,... %Frequency of network validation
    'InitialLearnRate', 0.001,... %Initial learning rate
    'ValidationData',[{uval} {yval}]); %Data to use for validation during training

%% unchanged values  in options

% 'Momentum' — Contribution of previous step
% 0.9 (default) | scalar from 0 to 1


% 'GradientDecayFactor' — Decay rate of gradient moving average
% 0.9 (default) | nonnegative scalar less than 1

% 'SquaredGradientDecayFactor' — Decay rate of squared gradient moving average
% 0.9 | 0.999 | nonnegative scalar less than 1

% 'Epsilon' — Denominator offset
% 10-8 (default) | positive scalar


% 'ResetInputNormalization' — Option to reset input layer normalization
% true (default) | false

% 'BatchNormalizationStatistics' — Mode to evaluate statistics in batch normalization layers
% 'population' (default) | 'moving'


%% train/load
% true when you want to train a network
% Fasle when you want to load pretrained network
training_load = false;
if training_load == true
    [LoadModel, info_load] = trainNetwork(utrain,ytrain,Networklayers,options);
else
    load('load_model.mat')
    load('load_model_info.mat')
end

%% Predcition on whole data set
% lets check out network configuration
analyzeNetwork(LoadModel)

% predicition output
YPredt_load = predict(LoadModel,[mf_n'; SOI_n']);

%% Non standard prediciton
% scaling back Load value from staandardized value
YPredt_load_f = sig_Tout*YPredt_load + mu_Tout;
%% Plotting results


figure(2)
set(gcf, 'Position', [100, 100, 1600, 700]);
set(gcf,'color','w');

subplot(311)
plot(eng_cycle(1:100000), Tout(1:100000)', 'c','LineWidth',1.5)
hold on
plot(eng_cycle(100000:110000), Tout(100000:110000)', 'r','LineWidth',1.5)
plot(eng_cycle(110000:end), Tout(110000:end)', 'g','LineWidth',1.5)
plot(eng_cycle, YPredt_load_f, 'k--','LineWidth',1.5)
legend('train data', 'Validation data' ,'test data', 'prediciton','Interpreter','latex')
ylabel('Load [N.m]','Interpreter','latex')
xlabel('Engine Cycle','Interpreter','latex')
set(gca,'FontSize',12)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;

subplot(312)
plot(eng_cycle(1:100000)', mf(1:100000)', 'c','LineWidth',1.5)
hold on
plot(eng_cycle(100000:110000)', mf(100000:110000)', 'r','LineWidth',1.5)
hold on
plot(eng_cycle(110000:end)', mf(110000:end)', 'g','LineWidth',1.5)
ylabel('$m_{f,main}$ [mg]','Interpreter','latex')
xlabel('Engine Cycle','Interpreter','latex')
set(gca,'FontSize',12)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;

subplot(313)
plot(eng_cycle(1:100000)', SOI(1:100000)', 'c','LineWidth',1.5)
hold on
plot(eng_cycle(100000:110000)', SOI(100000:110000)', 'r','LineWidth',1.5)
hold on
plot(eng_cycle(110000:end)', SOI(110000:end)', 'g','LineWidth',1.5)
ylabel('SOI [CAD]','Interpreter','latex')
xlabel('Engine Cycle','Interpreter','latex')
set(gca,'FontSize',12)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;



%% ploting loss vs iteration

figure(3)
set(gcf, 'Position', [100, 100, 1600, 700]);
set(gcf,'color','w');

subplot(211)

plot(info_load.TrainingLoss, 'r','LineWidth',1.5)
hold on
plot(info_load.ValidationLoss, 'k--','LineWidth',1.5)
legend('Training','Validation','Interpreter','latex', 'Location','northeast','Orientation','horizontal')
ylabel('Loss','Interpreter','latex')
xlabel('# iteration','Interpreter','latex')
set(gca,'FontSize',12)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;

subplot(212)
plot(info_load.TrainingLoss, 'r','LineWidth',1.5)
hold on
plot(info_load.ValidationLoss, 'k--','LineWidth',1.5)
legend('Training','Validation','Interpreter','latex')
ylabel('Loss','Interpreter','latex')
xlabel('# iteration','Interpreter','latex')
set(gca,'FontSize',12)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;