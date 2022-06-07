function nextState = dyn_lstm(state, action, Par)
%
%   Signature   : nextState = dyn_lstm(state, action, Par)
% 
%   Inputs      : state -> State, consisting of concatenation of cell states
%                          and hidden states
%                 action -> Input vector of LSTM
%                 Params -> Struct containing all required weights and
%                           biases
% 
%   Outputs     : nextState -> Updated cell and hidden states
% 
%-------------------------------------------------------------------------%

% Get cell and hidden states
cellState = state(1:Par.nCellStates);
hiddenState = state(Par.nCellStates + 1:end-1);

% LSTM Layer - Input gate
it = logistic_function(Par.wi * action + Par.Ri * hiddenState + Par.bi);

% LSTM Layer - Forget gate
ft = logistic_function(Par.wf * action + Par.Rf * hiddenState + Par.bf);

% LSTM Layer - External input gate (cell candidate)
gt = tanh(Par.wg * action + Par.Rg * hiddenState + Par.bg);

% LSTM Layer - Update cell states 
cellStateNext = ft .* cellState + gt .* it;

% LSTM Layer - Output gate
ot = logistic_function(Par.wo * action + Par.Ro * hiddenState + Par.bo);

% LSTM Layer - Update hidden states 
hiddenStateNext = tanh(cellStateNext) .* ot;

% Layer 1 - Fully connected
ZFc1 = Par.WFc * hiddenStateNext + Par.bFc;

nextState = [cellStateNext; hiddenStateNext; ZFc1];



end



%% Auxiliary functions
function y = logistic_function(x)
y = 1 ./ (1 + exp(-x));
end
