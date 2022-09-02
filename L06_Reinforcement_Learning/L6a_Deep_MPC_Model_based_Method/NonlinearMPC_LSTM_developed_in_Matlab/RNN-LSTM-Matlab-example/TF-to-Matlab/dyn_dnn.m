function nextState = dyn_dnn(state, action, Par)
%
%   Signature   : nextState = dyn_dnn(state, action, Par)
%   Inputs      : state -> State, consisting of concatenation of cell states
%                          and hidden states
%                 action -> Input vector 
%                 Params -> Struct containing all required weights and
%                           biases
% 
%   Outputs     : nextState -> Updated states
% 
%-------------------------------------------------------------------------%



% Layer 1 - Fully connected
ZFc1 = ReLu_function(Par.WFc1 * action + Par.bFc1);

% Layer 1 - Fully connected
ZFc2 = ReLu_function(Par.WFc2 * ZFc1 + Par.bFc2);

% Layer 1 - Fully connected
ZFc3 = ReLu_function(Par.WFc3 * ZFc2 + Par.bFc3);

nextState = [ZFc3];



end



%% Auxiliary functions
function y = ReLu_function(x)
y = max(0,x);
% if x < 0
%    y = 0;
% else
%     y = x;
% end

end
