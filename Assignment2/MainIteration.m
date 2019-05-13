function [W, b, Jtr, Jva]=MainIteration(Xtr, Ytr, Xva, Yva, hyperParams)

%method of random search
%
%Input:
%Xtr - training data 
%Ytr - label of training data
%Xva - validation data 
%Yva - label of validation data
%
%Output:
%W   - weight matrices
%b   - bias
%Jtr - training loss
%Jva - validation loss

d = size(Xtr,1);
K = size(Ytr,1);
m = hyperParams.m;
[W, b] = initialize(d, K, m);

Jtr = zeros(1, hyperParams.n_epochs);
Jva = zeros(1, hyperParams.n_epochs);

for i = 1 : hyperParams.n_epochs
        
    Jtr(i) = ComputeCost(Xtr, Ytr, W, b, hyperParams.lambda);
    Jva(i) = ComputeCost(Xva, Yva, W, b, hyperParams.lambda);
    
    [W, b] = MiniBatch(Xtr, Ytr, hyperParams, W, b);

end

end