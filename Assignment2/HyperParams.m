function HyperParams = HyperParams(n_batch, n_epochs, m, lambda, eta_max, eta_min, n_s, n_cycle)
%method to set GD parameters
%
%Input:
%n_batch - size of minibatches
%eta     - learning rate
%n_epochs-number of runs through the training set
%alpha   - decay rate
%gamma   - momentum coefficient
%
%Output:
%WGDparams - object

if nargin == 8
    HyperParams.eta_max = eta_max;
    HyperParams.eta_min = eta_min;
    HyperParams.n_s = n_s;
    HyperParams.n_cycle = n_cycle;
end


HyperParams.n_batch = n_batch;
HyperParams.n_epochs = n_epochs;
HyperParams.m = m;
HyperParams.lambda = lambda;

end