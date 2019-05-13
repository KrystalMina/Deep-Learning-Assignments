function [Wstar, bstar] = MiniBatchCycle(X, Y, hyperParams, W, b, eta)

%method to calculate the gradients of minibatch method
%
%Input:
%X   - pixel data of images d*N
%Y   - one-hot representation K*N 
%W   - weight of the network {m*d, K*m}
%b   - bias of the network {m*1, K*1}
%lambda - regularization term 1*1
%GDparams - object containing 
%             n_batch - size of minibatches
%             eta     - learning rate
%             n_epochs-number of runs through the training set
%             lambda  - regularization term 1*1
%
%Output:
%Wstar - gradient matrix of the cost relative to W, {K*m, m*d}
%bstar - gradient matrix of the cost relative to b, K*1

n_batch = hyperParams.n_batch;
%eta = hyperParams.eta;

N = size(X, 2);

vecW = {zeros(size(W{1})),zeros(size(W{2}))};
vecb = {zeros(size(b{1})),zeros(size(b{2}))};

for j = 1 : N/n_batch
    
    j_start = (j-1)*n_batch + 1;
    j_end = j*n_batch;
    Xbatch = X(:, j_start:j_end);
    Ybatch = Y(:, j_start:j_end);
    
    % compute gradients for each mini-batch
    h = hiddenlayer(Xbatch, W, b);
    P = EvaluateClassifier(h, W, b);
    [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, h, W, b, hyperParams.lambda);
    
    % momentum
    vecW{1} =  eta(j)*grad_W{1};
    vecW{2} =  eta(j)*grad_W{2};
    vecb{1} =  eta(j)*grad_b{1};
    vecb{2} =  eta(j)*grad_b{2};
    
    % update weights and bias
    W{1} = W{1} - vecW{1};
    W{2} = W{2} - vecW{2};
    b{1} = b{1} - vecb{1};
    b{2} = b{2} - vecb{2};
end

% optimized W and b
Wstar = W;
bstar = b;

end