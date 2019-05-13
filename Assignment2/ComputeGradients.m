function [grad_W, grad_b] = ComputeGradients(X, Y, P, h, W, b,lambda)

%method to calculate the gradients
% d dimensionaility of the input image
% N number of samples
% K number of unique labels
%
%Input:
%X   - pixel data of images d*N
%Y   - one-hot representation K*N 
%W   - weight of the network {m*d, K*m}
%P   - probability of each label K*N
%h   - hidden layer activation K*N
%lambda - regularization term 1*1
%
%Output:
%grad_w - gradient matrix of the cost relative to W, K*d
%grad_b - gradient matrix of the cost relative to b, K*1


W1 = W{1};
W2 = W{2};

b1 = b{1};
b2 = b{2};

grad_W1 = zeros(size(W1));
grad_W2 = zeros(size(W2));
grad_b1 = zeros(size(b1));
grad_b2 = zeros(size(b2));

N = size(X,2);  

for i = 1 : N
    Pi = P(:, i);
    Yi = Y(:, i);
    Xi = X(:, i);
    hi = h(:, i);
    
    % Chain rule
    g = -Yi'*(diag(Pi) - Pi*Pi')/(Yi'*Pi);
    grad_b2 = grad_b2 + g';
    grad_W2 = grad_W2 + g'*hi';
    
    g = g*W2;
    ind = zeros(size(hi));
    ind(hi>0) = 1;
    g = g*diag(ind);
    
    grad_b1 = grad_b1 + g';
    grad_W1 = grad_W1 + g'*Xi';
    
end

grad_W1 = grad_W1/N+2*lambda*W1;
grad_W2 = grad_W2/N+2*lambda*W2;

grad_b1 = grad_b1/N;
grad_b2 = grad_b2/N;

grad_b = {grad_b1, grad_b2};

grad_W = {grad_W1, grad_W2}; 


end