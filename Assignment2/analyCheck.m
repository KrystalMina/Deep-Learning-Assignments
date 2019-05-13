function [gradcheck_W, gradcheck_b] = analyCheck(batch_size, X, Y, W, b, lambda, delta)

%method to analytically check the gradients of weights and bias
% d dimensionaility of the input image
% N number of samples
% K number of unique labels
%
%Input:
%X   - pixel data of images d*N
%Y   - one-hot representation K*N 
%W   - weight matrices {m*d, K*m}
%b   - bias {m*1, K*1}
%lambda - regularization term 1*1
%delta  -
%
%Output:
%gradcheck_W - 1*1
%gradcheck_b - 1*1

% analytical gradeints
h = hiddenlayer(X(:,1:batch_size), W, b);
P = EvaluateClassifier(h, W, b);
[grad_W, grad_b] = ComputeGradients(X(:,1:batch_size), ...
    Y(:,1:batch_size), P, h, W, b,lambda);

% numerical gradients
[ngrad_b, ngrad_W] = ComputeGradsNumSlow(X(:,1:batch_size), ...
    Y(:,1:batch_size), W, b, lambda, delta);


% relative error
epsilon = 1e-3;
gradcheck_b1 = sum(abs(ngrad_b{1} - grad_b{1}))/max(epsilon, sum(abs(ngrad_b{1}) + abs(grad_b{1})));
gradcheck_W1 = sum(sum(abs(ngrad_W{1} - grad_W{1})))/max(epsilon, sum(sum(abs(ngrad_W{1}) + abs(grad_W{1}))));
gradcheck_b2 = sum(abs(ngrad_b{2} - grad_b{2}))/max(epsilon, sum(abs(ngrad_b{2}) + abs(grad_b{2})));
gradcheck_W2 = sum(sum(abs(ngrad_W{2} - grad_W{2})))/max(epsilon, sum(sum(abs(ngrad_W{2}) + abs(grad_W{2}))));

gradcheck_W = [gradcheck_W1, gradcheck_W2];
gradcheck_b = [gradcheck_b1, gradcheck_b2];

end