function J = ComputeCost(X, Y, W, b, lambda)

%method to calculate the cost
%
%Input:
%X   - pixel data of images d*N
%Y   - one-hot representation K*N 
%W   - weight of the network {m*d, K*m}
%b   - bias of the network {m*1, K*1}
%lambda - regularization term 1*1
%
%Output:
%J   - sum of the loss of hte neural network predictions 1*1

W1 = W{1};
W2 = W{2};

h = hiddenlayer(X, W, b);

P = EvaluateClassifier(h, W, b);

N = size(X,2);

J1 = sum(diag(-log(Y'*P)))/N;
J2 = lambda*(sum(sum(W1.^2))+sum(sum(W2.^2)));
J = J1+J2;

end