function [W, b] = initialize(d, K, m)

%method to initialize the weight metrices and bias
%
%Input:
% d dimensionaility of the input image
% m number of nodes in the hidden layer
% K number of unique labels
%
%Output:
%W        - initialized weight matrices
%b        - initialized bias

std1 = 1/sqrt(d);
std2 = 1/sqrt(m);
W1 = randn(m, d)*std1;
W2 = randn(K, m)*std2;

b1 = zeros(m,1);
b2 = zeros(K,1);

W = {W1, W2};
b = {b1, b2};

end