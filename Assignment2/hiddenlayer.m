function h = hiddenlayer(X, W, b)

%method to calculater the hidden activation
%
%Input:
%X   - pixel data of images d*N
%W   - weight of the network K*d
%b   - bias of the network K*1
%
%Output:
%h   - hidden layer

W1 = W{1};
b1 = b{1};
b1 = repmat(b1, 1, size(X,2));
h = max(0, W1*X + b1);

end