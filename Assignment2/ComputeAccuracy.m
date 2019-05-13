function acc = ComputeAccuracy(X, y, W, b)

%method to calculate the accuracy
%
%Input:
%X   - pixel data of images d*N
%Y   - one-hot representation K*N 
%y   - label, size 1*N
%W   - weight of the network {m*d, K*m}
%b   - bias of the network {m*1, K*1}
%
%Output:
%acc - accuracy 1*1

h = hiddenlayer(X, W, b);
P = EvaluateClassifier(h, W, b);
[~, k_star]= max(P);
acc = length(find(y==k_star))/length(y);

end