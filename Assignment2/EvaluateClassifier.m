function P = EvaluateClassifier(h, W, b)
%method to evaluate the network function
%
%Input:
%X   - pixel data of images d*N
%W   - weight of the network K*d
%b   - bias of the network K*1
%h   - hidden layer
%
%Output:
%P   - probability of each label K*N

W2 = cell2mat(W(2));
b2 = cell2mat(b(2));
b2 = repmat(b2, 1, size(h,2));
s = W2*h + b2;
denorm = repmat(sum(exp(s),1),size(W2,1),1);
P = exp(s)./ denorm;

end