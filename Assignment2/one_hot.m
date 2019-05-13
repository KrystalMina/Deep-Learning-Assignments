function onehot = one_hot(label)
%method to find the one-hot representation of the label
%
%Input:
%label - 1*N
%
%Output:
%onehot - K*N

K = length(unique(label));
N = length(label);
onehot = zeros(K,N);
for i = 1:N
    onehot(label(i),i)=1;
end

end