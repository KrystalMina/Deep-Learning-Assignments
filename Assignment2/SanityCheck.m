function Ltr = SanityCheck(Xtr, Ytr, W, b, hyperParams, eta)

Ltr = zeros(1, hyperParams.n_epochs);
for i = 1 : hyperParams.n_epochs
    W1 = W{1};
    W2 = W{2};
    
    h = hiddenlayer(Xtr, W, b);
    P = EvaluateClassifier(h, W, b);
    N = size(Xtr,2); 
    Ltr(i) = sum(diag(-log(Ytr'*P)))/N;
    
    [W, b] = MiniBatch(Xtr, Ytr,hyperParams, W, b, eta);
end

end