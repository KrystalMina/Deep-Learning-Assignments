function [W, b, Jtr, Jva, Ltr, Lva, acc_tr,acc_va]=MainCycle(Xtr, Ytr, ytr, Xva, Yva, yva, hyperParams)

%method of cyclicle learning rate training
%
%Input:
%Xtr - training data 
%Ytr - label of training data
%Xva - validation data 
%Yva - label of validation data
%
%Output:
%W   - weight matrices
%b   - bias
%Jtr - training loss
%Jva - validation loss

d = size(Xtr,1);
N = size(Xtr,2);
K = size(Ytr,1);
m = hyperParams.m;
n_epochs = hyperParams.n_epochs;
n_batch = hyperParams.n_batch;
n_cycle = hyperParams.n_cycle;
n_s = hyperParams.n_s;

[W, b] = initialize(d, K, m);

eta_min = 1e-5;
eta_max = 1e-1;
Eta = zeros(1,2*n_s*n_cycle);

Jtr = zeros(1, n_epochs);
Jva = zeros(1, n_epochs);
Ltr = zeros(1, n_epochs);
Lva = zeros(1, n_epochs);
acc_tr = zeros(1, n_epochs);
acc_va = zeros(1, n_epochs);

for l = 0 : n_cycle -1
    for k = 1 : n_s
        t = 2*l*n_s + k;
        Eta(t) = eta_min + k*(eta_max-eta_min)/n_s;
    end
    for k = 1 : n_s
        t = 2*l*n_s +n_s+ k;
        Eta(t) = eta_max - k*(eta_max-eta_min)/n_s;
    end
end

for i = 1 : n_epochs   
    Jtr(i) = ComputeCost(Xtr, Ytr, W, b, hyperParams.lambda);
    Jva(i) = ComputeCost(Xva, Yva, W, b, hyperParams.lambda);
    eta = Eta( (i-1)*N/n_batch+1 : i*N/n_batch );
    
    h = hiddenlayer(Xtr, W, b);
    P = EvaluateClassifier(h, W, b);
    N = size(Xtr,2); 
    Ltr(i) = sum(diag(-log(Ytr'*P)))/N;
    
    h = hiddenlayer(Xva, W, b);
    P = EvaluateClassifier(h, W, b);
    N = size(Xtr,2); 
    Lva(i) = sum(diag(-log(Yva'*P)))/N;
    
    [W, b] = MiniBatch(Xtr, Ytr, hyperParams, W, b, eta);
    acc_tr(i) =  ComputeAccuracy(Xtr, ytr, W, b);
    acc_va(i) =  ComputeAccuracy(Xva, yva, W, b);
end


% for i = 1 : n_epochs   
%     Jtr(i) = ComputeCost(Xtr, Ytr, W, b, hyperParams.lambda);
%     Jva(i) = ComputeCost(Xva, Yva, W, b, hyperParams.lambda);
%     eta = Eta( (i-1)*N/n_batch+1 : i*N/n_batch );
%     [W, b] = MiniBatchCycle(Xtr, Ytr, hyperParams, W, b, eta);
% end
%    index_start = mod(((i-1)*size_step +1), N);
%     index_end = index_start + size_step;
%     
%     ind1 = index_start:index_end;
%     
%     eta_start = (i-1)*size_step +1;
%     eta_end = i*size_step +1;
%     
%     ind2 = eta_start:eta_end;
%     
%     Jtr(i) = ComputeCost(Xtr(ind1), Ytr(ind1), W, b, hyperParams.lambda);
%     Jva(i) = ComputeCost(Xva(ind1), Yva(ind1), W, b, hyperParams.lambda);
%     
%     [W, b] = MiniBatchCycle(Xtr(ind1), Ytr(ind1), hyperParams, W, b, Eta(ind2));



end