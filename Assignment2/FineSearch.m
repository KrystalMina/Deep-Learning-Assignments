clear
clc
addpath /Users/linghui/downloads/cifar-10-batches-mat


[X1, Y1, y1] = LoadBatch('data_batch_1.mat');
[X2, Y2, y2] = LoadBatch('data_batch_2.mat');
[X3, Y3, y3] = LoadBatch('data_batch_3.mat');
[X4, Y4, y4] = LoadBatch('data_batch_4.mat');
[X5, Y5, y5] = LoadBatch('data_batch_5.mat');

X1 = PreProcess(X1);
X2 = PreProcess(X2);
X3 = PreProcess(X3);
X4 = PreProcess(X4);
X5 = PreProcess(X5);

Xtr = [X1 X2 X3 X4 X5(:,1:5000)];
Ytr = [Y1 Y2 Y3 Y4 Y5(:,1:5000)];
ytr = [y1 y2 y3 y4 y5(:,1:5000)];
Xva = X5(:,5001:10000);
Yva = Y5(:,5001:10000);
yva = y5(:,5001:10000);

m = 50;
eta_min = 1e-5;
eta_max = 1e-1;
n_batch = 100;
n_cycle = 2;
n_s = 2*floor(size(Xtr,2)/n_batch);
n_epochs = 2*n_s*n_cycle/(size(Xtr,2)/n_batch);

n_pairs = 10;

l_min = -4;
l_max = -2;

acc = zeros(1, n_pairs);
lambda_list = zeros(1, n_pairs);

Jtr = zeros(n_pairs, n_epochs);
Jva = zeros(n_pairs, n_epochs);
Ltr = zeros(n_pairs, n_epochs);
Lva = zeros(n_pairs, n_epochs);
acc_tr = zeros(n_pairs, n_epochs);
acc_va = zeros(n_pairs, n_epochs);

% random search
for i = 1:n_pairs
    tic
    
    l = l_min + (l_max - l_min)*rand(1, 1); 
    lambda = 10^l;
    lambda_list(i) = lambda;
    
    hyperParams = HyperParams(n_batch, n_epochs, ...
        m, lambda, eta_max, eta_min, n_s, n_cycle);
    
    [W, b, Jtr(i,:), Jva(i,:), Ltr(i,:), Lva(i,:), acc_tr(i,:), acc_va(i,:)]=...
        MainCycle(Xtr,Ytr, ytr, Xva, Yva, yva, hyperParams);
    
    toc
end

%% store the coarse search result
CoarseSearch.lambda_list = lambda_list;
CoarseSearch.Jtr = Jtr;
CoarseSearch.Jva = Jva;
CoarseSearch.Ltr = Ltr;
CoarseSearch.Lva = Lva;
CoarseSearch.acc_tr = acc_tr;
CoarseSearch.acc_va = acc_va;