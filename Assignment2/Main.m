clear
clc
addpath /Users/linghui/downloads/cifar-10-batches-mat


%% read in data: training, validation data and test data
[Xtr, Ytr, ytr] = LoadBatch('data_batch_1.mat');
[Xva, Yva, yva] = LoadBatch('data_batch_2.mat');
[Xte, Yte, yte] = LoadBatch('test_batch.mat');

% pre-processing: transform them to have zero mean, then normalize
Xtr = PreProcess(Xtr);
Xva = PreProcess(Xva);
Xte = PreProcess(Xte);


%% initialization
d = size(Xtr, 1);
N = size(Xtr, 2);
K = size(Ytr, 1);
m = 50;

[W, b] = initialize(d, K, m);

%% analytically check gradient
lambda = 0;
n_batch = 1;
delta = 1e-5;
[gradcheck_W, gradcheck_b] = analyCheck(n_batch, ...
    Xtr(:,1:20), Ytr(:,20), W, b, lambda, delta);

%% sanity check
n_epochs = 200;
lambda = 0;
eta = 0.01;
hyperParams = HyperParams(10, n_epochs,m, lambda);
Ltr = SanityCheck(Xtr(:,1:100), ...
    Ytr(:,1:100), W, b, hyperParams, eta);
figure
plot(Ltr)
xlabel('epochs')
ylabel('training loss')
legend('training loss')

%% cyclical learning rate: one cycle
clc
eta_min = 1e-5;
eta_max = 1e-1;
lambda = 0.01;
n_batch = 100;
n_epochs = 10;
n_s = 500;
n_cycle = 1;
hyperParams = HyperParams(n_batch, n_epochs, m, lambda, eta_max, eta_min, n_s, n_cycle);
[~, ~, Jtr, Jva, Ltr, Lva, acc_tr, acc_val]=MainCycle(Xtr,Ytr, ytr, Xva, Yva, yva, hyperParams);

% plot
figure
plot(Jtr)
hold on 
plot(Jva)
legend('training', 'validation')
xlabel('epochs')
ylabel('cost')
title('Cost using Cyclicle learning rate')

figure
plot(Ltr)
hold on 
plot(Lva)
legend('training', 'validation')
xlabel('epochs')
ylabel('loss')
title('Loss using Cyclicle learning rate')


figure
plot(acc_tr)
hold on 
plot(acc_val)
legend('training', 'validation')
xlabel('epochs')
ylabel('accuracy')
title('Accuracy using Cyclicle learning rate')

%% cyclical learning rate: three cycle
clc
eta_min = 1e-5;
eta_max = 1e-1;
lambda = 0.01;
n_batch = 100;
n_s = 800;
n_cycle = 3;
n_epochs = 2*n_s*n_cycle/(size(Xtr,2)/n_batch);
hyperParams = HyperParams(n_batch, n_epochs, m,...
    lambda, eta_max, eta_min, n_s, n_cycle);
[W, b, Jtr, Jva, Ltr, Lva, acc_tr, acc_val]=...
    MainCycle(Xtr,Ytr, ytr, Xva, Yva, yva, hyperParams);

% plot
figure
plot(Jtr)
hold on 
plot(Jva)
legend('training', 'validation')
xlabel('epochs')
ylabel('cost')
title('Cost using Cyclicle learning rate')

figure
plot(Ltr)
hold on 
plot(Lva)
legend('training', 'validation')
xlabel('epochs')
ylabel('loss')
title('Loss using Cyclicle learning rate')


figure
plot(acc_tr)
hold on 
plot(acc_val)
legend('training', 'validation')
xlabel('epochs')
ylabel('accuracy')
title('Accuracy using Cyclicle learning rate')

%% coarse search
clc

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

l_min = -5;
l_max = -1;

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

%% plot coarse search result
[acc_sort, acc_index] = sort(acc_va(:,n_epochs),'descend');
figure
semilogx(lambda_list(acc_index), acc_sort);

%% plot best 3 coarse search result
for i = 1:3
    h=figure('visible','off');
    plot(Jtr(acc_index(i),:))
    hold on
    plot(Jva(acc_index(i),:))
    legend('training', 'validation')
    xlabel('epochs')
    ylabel('cost')
    title(['Plot of cost with lambda = ', num2str(lambda_list(acc_index(i)))])
    saveas(h,['coarse',num2str(i),'cost'],'epsc')
    
    h=figure('visible','off');
    plot(Ltr(acc_index(i),:))
    hold on
    plot(Lva(acc_index(i),:))
    legend('training', 'validation')
    xlabel('epochs')
    ylabel('loss')
    title(['Plot of loss with lambda = ', num2str(lambda_list(acc_index(i)))])
    saveas(h,['coarse',num2str(i),'loss'],'epsc')

    h=figure('visible','off');
    plot(acc_tr(acc_index(i),:))
    hold on
    plot(acc_va(acc_index(i),:))
    legend('training', 'validation')
    xlabel('epochs')
    ylabel('accuracy')
    title(['Plot of accuracy with lambda = ', num2str(lambda_list(acc_index(i)))])
    saveas(h,['coarse',num2str(i),'acc'],'epsc')
    
end


%% store the coarse search result
CoarseSearch.lambda_list = lambda_list;
CoarseSearch.Jtr = Jtr;
CoarseSearch.Jva = Jva;
CoarseSearch.Ltr = Ltr;
CoarseSearch.Lva = Lva;
CoarseSearch.acc_tr = acc_tr;
CoarseSearch.acc_va = acc_va;

%% fine search
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

n_pairs = 5;

l_min = -5;
l_max = -3.5;

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

[~,lambda_index] = max(acc_va(:,n_epochs));
lambda_star = lambda_list(lambda_index);

%% plot fine search result
% [acc_sort, acc_index] = sort(acc_va(:,n_epochs),'descend');
% % figure
% % semilogx(lambda_list(acc_index), acc_sort);
% lambda_list(acc_index(1:3))
% acc_va(n_epochs, acc_index(1:3))
% acc_tr(n_epochs,acc_index(1:3))

%% test on the best fine search result

Xtr = [X1 X2 X3 X4 X5(:,1:9000)];
Ytr = [Y1 Y2 Y3 Y4 Y5(:,1:9000)];
ytr = [y1 y2 y3 y4 y5(:,1:9000)];
Xva = X5(:,9001:10000);
Yva = Y5(:,9001:10000);
yva = y5(:,9001:10000);

m = 50;
eta_min = 1e-5;
eta_max = 1e-1;
n_batch = 100;
n_cycle = 3;
n_s = 2*floor(size(Xtr,2)/n_batch);
n_epochs = 2*n_s*n_cycle/(size(Xtr,2)/n_batch);

Jtr = zeros(n_pairs, n_epochs);
Jva = zeros(n_pairs, n_epochs);
Ltr = zeros(n_pairs, n_epochs);
Lva = zeros(n_pairs, n_epochs);
acc_tr = zeros(n_pairs, n_epochs);
acc_va = zeros(n_pairs, n_epochs);
  

tic 

lambda = lambda_star;

hyperParams = HyperParams(n_batch, n_epochs, ...
    m, lambda, eta_max, eta_min, n_s, n_cycle);

[W, b, Jtr(i,:), Jva(i,:), Ltr(i,:), Lva(i,:), acc_tr(i,:), acc_va(i,:)]=...
    MainCycle(Xtr,Ytr, ytr, Xva, Yva, yva, hyperParams);

toc
