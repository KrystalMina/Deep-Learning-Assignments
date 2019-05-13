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

%% functions used

function [gradcheck_W, gradcheck_b] = analyCheck(batch_size, X, Y, W, b, lambda, delta)

%method to analytically check the gradients of weights and bias
% d dimensionaility of the input image
% N number of samples
% K number of unique labels
%
%Input:
%X   - pixel data of images d*N
%Y   - one-hot representation K*N 
%W   - weight matrices {m*d, K*m}
%b   - bias {m*1, K*1}
%lambda - regularization term 1*1
%delta  -
%
%Output:
%gradcheck_W - 1*1
%gradcheck_b - 1*1

% analytical gradeints
h = hiddenlayer(X(:,1:batch_size), W, b);
P = EvaluateClassifier(h, W, b);
[grad_W, grad_b] = ComputeGradients(X(:,1:batch_size), ...
    Y(:,1:batch_size), P, h, W, b,lambda);

% numerical gradients
[ngrad_b, ngrad_W] = ComputeGradsNumSlow(X(:,1:batch_size), ...
    Y(:,1:batch_size), W, b, lambda, delta);


% relative error
epsilon = 1e-3;
gradcheck_b1 = sum(abs(ngrad_b{1} - grad_b{1}))/max(epsilon, sum(abs(ngrad_b{1}) + abs(grad_b{1})));
gradcheck_W1 = sum(sum(abs(ngrad_W{1} - grad_W{1})))/max(epsilon, sum(sum(abs(ngrad_W{1}) + abs(grad_W{1}))));
gradcheck_b2 = sum(abs(ngrad_b{2} - grad_b{2}))/max(epsilon, sum(abs(ngrad_b{2}) + abs(grad_b{2})));
gradcheck_W2 = sum(sum(abs(ngrad_W{2} - grad_W{2})))/max(epsilon, sum(sum(abs(ngrad_W{2}) + abs(grad_W{2}))));

gradcheck_W = [gradcheck_W1, gradcheck_W2];
gradcheck_b = [gradcheck_b1, gradcheck_b2];

end

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

function [grad_W, grad_b] = ComputeGradients(X, Y, P, h, W, b,lambda)

%method to calculate the gradients
% d dimensionaility of the input image
% N number of samples
% K number of unique labels
%
%Input:
%X   - pixel data of images d*N
%Y   - one-hot representation K*N 
%W   - weight of the network {m*d, K*m}
%P   - probability of each label K*N
%h   - hidden layer activation K*N
%lambda - regularization term 1*1
%
%Output:
%grad_w - gradient matrix of the cost relative to W, K*d
%grad_b - gradient matrix of the cost relative to b, K*1


W1 = W{1};
W2 = W{2};

b1 = b{1};
b2 = b{2};

grad_W1 = zeros(size(W1));
grad_W2 = zeros(size(W2));
grad_b1 = zeros(size(b1));
grad_b2 = zeros(size(b2));

N = size(X,2);  

for i = 1 : N
    Pi = P(:, i);
    Yi = Y(:, i);
    Xi = X(:, i);
    hi = h(:, i);
    
    % Chain rule
    g = -Yi'*(diag(Pi) - Pi*Pi')/(Yi'*Pi);
    grad_b2 = grad_b2 + g';
    grad_W2 = grad_W2 + g'*hi';
    
    g = g*W2;
    ind = zeros(size(hi));
    ind(hi>0) = 1;
    g = g*diag(ind);
    
    grad_b1 = grad_b1 + g';
    grad_W1 = grad_W1 + g'*Xi';
    
end

grad_W1 = grad_W1/N+2*lambda*W1;
grad_W2 = grad_W2/N+2*lambda*W2;

grad_b1 = grad_b1/N;
grad_b2 = grad_b2/N;

grad_b = {grad_b1, grad_b2};

grad_W = {grad_W1, grad_W2}; 


end

function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

grad_W = cell(numel(W), 1);
grad_b = cell(numel(b), 1);

for j=1:length(b)
    grad_b{j} = zeros(size(b{j}));
    
    for i=1:length(b{j})
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) - h;
        c1 = ComputeCost(X, Y, W, b_try, lambda);
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W, b_try, lambda);
        
        grad_b{j}(i) = (c2-c1) / (2*h);
    end
end

for j=1:length(W)
    grad_W{j} = zeros(size(W{j}));
    
    for i=1:numel(W{j})
        
        W_try = W;
        W_try{j}(i) = W_try{j}(i) - h;
        c1 = ComputeCost(X, Y, W_try, b, lambda);
    
        W_try = W;
        W_try{j}(i) = W_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W_try, b, lambda);
    
        grad_W{j}(i) = (c2-c1) / (2*h);
    end
end
end

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

function HyperParams = HyperParams(n_batch, n_epochs, m, lambda, eta_max, eta_min, n_s, n_cycle)
%method to set GD parameters
%
%Input:
%n_batch - size of minibatches
%eta     - learning rate
%n_epochs-number of runs through the training set
%alpha   - decay rate
%gamma   - momentum coefficient
%
%Output:
%WGDparams - object

if nargin == 8
    HyperParams.eta_max = eta_max;
    HyperParams.eta_min = eta_min;
    HyperParams.n_s = n_s;
    HyperParams.n_cycle = n_cycle;
end


HyperParams.n_batch = n_batch;
HyperParams.n_epochs = n_epochs;
HyperParams.m = m;
HyperParams.lambda = lambda;

end

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

function [X, Y, y] = LoadBatch(filename)
%method to read in the data from the batch file 
%and returns the image and label data in seperate files
%
%Input:
%filename - string
%
%Output:
%X        - image pixel data with size d*N, type double/single, entries 0/1
%Y        - one-hot representation of the label with size K*N
%y        - label, size 1*N


readdata = load(filename);
X = double(readdata.data')/255;
y = double(readdata.labels')+1;
Y = one_hot(y);

end

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
end


function [Wstar, bstar] = MiniBatch(X, Y, hyperParams, W, b, eta)

%method to calculate the gradients of minibatch method
%
%Input:
%X   - pixel data of images d*N
%Y   - one-hot representation K*N 
%W   - weight of the network {m*d, K*m}
%b   - bias of the network {m*1, K*1}
%lambda - regularization term 1*1
%GDparams - object containing 
%             n_batch - size of minibatches
%             eta     - learning rate
%             n_epochs-number of runs through the training set
%             lambda  - regularization term 1*1
%
%Output:
%Wstar - gradient matrix of the cost relative to W, {K*m, m*d}
%bstar - gradient matrix of the cost relative to b, K*1

n_batch = hyperParams.n_batch;
%eta = hyperParams.eta;


N = size(X, 2);
if size(eta,2)==1
    eta = repmat(eta, 1, N/n_batch);
end

vecW = {zeros(size(W{1})),zeros(size(W{2}))};
vecb = {zeros(size(b{1})),zeros(size(b{2}))};



for j = 1 : N/n_batch
    j_start = (j-1)*n_batch + 1;
    j_end = j*n_batch;
    Xbatch = X(:, j_start:j_end);
    Ybatch = Y(:, j_start:j_end);
    
    % compute gradients for each mini-batch
    h = hiddenlayer(Xbatch, W, b);
    P = EvaluateClassifier(h, W, b);
    [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, h, W, b, hyperParams.lambda);
   
    vecW{1} = eta(j)*grad_W{1};
    vecW{2} = eta(j)*grad_W{2};
    vecb{1} = eta(j)*grad_b{1};
    vecb{2} = eta(j)*grad_b{2};
    
    % update weights and bias
    W{1} = W{1} - vecW{1};
    W{2} = W{2} - vecW{2};
    b{1} = b{1} - vecb{1};
    b{2} = b{2} - vecb{2};
end

% optimized W and b
Wstar = W;
bstar = b;

end

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

function outX = PreProcess(X)


%method of pre-processing
%
%Input:
%X     - read in data
%
%Output:
%outX  - pre-processed data
%

mean_X = mean(X, 2); 
std_X = std(X, 0, 2);
X = X - repmat(mean_X, [1, size(X, 2)]);
outX = X ./ repmat(std_X, [1, size(X, 2)]);

end

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

