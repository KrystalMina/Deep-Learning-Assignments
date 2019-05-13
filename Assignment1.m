clear
clc
addpath /Users/linghui/downloads/cifar-10-batches-mat

%% read in the training, validation and test datasets
[Xtr, Ytr, ytr] = LoadBatch('data_batch_1.mat');
[Xva, Yva, yva] = LoadBatch('data_batch_2.mat');
[Xte, Yte, yte] = LoadBatch('test_batch.mat');

%% initialization
mean = 0;
std = 0.01;
K = size(Ytr, 1); % bumber of labels, 10
d = size(Xtr, 1); % dimensionality of each image 
W = mean + randn(K, d)*std; % random initialization
b = mean + randn(K, 1)*std; % random initialization
lambda = 0;

%% analytical check
batch_size = 100;
[ngrad_b, ngrad_W] = ComputeGradsNumSlow(Xtr(:, 1 : batch_size), ...
    Ytr(:, 1 : batch_size), W, b, lambda, 1e-5);
P = EvaluateClassifier(Xtr(:, 1 : batch_size), W, b);
[grad_W, grad_b] = ComputeGradients(Xtr(:, 1 : batch_size), ...
    Ytr(:, 1 : batch_size), P, W, lambda);
gradcheck_b = max(abs(ngrad_b - grad_b)./max(0, abs(ngrad_b) + abs(grad_b)))
gradcheck_W = max(max(abs(ngrad_W - grad_W)./max(0, abs(ngrad_W) + abs(grad_W))))

%% perform the mini-batch gradient descent algorithm
GDparams = setGDparams(100, 0.01, 40);
Jtr = zeros(1, GDparams.n_epochs);
Jva = zeros(1, GDparams.n_epochs);
for i = 1 : GDparams.n_epochs
    Jtr(i) = ComputeCost(Xtr, Ytr, W, b, lambda);
    Jva(i) = ComputeCost(Xva, Yva, W, b, lambda);
    [W, b] = MiniBatchGD(Xtr, Ytr, GDparams, W, b, lambda);
end

%% display accuracy
acc_tr = ComputeAccuracy(Xtr, ytr, W, b);
disp(['training accuracy:' num2str(acc_tr*100) '%'])
acc_te = ComputeAccuracy(Xte, yte, W, b);
disp(['test accuracy:' num2str(acc_te*100) '%'])

%% plot cost score
figure
plot(1 : GDparams.n_epochs, Jtr, 'r')
hold on
plot(1 : GDparams.n_epochs, Jva, 'b')
hold off
xlabel('epoch');
ylabel('loss');
legend('training loss', 'validation loss');

% visualize weight matrix
for i = 1 : K
image = reshape(W(i, :), 32, 32, 3);
s_image{i} = (image - min(image(:))) / (max(image(:)) - min(image(:)));
s_image{i} = permute(s_image{i}, [2, 1, 3]);
end
figure
montage(s_image, 'size', [1, K])



