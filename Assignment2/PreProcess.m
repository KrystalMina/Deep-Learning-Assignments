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
