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