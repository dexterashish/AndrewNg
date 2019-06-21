function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.
%data = load('ex1data2.txt');
%X = data(:, 1:2);
%y = data(:, 3);
m = length(X(:,1));
% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

mu1 = mean(X(:,1));
mu2 = mean(X(:,2));
mu = [mu1 mu2];
sigma1 = std(X(:,1));
sigma2 = std(X(:,2));
sigma = [sigma1 sigma2];

for i = 1:size(X, 2)
  mu(1,i) = mean(X(:,i));
  sigma(1,i) = std(X(:,i));
endfor

dim = size(X, 2);
for d = 1:dim
  for i = 1:m,
    X_norm(i,d) = (X(i,d) - mu(1,d))/sigma(1,d);
  end
end

% ============================================================

end
