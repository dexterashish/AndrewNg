function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C = 0.01;
sigma = 0.01;

error_min = 10000;
sigma_init = sigma;
for i=1:4
  %C
  %sigma
  for j=1:4
    %model= svmTrain(X, y, C, @(X, Xval) gaussianKernel(X, Xval, sigma));
    model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    predictions = svmPredict(model, Xval);
    error = mean(double(predictions ~= yval));
    if (error_min > error)
      error_min = error;
      C_min = C;
      sigma_min = sigma;
    endif
    sigma = sigma*10;
  endfor
  sigma = sigma_init;
  C = C*10;
endfor

C = 0.03;
sigma = 0.03;

for i=1:4
  %C
  %sigma
  for j=1:4
    %model= svmTrain(X, y, C, @(X, Xval) gaussianKernel(X, Xval, sigma));
    model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    predictions = svmPredict(model, Xval);
    error = mean(double(predictions ~= yval));
    if (error_min > error)
      error_min = error;
      C_min = C;
      sigma_min = sigma;
    endif
    sigma = sigma*10;
  endfor
  sigma = sigma_init;
  C = C*10;
endfor


C = C_min;
sigma = sigma_min;


% =========================================================================

end
