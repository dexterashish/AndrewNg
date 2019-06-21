function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

n = length(X(1,:));

thetaT_X = X*theta;
H = sigmoid(thetaT_X);

logH = log(H);
log1minusH = log(1-H);
Jy0 = 0;
Jy1 = 0;
Jy1 = sum(y.*logH);
Jy0 = sum((1-y).*log1minusH);
%for i = 1:m
%  Jy1 = Jy1 + y(i,1).*logH(i,1);
%  Jy0 = Jy0 + (1-y(i,1)).*log1minusH(i,1);
%endfor



thetasquare = 0;
thetasquare = sum(theta.*theta)-(theta(1,1)*theta(1,1));
%for j = 2:n
%  thetasquare = thetasquare + theta(j,1)*theta(j,1);
%endfor

J = -(Jy1+Jy0)/m + thetasquare*lambda/(2*m);

%%%%%%%%%%% gradient %%%%%%%%%%%%%
B = zeros(n,1);
B = sum((H-y).*X);


%for i = 1:m       % no of rows in X
%  for j = 1:n     % no of columns in X
%    B(j,1) = B(j,1) + (H(i,1)-y(i,1)).*X(i,j);
%endfor
B = B';


grad = zeros(n,1);
grad = (B+lambda*theta)/m;
grad(1,1) = B(1,1)./m;

%for j = 1:n
%  if (j == 1)
%    grad(j,1) = B(j,1)./m;
%  else
%    grad(j,1) = B(j,1)./m + lambda*theta(j,1)/m;
%  endif
%endfor

% =============================================================

grad = grad(:);

end
