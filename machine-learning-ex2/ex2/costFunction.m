function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
% Note: grad should have the same dimensions as theta
%

%%%%%%%%%%% cost %%%%%%%%%%%%%%%%

n = length(X(1,:));

thetaT_X = X*theta;
H = sigmoid(thetaT_X);

logH = log(H);
log1minusH = log(1-H);
Jy0 = 0;
Jy1 = 0;
for i = 1:m
  Jy1 = Jy1 + y(i,1).*logH(i,1);
  Jy0 = Jy0 + (1-y(i,1)).*log1minusH(i,1);
endfor

J = -(Jy1+Jy0)/m;

%%%%%%%%%%% gradient %%%%%%%%%%%%%
B = zeros(n,1);
for i = 1:m       % no of rows in X
  for j = 1:n     % no of columns in X
    B(j,1) = B(j,1) + (H(i,1)-y(i,1)).*X(i,j);
endfor

grad = zeros(n,1);
for j = 1:n
  grad(j,1) = B(j,1)./m;
endfor

% =============================================================

end
