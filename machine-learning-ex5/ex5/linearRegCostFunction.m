function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
%size(X) % mxn+1  mx2
%size(y) % mx1    mx1
%size(theta) % n+1x1  2x1

H = X*theta; %mx1  12x1

theta(1,:) = 0;

%J = sum((H-y).^2)/(2*m);
%J = J + lambda*(theta.^2 )/(2*m);
%%J = (sum((H-y).^2) + lambda*(theta(2,1).*theta(2,1)))/(2*m);
J1 = 0;
J2 = 0;
for i=1:m
  J1 = J1 + (H(i,1)-y(i,1)).^2;
endfor
t = length(theta);
for i=1:t
  J2 = J2 + lambda*(theta(i,1).^2);
endfor
J = (J1+J2)/(2*m);

%theta

%theta([1],:) = [];

%theta
%grad0 = X'*(H-y)/m; %2x1
grad = X'*(H-y)/m + lambda*theta/m;
%grad = [grad0 ; grad1];







% =========================================================================

grad = grad(:);

end
