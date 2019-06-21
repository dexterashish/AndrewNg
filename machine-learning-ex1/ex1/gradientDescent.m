function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    B0 = 0;
    B1 = 0;
    H = X*theta;
    for i = 1:m
      B0 = B0 + (H(i)-y(i)).*X(i,1);
      B1 = B1 + (H(i)-y(i)).*X(i,2);
    end
    theta(1,1) = theta(1,1) - alpha*B0/m;
    theta(2,1) = theta(2,1) - alpha*B1/m;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
