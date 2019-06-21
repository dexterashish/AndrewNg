function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
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
    %       of the cost function (computeCostMulti) and gradient here.
    %

    B = zeros((size(X,2)),1);
    H = X*theta;
    for i = 1:m
      for j = 1:size(X,2)
        B(j) = B(j) + (H(i)-y(i)).*X(i,j);
      endfor
      %B0 = B0 + (H(i)-y(i)).*X(i,1);
      %B1 = B1 + (H(i)-y(i)).*X(i,2);
      %B2 = B2 + (H(i)-y(i)).*X(i,3);
    end
    for j = 1:size(X,2)
      theta(j,1) = theta(j,1) - alpha*B(j)/m;
    endfor
    %theta(1,1) = theta(1,1) - alpha*B0/m;
    %theta(2,1) = theta(2,1) - alpha*B1/m;
    %theta(3,1) = theta(3,1) - alpha*B2/m;
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);
end

end
