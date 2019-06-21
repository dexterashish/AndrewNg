function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
n = size(X, 2);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
%size(Theta1_grad)
%size(Theta2_grad)

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%%%%%%%%%% Part 1 %%%%%%%%%%%


%size(Theta1)
%size(Theta2)

mTheta1 = size(Theta1, 1);
mTheta2 = size(Theta2, 1);

%size(Theta1)
X = [ones(m, 1) X];
%n = size(X,2);

%Theta1 = [ones(size(1,n), 1); Theta1];
%Theta1 = [ones(size(Theta1)(2), 1)'; Theta1];
%size(Theta1)


%size(X)
%size(Theta1)
%size(Theta2)

%H2 = sigmoid(X*Theta1');
%H2 = [ones(m, 1) H2];
%H3 = sigmoid(H2*Theta2');


% first layer
H2 = sigmoid(X*Theta1'); % 5000x26
H2 = [ones(m, 1) H2];
H3 = sigmoid(H2*Theta2'); % 5000x10

logH       = zeros(m,num_labels);
log1MinusH = zeros(m,num_labels);

for i=1:m   %5000
  for k=1:num_labels   %10
    logH(i,k) = logH(i,k) + log(H3(i,k));
    log1MinusH(i,k) = log1MinusH(i,k) + log(1-H3(i,k));
  endfor
endfor

%size(y)
%size(logH)
%size(log1MinusH)

yy = zeros(num_labels,m); %10x5000
for i=1:num_labels
  for j=1:m
    if (y(j)==i)
      yy(i,j)=1;
    endif
  endfor
endfor

% logH       5000x10
% log1MinusH 5000x10
% yy         10x5000
JY0=0;
JY1=0;
for i=1:m
    JY0 = JY0 + logH(i,:)*yy(:,i);     %  1x10   X  10x1
    JY1 = JY1 + log1MinusH(i,:)*(1-yy(:,i));
endfor
%JY0 = sum(y.*logH);
%JY1 = sum((1-y).*log1MinusH);

J = -(JY0+JY1)/m;

%%% Regularization %%%
squareTheta1 = 0;
for i=1:hidden_layer_size
  for k=2:input_layer_size+1
    squareTheta1 = squareTheta1 + Theta1(i,k)*Theta1(i,k);
  endfor
endfor
squareTheta2 = 0;
for i=1:num_labels
  for k=2:hidden_layer_size+1
    squareTheta2 = squareTheta2 + Theta2(i,k)*Theta2(i,k);
  endfor
endfor
J = J + lambda*(squareTheta1+squareTheta2)/(2*m);

%%%%%%%%%% Part 2 %%%%%%%%%%%

a1 = X;  %5000x401
%a1(:,[1]) = []; %5000x400
%a2 = sigmoidGradient(X*Theta1'); %5000x26
a2 = H2; %5000x26
gz2 = sigmoidGradient(X*Theta1'); %5000x26
%a2(:,[1]) = []; %5000x25
a3 = H3; %5000x10
%size(X)       %5000x401
%size(Theta1)  %25x401
%size(Theta2)  %10%26

%Theta1(:,[1]) = []; %25x400
%Theta2(:,[1]) = []; %10x25

d3 = zeros(m,num_labels); %5000x10
d3 = a3 - yy';
%-------------------

%%%%d2 = (d3*Theta2).*a2; %5000x26
%size(a1)
%size(d2)
delta2 = zeros(hidden_layer_size,input_layer_size+1); %25x401
Theta2_1 = Theta2;
Theta2_1(:,[1]) = []; %10x25
%a2_1 = a2;
%a2_1(:,[1]) = []; %5000x25
%size(d3)
%size(Theta2_1)
%size(gz2)
d2 = (d3*Theta2_1).*gz2;;
delta2 = delta2 + d2'*a1;

#{
a1_1 = a1;
%a1_1([1],:) = []; % remove first row
d2_1 = d2;
d2_1(:,[1]) = []; %5000x25
delta2 = delta2 + d2_1'*a1_1;  % should be 25x400 ----      X 26x5000x5000x401
#}

Theta1_grad = delta2./m;

%------------------------
%size(d3)
%size(a2)
delta3 = zeros(num_labels,hidden_layer_size+1);
delta3 = delta3 + d3'*a2; % 10x26
Theta2_grad = delta3./m;


%size(delta2)
%size(delta3)
% -------------------------------------------------------------

%%% Regularization in gradient %%%

for i=1:hidden_layer_size
  for j=1:input_layer_size+1
    if (j==1)
      Theta1_grad(i,j) = Theta1_grad(i,j) + 0;
    else
      Theta1_grad(i,j) = Theta1_grad(i,j) + (lambda*Theta1(i,j))./(m);
    endif
  endfor
endfor

for i=1:num_labels
  for j=1:hidden_layer_size+1
    if (j==1)
      Theta2_grad(i,j) = Theta2_grad(i,j) + 0;
    else
      Theta2_grad(i,j) = Theta2_grad(i,j) + (lambda*Theta2(i,j))./(m);
    endif
  endfor
endfor
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
