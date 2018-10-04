function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad


m = length(y); % # training examples

J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

ShiftedTheta = theta(2:size(theta));

J = 1/(2*m) * sum((X*theta - y).^2) + lambda/(2*m) * sum(ShiftedTheta.^2);

grad = (1/m) * X'*(X*theta - y)  + (lambda/m) * [0;ShiftedTheta];







% =========================================================================

grad = grad(:);

end
