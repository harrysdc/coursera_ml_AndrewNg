function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters. 
%               You should set p to a vector of 0's and 1's
%

p = sigmoid(X*theta) >= 0.5

% Note...
% Vectorization somewhat serves as a for loop.
% The reason why use vector instead of for loop is that Octave/MATLAB is optimized for matrix computing.
% When dealing with large-scale dataset, the computing speed will play a significant role.
% =========================================================================


end
