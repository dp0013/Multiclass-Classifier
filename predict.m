function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);  % 5000 x 1 matrix

% Add column of ones to X data matrix
X = [ones(m, 1) X];  % 5000 x 401 matrix

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

z2 = Theta1 * X';  % 25 x 5000 matrix
a2 = sigmoid(z2);  % 25 x 5000 matrix

a2 = [ones(1, columns(a2)); a2];  % add row of ones to a2 to make 26 x 5000 matrix
z3 = Theta2 * a2;  % produces a 10 x 5000 matrix  


probability = (1 ./ (1 + e .^ -(z3)))';  % probability matrix (5000 x 10)
[v p] = max(probability, [], 2);  % p for each row is the max probability






% =========================================================================


end
