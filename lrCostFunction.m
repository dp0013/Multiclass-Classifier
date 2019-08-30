function [J, grad] = lrCostFunction(theta_t, X_t, y_t, lambda_t)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y_t); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta_t));

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

% ================== FROM "costFunction" IN EXERCISE 2:

 hyp = 1 ./ (1 + e .^ -(X_t * theta_t));  % hypothesis

 J = -(1/m * ((-y_t' * log(hyp)) - ((1-y_t)' * log(1 - hyp))));  % cost function

 %grad = 1/m * ((hyp - y_t)' * X_t)';  % compute gradient (3x1 matrix (same dimensions as theta))


 
% ================== FROM  "costFunctionReg" IN EXERCISE 2:

% ========== REGULARIZED COST =============== 
% define additonal part for regularized cost function 
for i = 1,
  theta1 = theta_t(1,1); % only take first theta
  thetaSqr1 = theta1.^2;  % square theta1
  thetaSqrSum1 = sum(thetaSqr1);  % sum thetaSqr1 
  reg_cost1 = (lambda_t/(2*m)) * thetaSqrSum1;  % calculate regularized cost part
  
  J = -(1/m * ((y_t' * log(hyp)) + ((1-y_t)' * log(1 - hyp)))) + reg_cost1;  % regularized cost function
endfor

  
for i = 2:size(theta_t,1),
  theta2 = theta_t;
  theta2(1,:) = [];  % remove first theta
  thetaSqr2 = theta2.^2;  % square theta2
  thetaSqrSum2 = sum(thetaSqr2);  % sum thetaSqr2 
  reg_cost2 = (lambda_t/(2*m)) * thetaSqrSum2;  % calculate regularized cost part
  
  J = -(1/m * ((y_t' * log(hyp)) + ((1-y_t)' * log(1 - hyp)))) + reg_cost2;  % regularized cost function
endfor
 
 
% VECTORIZED, REGULARIZED COST
    %J = -(1/m * ((y_t' * log(hyp)) + ((1-y_t)' * log(1 - hyp)))) + ((lambda_t/(2*m)) * ((sum(theta_t(1,1).^2)) + (sum(theta_t(2:end).^2)))); 
%regCost1 = ((lambda_t/(2*m)) * (sum(theta_t(1,1).^2)));
%regCost2 = ((lambda_t/(2*m)) * (sum(theta_t(2:end).^2)));

%J = 2*J + regCost1 + regCost2; 

    
% ========== REGULARIZED GRADIENT =============== 

% VECTORIZED, REGULARIZED GRADIENT FUNCTION

grad(1,:) = (1/m * ((hyp - y_t)' * X_t(:,1))'); 
grad(2:end, :) = (1/m * ((hyp - y_t)' * X_t(:,(2:end)))') + ((lambda_t/m) * theta_t((2:end),1));


% ----------  OLD GRAD_REG CODE  --------------
% define additonal part for regularized gradient 
%for i = 1,
%  grad(i,:) = (1/m * ((hyp - y_t)' * X_t(:,i))');  % only for 
%endfor

%for i = 2:size(theta_t,1),  % for every theta starting at theta_1
%  grad(i,:) = (1/m * ((hyp - y_t)' * X_t(:,i))') + ((lambda_t/m) * theta_t(i,1));  
%endfor

% =============================================================

grad = grad(:);

end
