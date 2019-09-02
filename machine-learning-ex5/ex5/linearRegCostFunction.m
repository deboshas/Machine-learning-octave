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
%a1=[ones(size(X,1),2) X];%adding bias term i.e 1 for theta0

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
J= ((sum(((X * theta) - y).^2))+ (lambda .* ((theta(2)).^2))) ./ (2*m);%cost function

%error=((theta * a1') -y);
for i=1:size(theta,1),
  if i==1
    tempgrad=sum(((X * theta) -y) .* X(:,i)) .* (1/m);%excliding regularizatio for theta0
  else
    tempgrad=(sum(((X * theta) -y) .* X(:,i))+(lambda .*theta(i) )) .* (1/m);
  end
  grad(i,:)=tempgrad;
endfor
% =========================================================================
end
