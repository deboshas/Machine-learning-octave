function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values

J=0;
grad=zeros(size(theta,1),1);
m=size(X,1);%no fo traning examples used 
n=size(X,2);%no fo features used including X0

pred=(sigmoid(X*theta));
oneminuspred=1 - pred;
logpred=log(pred);
logminuspred=log(oneminuspred);
error=pred-y;

J= sum((y .* logpred) + ((1 - y) .* logminuspred)) *(-1/m);

for j=1:n   
      grad(j)=sum((error .* X(:,j)));
  end
 
grad = (1/m) * grad;

% =============================================================

end
