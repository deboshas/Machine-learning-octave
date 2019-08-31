function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

J=0;
grad=zeros(size(theta,1),1);
m=size(X,1);%no fo traning examples used 
n=size(X,2);%no fo features used including X0
temptheta=theta;
temptheta(1)=0;


pred=(sigmoid(X*theta));
oneminuspred=1 - pred;
logpred=log(pred);
logminuspred=log(oneminuspred);
error=pred-y;

J= (sum((y .* logpred) + ((1 - y) .* logminuspred)) *(-1/m))+((lambda/(2*m)) * sum(temptheta .^2)); %exclude theat0 for regularization
%J = (-1 / m) * sum(y.*log(sigmoid(X * theta)) + (1 - y).*log(1 - sigmoid(X * theta))) + (lambda / (2 * m))*sum(temptheta.^2);

for j=1:n
    if j==1,
      grad(j)=sum((error .* X(:,j))) * (1/m);
    else
     grad(j)=(sum((error .* X(:,j))) * (1/m))+((lambda * theta(j)) * (1/m));%exclude theat0 for regularization parameter
     end 
  end



% =============================================================

end

