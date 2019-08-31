function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
J=0;
grad=zeros(size(theta,1),1);
m=size(X,1);%no fo traning examples used 
n=size(X,2);%no fo features used including X0
temptheata=theta;
temptheata(1)=0;%excludinh theat0 i.e bias  parameter


pred=(sigmoid(X*theta));
oneminuspred=1 - pred;
logpred=log(pred);
logminuspred=log(oneminuspred);
error=pred-y;


J= (sum((y .* logpred) + ((1 - y) .* logminuspred)) *(-1/m))+((lambda/(2*m)) * sum(temptheata .^2)); %regularized cost function
grad = (1/m) .* (X' *(error)) + ((lambda/m) .* temptheata) ; %need to understand throughly look at the ex3.pdf,this si an regularized expression
%regularized gradient
end
