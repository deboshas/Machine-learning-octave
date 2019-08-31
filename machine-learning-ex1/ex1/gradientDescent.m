function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values

JHistory=zeros(num_iters,1);
m=size(X,1);%no of traning exapmles in.no fo rows in X

for i=1:num_iters
  predictions=X*theta;
  error=predictions-y;
  temp0=sum((error.* X(:,1)));%need to understand
  temp1=sum((error.* X(:,2)));%%need to understand
  theta0=theta(1) - (alpha/m) *(temp0);%simaltaneous update of parameters
  theta1=theta(2) - (alpha/m) *(temp1);%simaltaneous update of parameters
  theta=[theta0;theta1];
  cost=computeCost(X,y,theta);
  JHistory(i,:)=cost;
  disp(sprintf('Loss after : %0.1f iteration is  %0.4f', i,cost))
  
end



end
