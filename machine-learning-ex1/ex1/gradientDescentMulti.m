function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values

J_history=zeros(num_iters,1);
m=size(y,1);%no of traning exapmles in.no fo rows in X
n=size(X,2);%no of features including X0
grad=zeros(n,1);
for i=1:num_iters
  predictions=X*theta;
  error=predictions-y;
  for j=1:n
    grad(j)=sum((error .* X(:,j)));%need to understand   
  end
  grad= grad ./m;
  theta=theta - ((alpha).*(grad));
  cost=computeCost(X,y,theta);
  J_history(i)=cost;
  disp(sprintf('Loss after : %0.1f iteration is  %0.4f', i,cost))
  
end



end