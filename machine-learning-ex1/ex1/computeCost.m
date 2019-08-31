function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m=size(X,1); %no of traning exapmles in.no fo rows in X
J=0;

predictions=X*theta;
error=predictions-y;
squarderror=error .^ 2;
sumofsquarederror=sum(squarderror);
J=sumofsquarederror/(2*m);%MSD cost /loss function,mean squaed distance


% =========================================================================

end
