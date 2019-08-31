function out = mapFeature(X1, X2)
% MAPFEATURE Feature mapping function to polynomial features
%
%   MAPFEATURE(X1, X2) maps the two input features
%   to quadratic features used in the regularization exercise.
%
%   Returns a new feature array with more features, comprising of 
%   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
%
%   Inputs X1, X2 must be the same size
%
out=[X1  X2 X1.^2 X2.^2 X1.*X2  (X1 .* X2).^2  X1.^3  X2.^3 (X1 .* X2).^3];
m=size(out,1);
out = [ones(m, 1) out];

end