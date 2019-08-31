function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g=zeros(size(z));%matrix to conatin values for every element

g=1 ./(1+exp(-1 * z)); %./ and /  are not same thing

end
