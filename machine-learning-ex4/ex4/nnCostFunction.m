function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
J=0;

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

                 
Theta1_grad=zeros(size(Theta1));%gradient for layer2
Theta2_grad=zeros(size(Theta2));%gradient for layer3


m = size(X, 1);
X = [ones(m, 1) X];%add bias to the  input

tempy=zeros(size(Theta2,1),size(X, 1));

for i=1:m,
  for k=1:num_labels,
    if k==y(i)
      tempy(k,i)=1;
    end
   end
end

#Use feed forward to calculate  predicted value for every traning examples

sec_layer_activation_input=Theta1 * X';
sec_layer_activation_output=sigmoid(sec_layer_activation_input);

%Add bias to sec layer outputs
sec_layer_activation_output = [ones(1,m) ;sec_layer_activation_output];

out_layer_activation_input=Theta2 * sec_layer_activation_output;

pred=sigmoid(out_layer_activation_input);

oneminuspred=1 - pred;
logpred=log(pred);
logminuspred=log(oneminuspred);


Theta1(:,1)=0;%exclude bias parameter
Theta2(:,1)=0;%exclude bias parameter

regularizedTerm=(lambda/(2*m)) *( sum((sum(Theta1 .^2))) + sum(sum((Theta2 .^2))));


J= (sum(sum((tempy .* logpred) + ((1 - tempy) .* logminuspred))) *(-1/m)) + regularizedTerm;

end
