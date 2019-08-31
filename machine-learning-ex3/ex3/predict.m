function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values

% Add bias  to the X data matrix i.e input  layer of neural network
m = size(X, 1);
X = [ones(m, 1) X];

num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);




% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
%compue  seconde  layer activation outputs

sec_layer_activation_input=Theta1 * X';
sec_layer_activation_output=sigmoid(sec_layer_activation_input);

%Add bias to sec layer outputs
sec_layer_activation_output = [ones(1,m) ;sec_layer_activation_output];

out_layer_activation_input=Theta2 * sec_layer_activation_output;

out_layer_activation_output=sigmoid(out_layer_activation_input);

[probability indices]=max(out_layer_activation_output);%need to understand this function
p=indices';




% =========================================================================


end
