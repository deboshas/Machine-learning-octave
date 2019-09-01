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
a1 = [ones(m,1) X];%add bias to the  input
tempy=zeros(size(Theta2,1),size(a1, 1));

for i=1:m,
  for k=1:num_labels,
    if k==y(i)
      tempy(k,i)=1;
    end
   end
end

#Use feed forward to calculate  predicted value for every traning examples


z2=Theta1 * a1';%2nd layer input
a2=sigmoid(z2);%2nd layer activation output
a2=[ones(1,m) ;a2];%adding bias to 2 nd layer output
z3=Theta2 * a2;%3 rd layer activation input
a3=sigmoid(z3);%3 rd layer activation output

%pred=sigmoid(out_layer_activation_input);

%oneminuspred=1 - pred;
%logpred=log(pred);
%logminuspred=log(oneminuspred);

regTheta1=Theta1;
regTheta2=Theta2;
%regTheta1 =  Theta1(:,2:end);
%regTheta2 =  Theta2(:,2:end);
regTheta1(:,1)=0; %exclude bias parameter%
regTheta2(:,1)=0; %exclude bias parameter

regularizedTerm=(lambda/(2*m)) *( sum((sum(regTheta1 .^2))) + sum(sum((regTheta2 .^2))));


J= (sum(sum((tempy .* log(a3)) + ((1 - tempy) .* log(1-a3)))) *(-1/m)) + regularizedTerm;



#Gradient calculation  usign back propagation
for t=1:m,
  a1t=a1(t,:);%fetch the data row
  z2t=z2(:,t);%fetch t thcolmn
  a2t=a2(:,t);%fetch t the column
  z3t=z3(:,t);%fetch t the colun
  a3t=a3(:,t);%fetch  t th column
  err3t=a3t - tempy(:,t);%fetch t the  column
  err2t=(Theta2(:,2:end))'* err3t .* sigmoidGradient(z2t);
  Theta1_grad=Theta1_grad+err2t * a1t;
  Theta2_grad=Theta2_grad+err3t * a2t';
endfor

Theta1_grad=(Theta1_grad * (1/m)) + ((lambda/m) .* regTheta1); %Gradient with respect to theat1 with regularization
Theta2_grad=(Theta2_grad * (1/m)) + ((lambda/m) .* regTheta2); %Gradient with respect to theat2 with regularization


%Unroll gradients
grad=[Theta1_grad(:);Theta2_grad(:)];


end
