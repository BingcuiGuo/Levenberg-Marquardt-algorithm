function [output_layer] = simple_net(input_layer, network_layer, input_num, node_num, output_num)

%calculate the weight matrix
weight = network_layer([1:node_num])';
for c =1:input_num-1
    weight = [weight; network_layer([(c*node_num+1) : (c+1) * node_num])'];
end

input_layer = input_layer * weight;
row_num = size(input_layer, 1);
bias = network_layer(((input_num)*node_num+1):(input_num+1)*node_num)';
bias = repmat(bias, row_num,1);
input_layer = input_layer + bias; 
input_layer = tanh(input_layer);
neuron = [];

%calculate the neuron matrix
  for d = 1:output_num
       new_neuron = network_layer((input_num+d)* node_num+1: (input_num + d+1)*node_num);
       neuron = [neuron, new_neuron];
   end

%calculate the output 
output = input_layer * neuron; 
output_bias = network_layer(size(network_layer,1)- output_num+1: size(network_layer,1))';
output_bias = repmat(output_bias, row_num, 1);
output_layer = output+output_bias; 
end


 