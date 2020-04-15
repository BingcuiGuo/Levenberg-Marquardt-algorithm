function [jacobian] = jacobian_func_revised(input_layer,network, input_num, node_num, output_num )

%calculate the relevant elements
row_num = size(input_layer, 1);
weight_num = size(network,1);
jacobian = zeros(output_num*row_num, weight_num);

% calculate the weight_matrix
weight_matrix = zeros(input_num, node_num);
for n = 1:node_num
   for c =1:input_num
       weight_matrix(c,n)= network((c-1)*node_num+n);
   end
end

%%calcultae all the weights 
weights_all = ... 
    input_layer * weight_matrix;
bias = ...
    network(((input_num)*node_num+1):(input_num+1)*node_num)';
bias = ...
    repmat(bias, row_num,1);
weights_all = weights_all + bias; 
weight_all_after_tanh = tanh(weights_all);
weight_all_after_sech = (sech(weights_all)).^2;


for big_circulation = 1:output_num
    
%the first node_num columns (like 1:20) 
%get the output_weight
  
network_matrix = ...
    network(((input_num+big_circulation)*node_num+1):(input_num+1+big_circulation)*node_num)';
network_matrix_expand = ...
    repmat(network_matrix, row_num, 1);

    row_range = ...
        (big_circulation-1)*row_num+1 :(big_circulation*row_num);
   for i = 1:input_num
       col_range_1 = ...
           (i-1)*node_num+1:i*node_num;
       jacobian(row_range,col_range_1) = ...
           weight_all_after_sech.*network_matrix_expand; 
       input_current = input_layer(:,i);
       input_current = ...
           repmat(input_current, 1,node_num);
       jacobian(row_range,col_range_1)=...
           jacobian(row_range,col_range_1).*input_current*(-1); 
   end
    
    
%get the biases in the hidden layer (such as 21:30)

col_range_2 = ...
    (input_num*node_num+1):(input_num+1)*node_num; 
sech_matrix_weight = ...
    weight_all_after_sech .* network_matrix_expand; 
jacobian(row_range, col_range_2) = sech_matrix_weight*(-1);

%for output_node's weights like 31:70 
col_range_output = ...
    (input_num+1)*node_num+(big_circulation-1)*node_num+1:((input_num+1)*node_num+big_circulation*node_num);
jacobian(row_range, col_range_output) = (-1)*weight_all_after_tanh;


%for output biases (like 71-74) 
ones_matrix = ones(row_num,1);
jacobian(((big_circulation-1)*row_num+1):(big_circulation*row_num),weight_num-output_num+big_circulation)=...
    (-1)*ones_matrix; 

end



