function [loss_collection] = Levenberg_algorithm_revised(input_layer, network_layer,target, input_num, node_num, output_num,training_times, regularizer)

%define common factors that will be used 
loss_collection = zeros(training_times, 1);
row_num_network = size(network_layer,1);
identity_matrix = eye(row_num_network, row_num_network);

%training loop 
   for i = 1:training_times
      %calculate the forward output based on the simple net
      current_output = ...
           simple_net(input_layer, network_layer, input_num, node_num, output_num);
      %calculate the jacobian matrix for training
      jacobian_matrix = ...
           jacobian_func_revised(input_layer,network_layer, input_num,node_num, output_num);
   
      %calculate the loss function
      error_matrix = target-current_output; 
      error_vec = error_matrix(:); 
      loss_collection(i) = sum(error_vec.^2);  
   
      %local optimize 
      for m = 1:5
          %calculate the elements for training and train new network 
          identity_matrix_this_time = regularizer*identity_matrix; 
          new_network = ...
               network_layer - (jacobian_matrix'*jacobian_matrix+identity_matrix_this_time)\jacobian_matrix'*error_vec; 
          new_output = ...
               simple_net(input_layer, new_network, input_num, node_num, output_num);
          new_error_matrix = target-new_output;
          new_error_vec = ...
              new_error_matrix(:); 
          new_loss = sum(new_error_vec.^2); 

          % change the regularizer to update the network 
          if (new_loss <= loss_collection(i))
              regularizer = regularizer / 10; 
              break;
          else
              regularizer = regularizer * 10; 
          end
      end
      network_layer = new_network;
 
   end
end