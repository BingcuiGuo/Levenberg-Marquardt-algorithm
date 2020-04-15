function [error_vec] = error_vec_func(error_matrix, output_num, row_num)
error_vec = zeros(row_num*output_num, 1);
i = 1;
for k = 1:output_num
    error_vec(i:(i+size(error_matrix,1))) = error_matrix(:,k);
    i = i+size(error_matrix, 1); 
end

%error_vec = error_matrix(:); 