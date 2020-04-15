function [total_loss] = loss_func(error_vec)
   %row_num = size(error_vec,1);
    %total_loss = 0;
    %for i = 1:row_num
       % total_loss = total_loss+error_vec(i)*error_vec(i);
    %end
    %total_loss = error_vec'*error_vec;
    total_loss = sum(error_vec.^2); 
end