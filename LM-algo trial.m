%generate the data
input = rand(1000,2);
network = rand(74,1); 
network = network/74;
target1 = input(:,1)-input(:,2); 
target = [tanh(target1), sech(target1).^sech(target1), sinh(target1).^2+tanh(target1), cosh(target1)];

%train data
loss_function_collection = Levenberg_algorithm_revised(input,network,target, 2, 10, 4,100, 0.9); 
loss_func_colletion_2 = Levenberg_algorithm_revised(input,network,target, 2, 10, 4,100, 1); 

%plot data
x = 1:100;
plot_1 = plot(x,loss_function_collection); 
plot_2 = plot(x,loss_func_colletion_2); 

