#Import standard packages
import numpy as np
import random
import matplotlib.pyplot as plt
import math

# Global array to store the training error for each degree
train_errors_list = []
# Compute the predicted values for given weights and input features
def hypothesis(weights,input_x):
	# Weights = (n+1)x1, n - degree of polynomial
	# input_x = m x (n+1), m - no. of points
	# Output = m x 1 - predicted values for each example
	result = np.matmul(input_x,weights)
	return result

# Mean squared error - cose function
def mean_squared_error(predictions,labels):
	squared_error = (predictions-labels)**2
	mse = squared_error.mean()
	return mse/2

# Gradient Descent using MSE
def gradient_descent(train_x,train_y,alpha=0.05,iterations=None,epsilon=10e-7):
	num_points = int(train_x.shape[0])
	num_features = int(train_x.shape[1])
	train_y = train_y.reshape(num_points,1)
	weights = np.random.rand(num_features,1) - 0.5
	predictions = hypothesis(weights,train_x)
	# Run for given iterations
	if iterations != None:
		for it in range(iterations):
			for i in range(num_features):
				# Update weights according to the gradient
				weights[i] -= (alpha/num_points)*np.sum(np.multiply((predictions-train_y).reshape(num_points,1),train_x[:,i].reshape(num_points,1)))
			predictions = hypothesis(weights,train_x)
			error = mean_squared_error(predictions,train_y)
	# If no iterations are specified, run for upto convergence with difference 10e-7
	else:
		it = 0
		prev_error = 0
		while True:
			it += 1
			for i in range(num_features):
				# Update weights according to the gradient
				diff = (alpha/num_points)*np.sum(np.multiply((predictions-train_y).reshape(num_points,1),train_x[:,i].reshape(num_points,1)))
				weights[i] -= diff

			predictions = hypothesis(weights,train_x)
			error = mean_squared_error(predictions,train_y)
			if(prev_error> error and prev_error-error < epsilon):
				break
			prev_error = error
	print('Training error after '+str(it)+' iterations = '+str(error))
	# Append to the global array and return the weights trained
	train_errors_list.append(error)
	return weights

# Given individual values of x and the degree, compute the array [1,x,x^2,..] for each value
def generate_data(data,degree):
	num_points = data.shape[0]
	num_features = degree
	new_data = np.ones(num_points).reshape(num_points,1)
	for i in range(degree):
		last_row = new_data[:,i]
		new_row = np.multiply(last_row,data).reshape(num_points,1)
		new_data = np.concatenate((new_data,new_row),axis=1)
	return new_data

# Split the data randomly into size specified. Shuffle ensures split is random
def train_test_split(input_x,input_y,split_size = 0.8):
	N = input_x.shape[0]
	data = list(zip(input_x,input_y))
	random.shuffle(data)
	m = int(split_size*N)
	train_data = data[:m]
	test_data = data[m:]
	return train_data,test_data

def main():
	# Data set sizes
	N_vals = [100,1000,10000]
	test_errors_list = []
	global train_errors_list
	# Repeat part 1 and 2 for each training sets
	for N in N_vals:
		test_errors_list.clear()
		split_size = 0.8
		input_x = []
		for i in range(N):
			input_x.append(random.uniform(0,1))
		
		input_x = np.array(input_x)
		noise = np.random.normal(0,0.3,N)
		
		input_y = []
		for i in range(N):
			input_y.append(math.sin(2*math.pi*input_x[i]))
		input_y = np.add(input_y,noise)

		train_data,test_data = train_test_split(input_x,input_y,split_size)
		data = [train_data,test_data]
		np.save('data'+str(N)+'.npy',np.array(data))
		train_x_scalar,train_y = zip(*train_data)
		train_x_scalar = np.array(train_x_scalar)
		train_y = np.array(train_y)
		test_x_scalar,test_y = zip(*test_data)
		test_x_scalar = np.array(test_x_scalar)
		test_y = np.array(test_y)

		tr_pl = plt.scatter(train_x_scalar,train_y)
		te_pl = plt.scatter(test_x_scalar,test_y,c='g')
		plt.xlabel('X values')
		plt.ylabel('Y values')
		plt.title('Plot of generated datasets')
		plt.legend((tr_pl,te_pl),('Train data','Test data'),loc=0)
		plt.show()
		trained_weights = []
		for degree in range(1,10):
			print('degree = '+str(degree))
			train_x = generate_data(train_x_scalar,degree)
			test_x = generate_data(test_x_scalar,degree)
			weights = gradient_descent(train_x,train_y,iterations=10000)
			trained_weights.append(weights)
			predictions = hypothesis(weights,test_x)
			error = mean_squared_error(predictions,test_y)
			print('Mean squared error on test data = '+str(error))
			test_errors_list.append(error)
		train_errors = np.array(train_errors_list)
		test_errors = np.array(test_errors_list)
		error_list =[train_errors,test_errors]
		np.save('errors'+str(N)+'.npy',np.array(error_list))
	# Load the errors to plot 
	errors10 = np.load('../Problem 1/errors10.npy')
	train_errors10 = errors10[0]
	test_errors10 = errors10[1]
	errors100 = np.load('errors100.npy')
	train_errors100 = errors100[0]
	test_errors100 = errors100[1]
	errors1000 = np.load('errors1000.npy')
	train_errors1000 = errors1000[0]
	test_errors1000 = errors1000[1]
	errors10000 = np.load('errors10000.npy')
	train_errors10000 = errors10000[0]
	test_errors10000 = errors10000[1]
	N_vals = ['10','100','1000','10000']
	# Plot the error vs dataset size for each degree
	for i in range(1,10):
		tr_errors = [train_errors10[i-1],train_errors100[i-1],train_errors1000[i-1],train_errors10000[i-1]]
		plt.title('Error vs dataset size for degree '+str(i)+' polynomial fit')
		plt.xlabel('Dataset size')
		plt.ylabel('Error')
		tr_er_plot = plt.plot(N_vals,tr_errors)
		te_errors = [test_errors10[i-1],test_errors100[i-1],test_errors1000[i-1],test_errors10000[i-1]]
		te_er_plot = plt.plot(N_vals,te_errors)
		plt.legend([tr_er_plot[0],te_er_plot[0]],('Train Error','Test Error'),loc=0)
		plt.show()

if __name__ == '__main__':
	main()
