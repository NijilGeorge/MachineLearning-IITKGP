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
def gradient_descent(train_x,train_y,alpha=0.05,iterations=None):
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
			if(prev_error> error and prev_error-error < 10e-7):
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
	N = 10		# Size of data set
	test_errors_list = []		# Array to store test errors
	global train_errors_list	
	train_errors_list.clear()
	split_size = 0.8	
	# Generate uniform values for x
	input_x = []
	for i in range(N):
		input_x.append(random.uniform(0,1))
	
	input_x = np.array(input_x)
	# Create random noise in guassian distribution with mean 0 and sd 0.3
	noise = np.random.normal(0,0.3,N)
	# Calculate y values
	input_y = []
	for i in range(N):
		input_y.append(math.sin(2*math.pi*input_x[i]))
	input_y = np.add(input_y,noise)
	# Split the data to 80-20 size
	train_data,test_data = train_test_split(input_x,input_y,split_size)
	data = [train_data,test_data]
	# Save the generated data
	np.save('data10.npy',np.array(data))
	# Separate out x and y from the data
	train_x_scalar,train_y = zip(*train_data)
	train_x_scalar = np.array(train_x_scalar)
	train_y = np.array(train_y)
	test_x_scalar,test_y = zip(*test_data)
	test_x_scalar = np.array(test_x_scalar)
	test_y = np.array(test_y)
	trained_weights = []
	# Loop over degrees from 1 to 9
	for degree in range(1,10):
		print('degree = '+str(degree))
		# Compute [1,x,x^2..] for each example in train and test data
		train_x = generate_data(train_x_scalar,degree)
		test_x = generate_data(test_x_scalar,degree)
		# Train the model
		weights = gradient_descent(train_x,train_y,iterations=10000)
		trained_weights.append(weights)
		# Compute test error
		predictions = hypothesis(weights,test_x)
		error = mean_squared_error(predictions,test_y)
		print('Mean squared error on test data = '+str(error))
		test_errors_list.append(error)
	train_errors = np.array(train_errors_list)
	test_errors = np.array(test_errors_list)
	# Save the weights trained
	np.save('weights10.npy',np.array(trained_weights))
	errors_list = [train_errors,test_errors]
	# Save the train and test errors obtained
	np.save('errors10.npy',np.array(errors_list))

if __name__ == '__main__':
	main()
