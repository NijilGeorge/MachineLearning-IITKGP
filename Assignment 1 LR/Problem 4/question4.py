#Import standard packages
import numpy as np
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

# Mean absolute error cost function
def mean_absolute_error(predictions,labels):
	absolute_error = np.absolute(predictions-labels)
	mae = absolute_error.mean()
	return mae/2

# Mean fourth power error cost function
def mean_fourth_pow_error(predictions,labels):
	fourth_pow_error = (predictions-labels)**4
	mfe = fourth_pow_error.mean()
	return mfe/2
# RMSE value
def root_mean_squared_error(predictions,labels):
	squared_error = (predictions-labels)**2
	mse = squared_error.mean()
	return math.sqrt(mse/2)

# Gradient descent using MAE cost function
def gradient_descent_mae(train_x,train_y,alpha=0.05,iterations=None):
	num_points = int(train_x.shape[0])
	num_features = int(train_x.shape[1])
	train_y = train_y.reshape(num_points,1)
	weights = np.random.rand(num_features,1) - 0.5
	predictions = hypothesis(weights,train_x)
	if iterations != None:
		for it in range(iterations):
			pred_error = predictions - train_y
			pred_error[pred_error<0] = -1
			pred_error[pred_error>0] = 1
			for i in range(num_features):
				diff = (alpha/(2*num_points))* np.sum(np.multiply(train_x[:,i].reshape(num_points,1),pred_error))
				weights[i] = weights[i] - diff
			predictions = hypothesis(weights,train_x)
			error = mean_absolute_error(predictions,train_y)
	else:
		it = 0
		prev_error = 0
		while True:
			it += 1
			pred_error = predictions - train_y
			pred_error[pred_error<0] = -1
			pred_error[pred_error>0] = 1
			for i in range(num_features):
				diff = (alpha/(2*num_points))* np.sum(np.multiply(train_x[:,i].reshape(num_points,1),pred_error))
				weights[i] -= diff
			predictions = hypothesis(weights,train_x)
			error = mean_absolute_error(predictions,train_y)
			if(prev_error> error and prev_error-error < 10e-7):
				break
			prev_error = error
	print('Training error after '+str(it)+' iterations = '+str(error))
	train_errors_list.append(error)
	return weights

# Gradient descent using FPE cost function
def gradient_descent_fourth_pow(train_x,train_y,alpha=0.05,iterations=None):
	num_points = int(train_x.shape[0])
	num_features = int(train_x.shape[1])
	train_y = train_y.reshape(num_points,1)
	weights = np.random.rand(num_features,1) - 0.5
	predictions = hypothesis(weights,train_x)
	if iterations != None:
		for it in range(iterations):
			for i in range(num_features):
				diff = (2*alpha/num_points)*np.sum(np.multiply(((predictions-train_y).reshape(num_points,1))**3,train_x[:,i].reshape(num_points,1)))
				weights[i] = weights[i] - diff
			predictions = hypothesis(weights,train_x)
			error = mean_fourth_pow_error(predictions,train_y)
	else:
		it = 0
		prev_error = 0
		while True:
			it += 1
			for i in range(num_features):
				diff = (2*alpha/num_points)*np.sum(np.multiply(((predictions-train_y).reshape(num_points,1))**3,train_x[:,i].reshape(num_points,1)))
				weights[i] -= diff
			predictions = hypothesis(weights,train_x)
			error = mean_fourth_pow_error(predictions,train_y)
			if(prev_error> error and prev_error-error < 10e-7):
				break
			prev_error = error
	print('Training error after '+str(it)+' iterations = '+str(error))
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
	global train_errors_list
	train_errors_list.clear()
	test_errors_list = []
	# Load the 10,000 point dataset created in part 3
	data = np.load('../Problem 3/data1000.npy')
	train_data = data[0]
	test_data = data[1]
	train_x_scalar,train_y = zip(*train_data)
	train_x_scalar = np.array(train_x_scalar)
	train_y = np.array(train_y)
	test_x_scalar,test_y = zip(*test_data)
	test_x_scalar = np.array(test_x_scalar)
	test_y = np.array(test_y)
	trained_weights = []
	# For each degree, train the model on the data using MAE
	for degree in range(1,10):
		print('degree = '+str(degree))
		train_x = generate_data(train_x_scalar,degree)
		test_x = generate_data(test_x_scalar,degree)
		weights = gradient_descent_mae(train_x,train_y,iterations=10000)
		trained_weights.append(weights)
		predictions = hypothesis(weights,test_x)
		error = mean_absolute_error(predictions,test_y)
		print('Mean absolute error on test data = '+str(error))
		test_errors_list.append(error)
	# Plot the error vs degree curve to know best fit
	train_errors = np.array(train_errors_list)
	test_errors = np.array(test_errors_list)
	degrees_list = list(range(1,10))
	tr_pl = plt.plot(degrees_list,train_errors)
	te_pl = plt.plot(degrees_list,test_errors)
	plt.title('Error vs degree for MAE')
	plt.xlabel('Degree')
	plt.ylabel('Error')
	plt.legend([tr_pl[0],te_pl[0]],('Train data','Test data'),loc=0)	
	plt.show()

	# Do the same now using FPE cost function
	train_errors_list.clear()
	test_errors_list.clear()
	trained_weights.clear()
	for degree in range(1,10):
		print('degree = '+str(degree))
		train_x = generate_data(train_x_scalar,degree)
		test_x = generate_data(test_x_scalar,degree)
		weights = gradient_descent_fourth_pow(train_x,train_y,iterations=10000)
		trained_weights.append(weights)
		predictions = hypothesis(weights,test_x)
		error = mean_fourth_pow_error(predictions,test_y)
		print('Mean fourth power error on test data = '+str(error))
		test_errors_list.append(error)
	# Plot the error vs degree to know best fit
	train_errors = np.array(train_errors_list)
	test_errors = np.array(test_errors_list)
	degrees_list = list(range(1,10))
	tr_pl = plt.plot(degrees_list,train_errors)
	te_pl = plt.plot(degrees_list,test_errors)
	plt.title('Error vs degree for Fourth power')
	plt.xlabel('Degree')
	plt.ylabel('Error')
	plt.legend([tr_pl[0],te_pl[0]],('Train data','Test data'),loc=0)	
	plt.show()

	# The learning we are testing for
	alpha_values = [0.025,0.05,0.1,0.2,0.5]
	# Degree of 3 is found best for both MAE and FPE
	degree = 3
	train_x = generate_data(train_x_scalar,degree)
	test_x = generate_data(test_x_scalar,degree)
	train_errors_list.clear()
	test_errors_list.clear()
	# Train the model for different values of alpha
	for alpha in alpha_values:
		print('Alpha = '+str(alpha))
		weights = gradient_descent_mae(train_x,train_y,alpha=alpha)
		predictions = hypothesis(weights,test_x)
		error = root_mean_squared_error(predictions,test_y)
		print('RMSE for learning rate '+str(alpha)+' = '+str(error))
		test_errors_list.append(error)	
	# Plot the train and test errors vs learning rate for MAE
	train_errors_mae = np.array(train_errors_list)
	test_errors_mae = np.array(test_errors_list)
	p1 = plt.plot(alpha_values,train_errors_mae)
	p2 = plt.plot(alpha_values,test_errors_mae)
	plt.title('RMSE vs learning rate for MAE')
	plt.xlabel('Learning Rate')
	plt.ylabel('RMSE')
	plt.legend([p1[0],p2[0]],('Train Error','Test Error'),loc=0)
	plt.show()
	# Repeat the same for FPE. Here also deg 3 is optimal
	degree = 3
	train_x = generate_data(train_x_scalar,degree)
	test_x = generate_data(test_x_scalar,degree)
	train_errors_list.clear()
	test_errors_list.clear()
	for alpha in alpha_values:
		print('Alpha = '+str(alpha))
		weights = gradient_descent_fourth_pow(train_x,train_y,alpha=alpha)
		predictions = hypothesis(weights,test_x)
		error = root_mean_squared_error(predictions,test_y)
		print('RMSE for learning rate '+str(alpha)+' = '+str(error))
		test_errors_list.append(error)	
	train_errors_fp = np.array(train_errors_list)
	test_errors_fp = np.array(test_errors_list)
	p1 = plt.plot(alpha_values,train_errors_fp)
	p2 = plt.plot(alpha_values,test_errors_fp)
	plt.title('RMSE vs learning rate for FPE')
	plt.xlabel('Learning Rate')
	plt.ylabel('RMSE')
	plt.legend([p1[0],p2[0]],('Train Error','Test Error'),loc=0)
	plt.show()

if __name__ == '__main__':
	main()

