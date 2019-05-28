# Standard imports
import numpy as np
import matplotlib.pyplot as plt

# Compute the predicted values
def hypothesis(weights,input_x):
	# Weights = (n+1)x1, n - degree of polynomial
	# input_x = m x (n+1), m - no. of points
	# Output = m x 1 - predicted values for each example
	result = np.matmul(input_x,weights)
	return result

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

def main():
	# Load the dataset from previous problem
	data = np.load('../Problem 1/data10.npy')
	train_data = data[0]
	test_data = data[1]
	train_x_scalar,train_y = zip(*train_data)
	train_x_scalar = np.array(train_x_scalar)
	train_y = np.array(train_y)
	test_x_scalar,test_y = zip(*test_data)
	test_x_scalar = np.array(test_x_scalar)
	test_y = np.array(test_y)
	# Plot the datapoints
	tr_pl = plt.scatter(train_x_scalar,train_y)
	te_pl = plt.scatter(test_x_scalar,test_y,c='g')
	plt.xlabel('X values')
	plt.ylabel('Y values')
	plt.title('Plot of generated datasets')
	plt.legend((tr_pl,te_pl),('Train data','Test data'),loc=0)
	plt.show()
	# Load the weights trained in previous part
	trained_weights = np.load('../Problem 1/weights10.npy')
	for degree in range(1,10):
		# Plot the trained hypothesis function
		tr_pl = plt.scatter(train_x_scalar,train_y)
		te_pl = plt.scatter(test_x_scalar,test_y,c='g')
		plt.xlabel('X values')
		plt.ylabel('Y values')
		plt.title('Plot for degree '+str(degree)+' polynomial')
		weights = trained_weights[degree-1]
		points = np.linspace(0,1)
		point_features = generate_data(points,degree)
		values = hypothesis(weights,point_features)
		model_fit = plt.plot(points,values,'r')
		plt.legend((tr_pl,te_pl,model_fit[0]),('Train data','Test data','Model'),loc=0)
		plt.ylim((-2,2))
		plt.show()
	# Load the errors obtained
	errors_list = np.load('../Problem 1/errors10.npy')
	train_errors = errors_list[0]
	test_errors = errors_list[1]
	# Plot the error vs degree plot 
	plt.title('Train and test errors vs degree')
	plt.xlabel('Degree')
	plt.ylabel('Mean squared Error')
	degrees = list(range(1,10))
	tr_pl = plt.plot(degrees,train_errors)
	te_pl = plt.plot(degrees,test_errors)
	plt.legend([tr_pl[0],te_pl[0]],('Train Error','Test Error'),loc=0)
	plt.show()
if __name__ == '__main__':
	main()