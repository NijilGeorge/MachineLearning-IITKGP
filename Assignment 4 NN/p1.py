import numpy as np
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

# Function used in relu_derivative
def max2(a,b):
    if(a>b):
        return 1
    else: 
        return 0

# Compute the relu derivative of given numpy array
def relu_derivative(x):
    return np.vectorize(max2)(x,0)

# Compute the relu activation output for the given numpy array
def relu(x):
    return np.vectorize(max)(x,0)

# Compute the sigmoid activation output for the given numpy array 
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# Preprocessing function 
def preprocess():
    X = []
    y = []
    vocabulary = set({})
    ps =PorterStemmer()
    # Stop words taken from nltk library
    stop_words = set(stopwords.words('english'))
    # Read the lines
    with open('Assignment_4_data.txt','r') as f:
        for line in f:
            # Split into tokens
            line = line.replace('\t',' ')
            line = line.replace('\n',' ')
            line = line.replace('.',' ')
            line = line.replace(',',' ')
            line = line.replace(':',' ')
            line = line.replace('-',' ')
            tokens = line.split()
            # Read and append the label
            if(tokens[0]=='ham'):
                y.append(0)
            elif(tokens[0]=='spam'):
                y.append(1)
            tokens = tokens[1:]
            stemmed_tokens = []
            # Stem the words and add to training array. Remove stop words
            for word in tokens:
                if(word not in stop_words):
                    stem_word = ps.stem(word)
                    vocabulary.add(stem_word)
                    stemmed_tokens.append(stem_word)
            X.append(stemmed_tokens)
    vocabulary = sorted(list(vocabulary))
    # Convert into one hot vectors
    one_hot_X = np.zeros((len(X),len(vocabulary)),dtype=np.int8)
    for i in range(len(X)):
        for j in range(len(X[i])):
            word_ind = vocabulary.index(X[i][j])
            one_hot_X[i][word_ind] = 1
    np.save('features.npy',one_hot_X)
    np.save('labels.npy',np.array(y))

# Load the preprocessed data and split into train and test sets of size 80-20
def dataloader():
    X = np.load('features.npy')
    y = np.load('labels.npy')
    train_size = int(0.8*len(X))
    X_train = X[0:train_size]
    y_train = y[0:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
    y_train = y_train.reshape(y_train.shape[0],1)
    y_test = y_test.reshape(y_test.shape[0],1)
    return X_train,y_train,X_test,y_test

# Function to compute the cross entropy loss of the given predicted and true labels
def cross_entopy(pred,label):
    epsilon = 1e-6
    temp = label*np.log(pred+epsilon)+(1.0-label)*(np.log((1.0-pred)+epsilon))
    return (-1 * np.sum(temp))/len(label)

# Initialise the weights with appropriate sizes. The weights are taken from a normal distribution 
# with mean 0 and standard deviation 0.1
def weight_initialiser(dim1,dim2,out):
    weight0 = np.random.normal(0,0.1,size = (dim1,dim2))
    weight1 = np.random.normal(0,0.1,size = (dim2,out))
    return weight0,weight1

# Function to compute the forward pass layer activations
def forward_pass(X_train,weight0,weight1):
    layer0 = X_train
    layer1 = relu(np.dot(layer0,weight0))
    layer2 = sigmoid(np.dot(layer1,weight1))
    return layer0,layer1,layer2

# Compute the deltas in backward propogation. Returns the changes in weights
def backprop(X_train,y_train,layer0,layer1,layer2,weight0,weight1):
    d2 = y_train-layer2
    dweights2 = np.dot(layer1.T,d2)
    dweights1 = np.dot(X_train.T,(np.dot(d2,weight1.T)*relu_derivative(np.dot(layer0,weight0))))
    return dweights1,dweights2

# Function to train the model
def train(X_train,y_train,weight0, weight1,X_test,y_test, alpha = 0.1,epsilon = 0.05,num_batches=4):
    it = 0
    train_errors = []
    test_errors = []
    while True:
        it += 1
        # Get the places where the train data is to be split to form required number of batches
        l = [int(i*len(X_train)/num_batches) for i in range(num_batches+1)]
        pred = []
        # Loop over the train data in batches
        for i in range(num_batches):
            # Do forward pass and store the results
            layer0,layer1,layer2 = forward_pass(X_train[l[i]:l[i+1]],weight0,weight1)
            pred.extend(layer2)
            # Do backward prop and get the updates
            delta1,delta2 = backprop(X_train[l[i]:l[i+1]],y_train[l[i]:l[i+1]],layer0,layer1,layer2,weight0,weight1)
            # Update the weights
            weight0 += (alpha/len(X_train))*delta1
            weight1 += (alpha/len(X_train))*delta2
        # Compute error for each epoch - both training and test data errors
        error = cross_entopy(np.array(pred),y_train)
        ltest0,ltest1,ltest2 = forward_pass(X_test,weight0,weight1)
        test_error = cross_entopy(ltest2,y_test)
        print('iteration = ',it,' trainerror  = ',error,' testerror = ',test_error)
        train_errors.append(error)
        test_errors.append(test_error)
        # If train error is small, stop the training. 
        if(error<=epsilon):
            # Plot the train and test error vs Epochs
            ephocs = [i+1 for i in range(len(train_errors))]
            plt.title('Train and test errors wrt epochs')
            plt.xlabel('Epochs')
            plt.ylabel('Cross entropy error')
            tr = plt.plot(ephocs,train_errors)
            te = plt.plot(ephocs,test_errors)
            plt.legend([tr[0],te[0]],('Train Error','Test Error'),loc=0)
            plt.show()
            break
    # Return the learnt weights
    return weight0,weight1

# Function to predict the labels on test data using the learnt weights
def test(X_test,y_test,weight0,weight1,threshold=0.5):
    # Do forward prop on test data
    layer_0 = X_test
    layer_1 = relu(np.dot(layer_0,weight0))
    layer_2 = sigmoid(np.dot(layer_1,weight1))
    correct = 0
    # Loop over the data and calculate accuracy
    for i in range(len(layer_2)):
        if(layer_2[i][0] > threshold):
            layer_2[i][0] = 1
        else:
            layer_2[i][0] = 0
        if(layer_2[i] == y_test[i]):
            correct += 1
    print('accuracy = , '+str(correct * 100.0 / len(layer_2)))

# Main Function
def main():
    preprocess()
    X_train,y_train,X_test,y_test = dataloader()
    dim1 = len(X_train[0])
    dim2 = 100
    out = 1
    weight0,weight1 = weight_initialiser(dim1,dim2,out)
    alpha = 0.1
    threshold = 0.5
    weight0,weight1 =  train(X_train,y_train,weight0,weight1,X_test,y_test,alpha=alpha,num_batches=8)
    test(X_test,y_test,weight0,weight1,threshold=threshold)

if __name__ == "__main__":
    main()    
