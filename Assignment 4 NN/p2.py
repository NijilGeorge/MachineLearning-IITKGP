import numpy as np
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import sys
import matplotlib.pyplot as plt

# Compute the sigmoid activation output for the given numpy array
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# Compute the sigmoid derivative output for the given numpy array
def sigmoid_derivative(x):
    return sigmoid(x)*(1.0-sigmoid(x))

# Convert the given numpy array of activations into softmax distributions
def softmax(x):
    num = np.exp(x)
    den = np.sum(np.exp(x),axis=1)
    den = den.reshape(den.shape[0],1)
    return num/den

# Function for preprocessing the data
def preprocess():
    X = []
    y = []
    vocabulary = set({})
    ps =PorterStemmer()
    # Get the stop words
    stop_words = set(stopwords.words('english'))
    # Read the lines
    with open('Assignment_4_data.txt','r') as f:
        for line in f:
            # Tokenize the words
            line = line.replace('\t',' ')
            line = line.replace('\n',' ')
            line = line.replace('.',' ')
            line = line.replace(',',' ')
            line = line.replace(':',' ')
            line = line.replace('-',' ')
            tokens = line.split()
            if(tokens[0]=='ham'):
                y.append([1,0])
            elif(tokens[0]=='spam'):
                y.append([0,1])
            tokens = tokens[1:]
            stemmed_tokens = []
            # Stem the tokens and check for stop words. Append into train data and vocabulary
            for word in tokens:
                if(word not in stop_words):
                    stem_word = ps.stem(word)
                    vocabulary.add(stem_word)
                    stemmed_tokens.append(stem_word)
            X.append(stemmed_tokens)
    vocabulary = sorted(list(vocabulary))
    # One hot encode the train data
    one_hot_X = np.zeros((len(X),len(vocabulary)),dtype=np.int8)
    for i in range(len(X)):
        for j in range(len(X[i])):
            word_ind = vocabulary.index(X[i][j])
            one_hot_X[i][word_ind] = 1
    np.save('features2.npy',one_hot_X)
    np.save('labels2.npy',np.array(y))

# Function to load and split the data into 80-20 train data and test data
def dataloader():
    X = np.load('features2.npy')
    y = np.load('labels2.npy')
    train_size = int(0.8*len(X))
    X_train = X[0:train_size]
    y_train = y[0:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
    y_train = y_train.reshape(y_train.shape[0],2)
    y_test = y_test.reshape(y_test.shape[0],2)
    return X_train,y_train,X_test,y_test

# Function to compute the categorical cross entropy of the given predicted and true labels
def cross_entopy(pred,label):
    epsilon = 1e-6
    temp = label[:,:1]*np.log(pred[:,:1]+epsilon)+(label[:,1:])*(np.log((pred[:,1:])+epsilon))
    return (-1 * np.sum(temp))/len(label)

# Function to initialize the weights according to the given dimensions
# The weights are drawn from a normal distribution of mean 0 and standard deviation 0.1
def weight_initialiser(dim1,dim2,dim3,out):
    weight0 = np.random.normal(0,0.1,size = (dim1,dim2))
    weight1 = np.random.normal(0,0.1,size = (dim2,dim3))
    weight2 = np.random.normal(0,0.1,size = (dim3,out))
    return weight0,weight1,weight2

# Perform forward propogation on the inputs
def forward_pass(X_train,weight0,weight1,weight2):
    layer0 = X_train
    layer1 = sigmoid(np.dot(layer0,weight0))
    layer2 = sigmoid(np.dot(layer1,weight1))
    layer3 = softmax(np.dot(layer2,weight2))
    return layer0,layer1,layer2,layer3

# Perform backward propogation and compute the required changes in weights
def backprop(X_train,y_train,layer0,layer1,layer2,layer3,weight0,weight1,weight2):
    d3 = y_train - layer3
    dweights3 = np.dot(layer2.T,d3)
    d2 = np.dot(d3,weight2.T)*sigmoid_derivative(np.dot(layer1,weight1))
    dweights2 = np.dot(layer1.T,d2)
    d1 = np.dot(d2,weight1.T)*sigmoid_derivative(np.dot(X_train,weight0))
    dweights1 = np.dot(X_train.T,d1)
    return dweights1,dweights2,dweights3

# Function to perform the training
def train(X_train,y_train,weight0, weight1,weight2,X_test,y_test, alpha = 0.1,epsilon = 0.1,batchsize=4):
    it = 0
    train_errors = []
    test_errors = []
    m = len(X_train)
    while True:
        it += 1
        # Get the split points to divide the data into mini batches
        l = [int(i*len(X_train)/batchsize) for i in range(batchsize+1)]
        pred = []
        # Loop over the batches in each epoch
        for i in range(batchsize):
            # Perform forward prop
            layer0,layer1,layer2,layer3 = forward_pass(X_train[l[i]:l[i+1]],weight0,weight1,weight2)
            pred.extend(layer3)
            # Perform backprop
            delta1,delta2,delta3 = backprop(X_train[l[i]:l[i+1]],y_train[l[i]:l[i+1]],layer0,layer1,layer2,layer3,weight0,weight1,weight2)
            # Update the weights
            weight0 += (alpha/m)*delta1
            weight1 += (alpha/m)*delta2
            weight2 += (alpha/m)*delta3
        # Calculate the categorical cross entropy after each epoch
        error = cross_entopy(np.array(pred),y_train)
        ltest0,ltest1,ltest2,ltest3 = forward_pass(X_test,weight0,weight1,weight2)
        test_error = cross_entopy(ltest3,y_test)
        print('iteration = ',it,' trainerror  = ',error,' testerror = ',test_error)
        train_errors.append(error)
        test_errors.append(test_error)
        # If the training error is low, then stop the training
        if(error<=epsilon):
            break
    # Plot the graph of train and test errors vs Epochs
    ephocs = [i+1 for i in range(len(train_errors))]
    plt.title('Train and test errors wrt epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Cross entropy error')
    tr = plt.plot(ephocs,train_errors)
    te = plt.plot(ephocs,test_errors)
    plt.legend([tr[0],te[0]],('Train Error','Test Error'),loc=0)
    plt.show()
    return weight0,weight1,weight2

# Function to predict and display the accuracy on test data wrt the learnt weights
def test(X_test,y_test,weight0,weight1,weight2):
    # Perform forward prop on test data
    layer_0 = X_test
    layer_1 = sigmoid(np.dot(layer_0,weight0))
    layer_2 = sigmoid(np.dot(layer_1,weight1))
    layer_3 = softmax(np.dot(layer_2,weight2))
    correct = 0
    preds = [0]*len(layer_3)
    # Loop over the test data and compute the accuracy
    for i in range(len(layer_3)):
        if(layer_3[i][0] > layer_3[i][1] and y_test[i][0]==1):
            correct += 1
        elif(layer_3[i][1]>layer_3[i][0] and y_test[i][1]==1):
            correct += 1
    print('accuracy = , '+str(correct * 100.0 / len(layer_2)))

# Main function
def main():
    preprocess()
    X_train,y_train,X_test,y_test = dataloader()
    dim1 = len(X_train[0])
    dim2 = int(sys.argv[1])
    dim3 = int(sys.argv[2])
    out = 2
    weight0,weight1,weight2 = weight_initialiser(dim1,dim2,dim3,out)
    alpha = 0.1
    weight0,weight1,weight2 =  train(X_train,y_train,weight0,weight1,weight2,X_test,y_test,batchsize=5)
    test(X_test,y_test,weight0,weight1,weight2)

if __name__ == "__main__":
    main()    
