import pandas as pd
import numpy as np
eps = np.finfo(float).eps
from numpy import log2 as log
import matplotlib.pyplot as plt

# Load the list of words into an array
def load_column_names(column_file):
    columns = []
    with open(column_file,"r") as f:
        columns = f.read().split()
    columns.append('label')
    return columns

# Load and process the train data
def load_train_data(train_data_file,train_label_file,columns):
    num_docs = 1061
    num_words = 3566
    train_data = {}
    for i in range(num_docs):
        train_data[i] = [0 for j in range(num_words)]

    with open(train_data_file,"r") as f:
        for line in f:
            l = line.split()
            train_data[int(l[0])-1][int(l[1])-1] = 1

    with open(train_label_file,"r") as f:
        i = 0
        for line in f:
            label = int(line)
            train_data[i].append(label)
            i += 1

    train_data_df = pd.DataFrame.from_dict(train_data,orient='index')
    train_data_df.columns = columns
    return train_data_df

# Load and process the test data
def load_test_data(test_data_file,test_label_file,columns):
    num_words = 3566
    num_test_data_docs = 707
    test_data = {}
    for i in range(num_test_data_docs):
        test_data[i] = [0 for j in range(num_words)]
    with open(test_data_file,"r") as f:
        for line in f:
            l = line.split()
            test_data[int(l[0])-1][int(l[1])-1] = 1
    with open(test_label_file,"r") as f:
        i=0
        for line in f:
            label = int(line)
            test_data[i].append(label)
            i += 1
    test_data_df = pd.DataFrame.from_dict(test_data,orient='index')
    test_data_df.columns = columns
    return test_data_df

# Find entropy of parent node
def entropy_parent(df):
    entropy_node = 0
    values = df['label'].unique()
    for value in values:
        fraction = df['label'].value_counts()[value]/len(df['label'])
        entropy_node += -fraction*np.log2(fraction)
    return entropy_node

# Function to find entropy of the split with an attribute
def entropy_split(df,attribute):
    target_variables = df['label'].unique()
    variables = df[attribute].unique()
    entropy_attribute = 0
    for variable in variables:
        entropy_each_feature = 0
        for target_variable in target_variables:
            num = len(df[attribute][df[attribute]==variable][df['label'] == target_variable])
            den = len(df[attribute][df[attribute]==variable])
            fraction = num/(den+eps)
            entropy_each_feature += -fraction*log(fraction+eps)
        fraction2 = den/len(df)
        entropy_attribute += -fraction2*entropy_each_feature
    return abs(entropy_attribute)

# Iteratively check all features and find maximum information gain as the current root
def find_best_split(df):
    info_gains = []
    ent_parent = entropy_parent(df)
    for key in df.keys()[:-1]:
        info_gains.append(ent_parent - entropy_split(df,key))
    return df.keys()[:-1][np.argmax(info_gains)]

# Get the subset of the data for given node having the given value
def get_subtable(df,node,value):
    return df[df[node]==value].reset_index(drop=True)

# Recursively build the tree upto maxdepth height using the best split function
def build_tree(df,depth,maxdepth,tree=None):
    full_tree = True
    node = find_best_split(df)
    attValues = np.unique(df[node])
    if tree is None:
        tree = {}
        tree[node] = {}
    for value in attValues:
        subtable = get_subtable(df,node,value)
        clValue,counts = np.unique(subtable['label'],return_counts=True)
        if len(counts)==1:
            tree[node][value] = clValue[0]
            full_tree = full_tree and True
        elif depth == maxdepth:
            tree[node][value] = clValue[np.argmax(counts)]
            full_tree = False
        else:
            tree[node][value],temp = build_tree(subtable,depth+1,maxdepth)
            full_tree = full_tree and temp
    return tree,full_tree

# Function to print the tree in the given format
def print_tree(tree,level):
    if isinstance(tree,dict):
        print('')
        attr = list(tree.keys())[0]
        for i in tree[attr]:
            if level>0:
                for j in range(level-1):
                    print('\t',end='')
                print('|',end=' ')
            print(attr + ' = '+str(i),end=' ')
            print_tree(tree[attr][i],level+1)
    else:
        print(': ' + str(tree))

# Function to predict the given instance of the data and the built tree        
def predict(inst,tree):
    for nodes in tree.keys():
        value = inst[nodes]
        tree = tree[nodes][value]
        prediction = ''
        if isinstance(tree,dict):
            prediction = predict(inst,tree)
        else:
            prediction = tree
            break
    return prediction

# Predict the outputs for the given test data and obtain the accuracy using the truth values
def predict_accuracy(test_x,test_y,tree):
    correct = 0
    total = len(test_x)
    for index,row in test_x.iterrows():
        if predict(row,tree)==test_y.iloc[index]:
            correct += 1
    return 100*(correct)/(total+eps)

# Main function
def main():
    column_file = 'dataset for part 2/words.txt'
    columns = load_column_names(column_file)
    train_data_file = 'dataset for part 2/traindata.txt'
    train_label_file = 'dataset for part 2/trainlabel.txt'
    train_data_df = load_train_data(train_data_file,train_label_file,columns)
    test_data_file = 'dataset for part 2/testdata.txt'
    test_label_file = 'dataset for part 2/testlabel.txt'
    test_data_df = load_test_data(test_data_file,test_label_file,columns)

    train_x = train_data_df.drop('label',axis=1)
    train_y = train_data_df['label']
    test_x = test_data_df.drop('label',axis=1)
    test_y = test_data_df['label']
    
    training_acc = []
    testing_acc = []

    full_tree = False
    size = 1
    while not full_tree:
        print('size of tree = '+str(size))
        tree,full_tree = build_tree(train_data_df,0,size)
        train_acc = predict_accuracy(train_x,train_y,tree)
        test_acc = predict_accuracy(test_x,test_y,tree)
        training_acc.append(train_acc)
        testing_acc.append(test_acc)
        print('train_acc = '+str(train_acc)+' test_acc = '+str(test_acc))
        size += 1

    sizes = list(range(1,size))
    plt.plot(sizes,training_acc)
    plt.plot(sizes,testing_acc)
    plt.show()

if __name__ == '__main__':
    main()