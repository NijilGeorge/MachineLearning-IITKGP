# Necessary imports
import numpy as np
import pandas as pd
eps = np.finfo(float).eps
from numpy import log2 as log

# Function to find gini index of the parent
def gini_parent(df):
    gini_node = 0
    # Get the class labels
    values = df['profitable'].unique()
    for value in values:
        fraction = df['profitable'].value_counts()[value]/len(df['profitable'])
        gini_node += np.square(fraction)
    return 1-gini_node

# Function to find gini index of the split with an attribute
def gini_split(df,attribute):
    # Get the class labels and attribute values
    target_variables = df['profitable'].unique()
    variables = df[attribute].unique()
    gini_attribute = 0
    for variable in variables:
        gini_each_feature = 0
        for target_variable in target_variables:
            num = len(df[attribute][df[attribute]==variable][df['profitable'] == target_variable])
            den = len(df[attribute][df[attribute]==variable])
            fraction = num/(den+eps)
            gini_each_feature += np.square(fraction)
        fraction2 = den/len(df)
        gini_attribute += fraction2*(1-gini_each_feature)
    return gini_attribute

# Iteratively check all features and find minimum gini split as the current root
def find_best_split(df):
    gini_splits = []
    for key in df.keys()[:-1]:
        gini_splits.append(gini_split(df,key))
    return df.keys()[:-1][np.argmin(gini_splits)]

# Get the subset of the data for given node having the given value
def get_subtable(df,node,value):
    return df[df[node]==value].reset_index(drop=True)

# Recursively build the tree using the best split function
def build_tree(df,tree=None):
    node = find_best_split(df)
    attValues = np.unique(df[node])
    if tree is None:
        tree = {}
        tree[node] = {}
    for value in attValues:
        subtable = get_subtable(df,node,value)
        clValue,counts = np.unique(subtable['profitable'],return_counts=True)
        if len(counts)==1:
            tree[node][value] = clValue[0]
        else:
            tree[node][value] = build_tree(subtable)
    return tree

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

# Function to find the gini of the root node of the tree
def gini_root_node(df,tree):
    node = list(tree.keys())[0]
    gini = gini_split(df,node)
    return gini

# Main function
def main():
    # Read the training and test data
    train_data = pd.read_excel('dataset for part 1.xlsx',sheet_name=0)
    test_data = pd.read_excel('dataset for part 1.xlsx',sheet_name=1)
    # Build the tree
    tree = build_tree(train_data)
    print('Generated tree')
    # Print the tree
    print_tree(tree,0)
    # Obtain the test data without labels, and the labels separately
    test_x = test_data.drop('profitable',axis=1)
    test_y = test_data['profitable']
    print()
    # Predict and print the accuracy
    print('Accuracy on testdata = '+ str(predict_accuracy(test_x,test_y,tree)))
    # Find and print the gini of the root node
    gini_root = gini_root_node(train_data,tree)
    print('GINI of root node = '+str(gini_root))

if __name__ == '__main__':
    main()


                    