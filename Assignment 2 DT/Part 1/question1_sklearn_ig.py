# Necessary imports
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

# Function to load and encode the data for sklearn tree
def load_and_preprocess_data():
    train_data = pd.read_excel('dataset for part 1.xlsx',sheet_name=0)
    test_data = pd.read_excel('dataset for part 1.xlsx',sheet_name=1)
    data = pd.concat((train_data,test_data),ignore_index=True)
    # Take each datatype as a string
    data = data.astype(str)
    # One hot encode the data
    data = pd.get_dummies(data)
    # Split the train and test data
    train_data = data.iloc[:9,:]
    test_data = data.iloc[9:,:].reset_index(drop=True)
    # Separate out the labels from the features
    train_x = train_data.drop(['profitable_yes','profitable_no'],axis=1)
    train_y = train_data[['profitable_yes','profitable_no']]
    test_x = test_data.drop(['profitable_yes','profitable_no'],axis=1)
    test_y = test_data[['profitable_yes','profitable_no']]
    return train_x,train_y,test_x,test_y

# Main function
def main():
    train_x,train_y,test_x,test_y = load_and_preprocess_data()
    # Fit the decision tree with entropy as splitting function
    tree_ig = DecisionTreeClassifier(criterion='entropy',random_state=2)
    tree_ig.fit(train_x,train_y)
    predictions = tree_ig.predict(test_x)
    accuracy = metrics.accuracy_score(predictions,test_y)
    print('Obtained accuracy on test data with info gain = '+str(accuracy))
    # Find and print the gini of the root node
    print('Obtained information gain of the root node is = '+str(tree_ig.tree_.impurity[0]-tree_ig.tree_.impurity[1]*(tree_ig.tree_.n_node_samples[1]/tree_ig.tree_.n_node_samples[0])))

if __name__ == '__main__':
    main()

