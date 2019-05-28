import pandas as pd
import numpy as np
eps = np.finfo(float).eps
from numpy import log2 as log
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

columns = []
column_file = 'dataset for part 2/words.txt'
with open(column_file,"r") as f:
    columns = f.read().split()

columns.append('label')

num_docs = 1061
num_words = 3566
train_data = {}
for i in range(num_docs):
    train_data[i] = [0 for j in range(num_words)]

train_data_file = 'dataset for part 2/traindata.txt'
with open(train_data_file,"r") as f:
    for line in f:
        l = line.split()
        train_data[int(l[0])-1][int(l[1])-1] = 1        

train_label_file = 'dataset for part 2/trainlabel.txt'
with open(train_label_file,"r") as f:
    i = 0
    for line in f:
        label = int(line)
        train_data[i].append(label)
        i += 1

train_data_df = pd.DataFrame.from_dict(train_data,orient='index')

train_data_df.columns = columns
train_data_df = train_data_df.astype(int)

num_test_data_docs = 707
test_data = {}
for i in range(num_test_data_docs):
    test_data[i] = [0 for j in range(num_words)]

test_data_file = 'dataset for part 2/testdata.txt'
with open(test_data_file,"r") as f:
    for line in f:
        l = line.split()
        test_data[int(l[0])-1][int(l[1])-1] = 1

test_label_file = 'dataset for part 2/testlabel.txt'
with open(test_label_file,"r") as f:
    i=0
    for line in f:
        label = int(line)
        test_data[i].append(label)
        i += 1
test_data_df = pd.DataFrame.from_dict(test_data,orient='index')
test_data_df.columns = columns
test_data_df = test_data_df.astype(int)

train_x = train_data_df.drop('label',axis=1)
train_y = train_data_df['label']
test_x = test_data_df.drop('label',axis=1)
test_y = test_data_df['label']

training_acc = []
testing_acc = []
train_acc = 0
sizes = []
i=1
while train_acc != 1:
    tree = DecisionTreeClassifier(criterion='entropy',max_depth=i)
    tree.fit(train_x,train_y)
    predictions = tree.predict(train_x)
    train_acc = metrics.accuracy_score(predictions,train_y)
    training_acc.append(train_acc)
    predictions = tree.predict(test_x)
    test_acc = metrics.accuracy_score(predictions,test_y)
    testing_acc.append(test_acc)
    print('train_acc = '+str(train_acc)+' test_acc = '+str(test_acc))
    sizes.append(i)
    i+=1
    
plt.plot(sizes,training_acc)
plt.plot(sizes,testing_acc)
plt.show()