import piplite
await piplite.install(['pandas'])
await piplite.install(['matplotlib'])
await piplite.install(['numpy'])
await piplite.install(['scikit-learn'])

import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree

from pyodide.http import pyfetch

async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f:
            f.write(await response.bytes())

#edit data path as necessary
path= 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv'
await download(path,"drug200.csv")
path="drug200.csv"

#take a peek at the data
my_data = pd.read_csv("drug200.csv", delimiter=",")
my_data[0:5]

#data shape
my_data.shape

"""
preprocessing
"""
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
X[0:5]

"""
returns an array like:
array([[23, 'F', 'HIGH', 'HIGH', 25.355],
       [47, 'M', 'LOW', 'HIGH', 13.093],
       [47, 'M', 'LOW', 'HIGH', 10.114],
       [28, 'F', 'NORMAL', 'HIGH', 7.798],
       [61, 'F', 'LOW', 'HIGH', 18.043]], dtype=object)
some features in this dataset are categorical, such as Sex or BP. 
Sklearn Decision Trees does not handle categorical variables. We can still convert these features to numerical values using pandas.get_dummies() 
to convert the categorical variable into dummy/indicator variables.       
"""

from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

X[0:5]

#find target var
y = my_data["Drug"]
y[0:5]

"""
returns:
0    drugY
1    drugC
2    drugC
3    drugX
4    drugY
Name: Drug, dtype: object
"""

#setting up decision tree
from sklearn.model_selection import train_test_split

X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

#shape of X_trainset & y_trainset
print('Shape of X training set {}'.format(X_trainset.shape),'&',' Size of Y training set {}'.format(y_trainset.shape))
#shape of test sets
print('Shape of X training set {}'.format(X_testset.shape),'&',' Size of Y training set {}'.format(y_testset.shape))

#modeling
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree # it shows the default parameters

drugTree.fit(X_trainset,y_trainset)

#make some predictions
predTree = drugTree.predict(X_testset)

#print results
print (predTree [0:5])
print (y_testset [0:5])

#eval metrics
from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))

"""
returns back: 0.98333 for this example with the dataset
"""

#visualize using pydotplus & python graph-viz
#conda install -c conda-forge pydotplus -y
#conda install -c conda-forge python-graphviz -y

tree.plot_tree(drugTree)
plt.show()
