# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 06:24:53 2021

@author: ASUS
"""
#Importing all the library files
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

#Loading up the data file
file=pd.read_csv("D:\Data Science Assignments\R-Assignment\Decision_Tree\Company_Data.csv")
file.describe()

#Data Manipulation also used Hot En-Coding Method
data=file
data.isna().sum()
data['Sales']=data['Sales'].fillna(method='pad')
data['Sales']=pd.cut(x=file['Sales'],bins=[0,4,8,12,17], labels=['Bad','Medium','Good','Best'],right=True)
data['US']=[1 if x =='Yes' else 0 for x in file['US']]
data['Urban']=[1 if x =='Yes' else 0 for x in file['Urban']]
data=pd.get_dummies(data,columns=['ShelveLoc'])
data.head()
data.dtypes
data.Sales.value_counts()

#Splitting the data into training and testing dataset
train,test=train_test_split(data,test_size=0.2)
train.Sales.value_counts()
test.Sales.value_counts()

#Defining the target and the predictor columns
colnames=list(data)
target = colnames[0]
predictor=colnames[1:]

#Formation of Model
model=DecisionTreeClassifier(criterion='entropy')
model.fit(train[predictor],train[target])

#Decision Tree plot
plot_tree(model)

#Checking the accuracy of the model
preds=model.predict(test[predictor])
pd.Series(preds).value_counts()
pd.crosstab(test['Sales'],preds)
np.mean(preds==test['Sales'])
