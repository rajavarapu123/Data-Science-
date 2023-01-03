# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 12:47:49 2022

@author: Anil Kumar
"""

import pandas as pd
import numpy as np
df = pd.read_csv('SalaryData_Train(1).csv')
df.info()

df_cont = df[df.columns[[0,3,9,10,11]]]
df_catg = df[df.columns[[1,2,4,5,6,7,8,12,13]]]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in range(0,9):
    df_catg.iloc[:,i] = le.fit_transform(df_catg.iloc[:,i])

df1 = pd.concat([df_cont,df_catg],axis=1,ignore_index=True)

X_train = df1.iloc[:,0:13]
Y_train = df['Salary']

import matplotlib.pyplot as plt
plt.scatter(X.iloc[:,0],X.iloc[:,2])
plt.scatter(X.iloc[:,5],X.iloc[:,8])
plt.scatter(X.iloc[:,10],X.iloc[:,13])

import seaborn as sns
sns.histplot(X_train.iloc[:,6]) # Univarite plot
sns.countplot(X_train.iloc[:,4])

plt.figure(figsize=(12,8)) # Bivariate plot
sns.barplot(X_train.iloc[:,2], Y_train)
sns.barplot(X_train.iloc[:,8], Y_train)


df2 = pd.read_csv('SalaryData_Test(1).csv')
df2.info()

df2_cont = df2[df2.columns[[0,3,9,10,11]]]
df2_catg = df2[df2.columns[[1,2,4,5,6,7,8,12,13]]]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in range(0,9):
    df2_catg.iloc[:,i] = le.fit_transform(df2_catg.iloc[:,i])

df3 = pd.concat([df2_cont,df2_catg],axis=1,ignore_index=True)

X_test = df3.iloc[:,0:13]
Y_test = df2['Salary']

import matplotlib.pyplot as plt
plt.scatter(X.iloc[:,1],X.iloc[:,4])
plt.scatter(X.iloc[:,5],X.iloc[:,7])
plt.scatter(X.iloc[:,11],X.iloc[:,12])

import seaborn as sns
sns.histplot(X_train.iloc[:,7]) # Univarite plot
sns.countplot(X_train.iloc[:,1])

plt.figure(figsize=(12,8)) # Bivariate plot
sns.barplot(X_train.iloc[:,10], Y_train)
sns.barplot(X_train.iloc[:,9], Y_train)

from sklearn.svm import SVC
clf = SVC(kernel='rbf', gamma=1) # got the best result.
#clf = SVC(kernel='poly', degree=2)

clf.fit(X_train, Y_train)

Y_pred_train = clf.predict(X_train)
Y_pred_test = clf.predict(X_test)

from sklearn.metrics import accuracy_score

print('train accuracy', accuracy_score(Y_train,Y_pred_train).round(3))
print('test accuracy', accuracy_score(Y_test,Y_pred_test).round(3))

# plotting the visualisation
pip install mlxtend
X = df.iloc[:,0:4]
Y = df.iloc[:,13]

from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X=X.values, 
                      y=Y.values,
                      clf=clf, 
                      legend=4)











