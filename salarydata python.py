# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 15:16:57 2022

@author: Anil Kumar
"""

import pandas as pd
import numpy as np
df = pd.read_csv('SalaryData_Train.csv')
df.info()

df_cont = df[df.columns[[0,3,9,10,11]]]
df_catg = df[df.columns[[1,2,4,5,6,7,8,12]]]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in range(0,8):
    df_catg.iloc[:,i] = le.fit_transform(df_catg.iloc[:,i])
df1 = pd.concat([df_cont,df_catg],axis=1,ignore_index=True)

X_train = df1.iloc[:,0:13]
Y_train = df['Salary']

ak = pd.read_csv('SalaryData_Test.csv')
ak.info()

ak_cont = ak[ak.columns[[0,3,9,10,11]]]
ak_catg = ak[ak.columns[[1,2,4,5,6,7,8,12]]]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in range(0,8):
    ak_catg.iloc[:,i] = le.fit_transform(ak_catg.iloc[:,i])
ak1 = pd.concat([ak_cont,ak_catg],axis=1,ignore_index=True)

X_test = ak1.iloc[:,0:13]
Y_test = ak['Salary']

from sklearn.naive_bayes import MultinomialNB
mb = MultinomialNB()
mb.fit(X_train, Y_train)

Y_pred_train = mb.predict(X_train)
Y_pred_test = mb.predict(X_test)

from sklearn.metrics import accuracy_score

print("tarin accuracy", accuracy_score(Y_train,Y_pred_train).round(3))
print("test accuracy", accuracy_score(Y_test,Y_pred_test).round(3))








