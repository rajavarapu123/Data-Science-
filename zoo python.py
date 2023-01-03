# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 13:09:20 2022

@author: Anil Kumar
"""

import pandas as pd
import numpy as np
df = pd.read_csv("Zoo.csv")
df.info()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['animal name'] = le.fit_transform(df['animal name'])

df.info()

X = df.iloc[:,0:17]
Y = df['type']

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_scale = ss.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X_scale,Y,test_size=0.3,random_state=(5))

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=9,p=2)

knn.fit(X_train, Y_train)
Y_pred_train = knn.predict(X_train)
Y_pred_test = knn.predict(X_test)

from sklearn.metrics import accuracy_score

print('train accuracy score', accuracy_score(Y_train,Y_pred_train))
print('test accuracy score', accuracy_score(Y_test,Y_pred_test))



