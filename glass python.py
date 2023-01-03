# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 12:26:34 2022

@author: Anil Kumar
"""

import pandas as pd
import numpy as np
df = pd.read_csv("glass.csv")
df.info()

X = df.iloc[:,0:9]
Y = df['Type']

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_scale = ss.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X_scale,Y,test_size=0.3,random_state=(3))

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=8,p=3)
knn.fit(X_train, Y_train)
Y_pred_train = knn.predict(X_train)
Y_pred_test = knn.predict(X_test)

from sklearn.metrics import accuracy_score

print('Train accuracy', accuracy_score(Y_train,Y_pred_train).round(3))
print('Test accuracy', accuracy_score(Y_test,Y_pred_test).round(3))








