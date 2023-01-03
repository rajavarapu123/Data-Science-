# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 17:13:16 2022

@author: Anil Kumar
"""

import pandas as pd
import numpy as np
df = pd.read_csv('Company_Data.csv')
df.info()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['ShelveLoc'] = le.fit_transform(df['ShelveLoc'])
df['Urban'] = le.fit_transform(df['Urban'])
df['US'] = le.fit_transform(df['US'])

X = df.iloc[:,1:]
Y = df['Sales']

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=(3))

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(max_depth=(6))
dt.fit(X_train, Y_train)

Y_pred_train = dt.predict(X_train)
Y_pred_test = dt.predict(X_test)

from sklearn.metrics import mean_squared_error

print('train mse', mean_squared_error(Y_train,Y_pred_train).round(3))
print('test mse', mean_squared_error(Y_test,Y_pred_test).round(3))

from sklearn.ensemble import BaggingRegressor
br = BaggingRegressor(base_estimator=(dt),n_estimators=300,max_samples=0.6,max_features=0.7,random_state=(3))
br.fit(X_train, Y_train)

Y_pred_train = br.predict(X_train)
Y_pred_test = br.predict(X_test)

from sklearn.metrics import mean_squared_error

print('train mse', mean_squared_error(Y_train,Y_pred_train).round(3))
print('test mse', mean_squared_error(Y_test,Y_pred_test).round(3))

train_error=[]
test_error=[]

for i in range(1,600):
    br = BaggingRegressor(base_estimator=(dt),n_estimators=300,max_samples=0.6,max_features=0.7,random_state=(3))
    br.fit(X_train, Y_train)
    Y_pred_train = br.predict(X_train)
    Y_pred_test = br.predict(X_test)
    train_error.append(mean_squared_error(Y_train,Y_pred_train).round(3))
    test_error.append(mean_squared_error(Y_test,Y_pred_test).round(3))

np.mean(train_error).round(3)
np.mean(test_error).round(3)

# after doing the multiple iteration also got the nearer value of errors and varience among the trian and test also no change.

