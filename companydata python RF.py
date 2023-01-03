# -*- coding: utf-8 -*-
"""
Created on

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


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=400,max_depth=(3),max_features=0.7,max_samples=0.6)

rfr.fit(X_train, Y_train)

Y_pred_train = rfr.predict(X_train)
Y_pred_test = rfr.predict(X_test)

from sklearn.metrics import mean_squared_error

print('train mse', mean_squared_error(Y_train,Y_pred_train).round(3))
print('test mse', mean_squared_error(Y_test,Y_pred_test).round(3))
