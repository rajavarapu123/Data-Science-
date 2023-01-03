# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 18:00:24 2022

@author: Anil Kumar
"""

import pandas as pd
import numpy as np
ak = pd.read_csv("50_Startups.csv")
ak.info()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
ak["State"] = le.fit_transform(ak["State"])
ak["State"] = pd.DataFrame(ak["State"])
ak.info()

from sklearn.preprocessing import StandardScaler
ss = StandardScaler() 
xs = ss.fit_transform(ak.iloc[:,0:3])
xs = pd.DataFrame(xs)
xs.columns = ['R&D Spend','Administration','Marketing Spend']

df = pd.concat([xs,ak["State"],ak['Profit']],axis=1)
df

df.corr()

import matplotlib.pyplot as plt
import seaborn as sns
plt.scatter(df.iloc[:,1],df.iloc[:,4])
plt.scatter(df.iloc[:,0],df.iloc[:,4])

sns.histplot(df.iloc[:,3])
sns.barplot(df.iloc[:,2], df.iloc[:,4])


#model 1 #r2 score= 95% and multicollinearity issues exist
X1=df.iloc[:,0:4] 
Y=ak["Profit"]

#model 2 #r2 score= 95% and multicollinearity issues doesnot exist
X2=df[["R&D Spend"]]
Y=ak["Profit"]

#model 3 #r2 score=95% and multicollinearity issues exist
X3=df[["R&D Spend","Marketing Spend"]] 
Y=ak["Profit"]

#model 4 #r2 score= 95% and multicollinearity issues exist
X4=df[["R&D Spend","Administration"]] 
Y=ak["Profit"]

#model 5 #r2 score= 61% and multicollinearity issues exist
X5=df[["Marketing Spend","Administration"]]
Y=ak["Profit"]

#model 6 #r2 score= 95% and multicollinearity issues exist
X6=df[["R&D Spend","State"]] 
Y=ak["Profit"]

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X2, Y)

Y_pred = lr.predict(X2)

from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(Y,Y_pred)
r2score = r2_score(Y,Y_pred)
print(mse)
print(r2score)

import statsmodels.api as sma
x_new = sma.add_constant(X2)
lm = sma.OLS(Y, x_new).fit()
lm.summary()
lm.rsquared_adj

ab = np.array([['model-1',95,'mlt exists'],
               ['model-2',95,'no mlt exists'],
               ['model-3',95,'mlt exists'],
               ['model-4',95,'mlt exists'],
               ['model-5',61,'mlt exists'],
               ['model-6',95,'mlt exists']])
ab
table = pd.DataFrame(ab)
table.columns = ['model name','R2 value','MUlticolliniarity']
table

# cooks distance method to find the outliers

np.printoptions(supress=True)
influence = lm.get_influence()
cooks_distance = influence.cooks_distance
cooks_distance
np.argmax(cooks_distance), np.max(cooks_distance)

# there is no ouliers in data.

# Leverage cutoff value
k=df.shape[1]
n=df.shape[0]
leverage_cutoff = (3*(k+1))/n
leverage_cutoff

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12,8))
fig = sma.graphics.plot_regress_exog(lm, 'R&D Spend',fig=fig)
fig





