# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 13:22:51 2022

@author: Anil Kumar
"""

import pandas as pd
import numpy as no
df = pd.read_csv("Toyota_Corolla.csv", encoding=("ISO-8859-1"))
df.head()
df.info()

df = df.drop(df.columns[[0,1,4,5,7,9,10,11,14,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]], axis=1)
df.info()
df.corr()

X = df.iloc[:,1:]
Y = df["Price"]

import matplotlib.pyplot as plt
import seaborn as sns

plt.scatter(X.iloc[:,3], X.iloc[:,0])
plt.scatter(X.iloc[:,6], X.iloc[:,0])

sns.histplot(X.iloc[:,6])
sns.histplot(X.iloc[:,4])

sns.barplot(X.iloc[:,7], X.iloc[:,1])

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_scale = ss.fit_transform(X)


from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
LS = Lasso(alpha=35)
LS.fit(X_scale, Y)

Y_pred = LS.predict(X_scale)

mse = mean_squared_error(Y, Y_pred)
print(mse)
pd.concat([pd.DataFrame(X.columns),pd.DataFrame(LS.coef_)], axis=1)

# After applying the lasso regression i found some of x varibles influencing the target varibles those are below
# Below x varibles doesn't contain any multi colliniarity issue and have the greater R square value.
X = df[["Age_08_04","Weight","KM","HP","Gears","Quarterly_Tax"]]
Y=df["Price"]

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_scale = ss.fit_transform(X)


from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X_scale,Y)

Y_pred = LR.predict(X_scale)

from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(Y,Y_pred)
r2score = r2_score(Y,Y_pred)
print(mse)
print(r2score)


import statsmodels.api as sma
x_1 = sma.add_constant(X)
tm = sma.OLS(Y, x_1).fit()
tm.summary()

# cooks distance
no.printoptions(supress=True)
influence = tm.get_influence()
cooks_distance = influence.cooks_distance
cooks_distance
df = df.drop(df.columns[[4,5]],axis=1)
df
no.argmax(cooks_distance), no.max(cooks_distance)

# Leverage cutoff value
k=df.shape[1]
n=df.shape[0]
leverage_cutoff = (3*(k+1))/n
leverage_cutoff

df[df.index.isin([221])] 
df

df_new=df.drop(df.index[[221]],axis=0).reset_index(drop=True)
df_new

while tm.rsquared < 0.90:
    for cooks_distance in [no.max(cooks_distance)>0.5]:
        model=sma.OLS(Y,X).fit()
        (cooks_distance,_)=model.get_influence().cooks_distance
        cooks_distance
        no.argmax(cooks_distance) , no.max(cooks_distance)
        df_new=df_new.drop(df_new.index[[no.argmax(cooks_distance)]],axis=0).reset_index(drop=True)
        df_new
    else:
        final_model=sma.OLS(Y,X).fit()
        final_model.rsquared , final_model.aic
        print("Thus model accuracy is improved to",final_model.rsquared)

# so finally model accuracy is improved to 98%

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12,8))
fig = sma.graphics.plot_regress_exog(tm, 'Weight',fig=fig)
fig

fig = sma.graphics.plot_regress_exog(tm, 'Gears',fig=fig)
fig

fig = sma.graphics.plot_regress_exog(tm, 'HP',fig=fig)
fig

