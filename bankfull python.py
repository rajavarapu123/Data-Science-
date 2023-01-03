# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 21:04:42 2022

@author: Anil Kumar
"""

import pandas as pd
import numpy as np
df = pd.read_csv("bank-full.csv")
df.head()
df.info()

X = df.iloc[:,0:16]
Y = df['y']

X = X.drop(X.columns[[8,9,10,15]],axis=1,inplace=True)

df_continuous = df[df.columns[[0,5,11,12,13,14]]]
df_continuous

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
df1 = ss.fit_transform(df_continuous)
df1 = pd.DataFrame(df1)
df1.columns = ['age','balance','duration','campaign','pdays','previous']
df1

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['job'] = le.fit_transform(df['job'])
df['marital'] = le.fit_transform(df['marital'])
df['education'] = le.fit_transform(df['education'])
df['default'] = le.fit_transform(df['default'])
df['housing'] = le.fit_transform(df['housing'])
df['loan'] = le.fit_transform(df['loan'])
df['y'] = le.fit_transform(df['y'])

X_scale = pd.concat([df1,df['job'],df['marital'],df['education'],df['default'],df['housing'],df['loan']],axis=1,ignore_index=True)
X_scale

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_scale, Y)
Y_pred = lr.predict(X_scale)

from sklearn.metrics import accuracy_score,log_loss,recall_score,f1_score,precision_score

print('accuracy score',accuracy_score(Y,Y_pred).round(3))
print('recall score',recall_score(Y,Y_pred).round(3))
print('f1 score',f1_score(Y,Y_pred).round(3))
print('precision score',precision_score(Y,Y_pred).round(3))

# visualization

X = df['job']
#X = df['marital']
#X = df['education']
Y = df['y']
import seaborn as sns
sns.regplot(x=X,y=Y,data=(df),logistic=True,ci=None)




