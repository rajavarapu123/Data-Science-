# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 18:05:10 2022

@author: Anil Kumar
"""

import pandas as pd
import numpy as np
df = pd.read_csv('Fraud_check.csv')
df.info()
df['Taxable.Income']

Y_new=[]
for i in df["Taxable.Income"]:
    if i<=30000:
        Y_new.append("Risky")
    else:
        Y_new.append("Good")

Y_new = pd.DataFrame(Y_new)
df_new = pd.concat([df,Y_new],axis=1)
df_new.drop(df.columns[[2]],axis=1,inplace=True)
df_new

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_new['Undergrad'] = le.fit_transform(df_new['Undergrad'])
df_new['Marital.Status'] = le.fit_transform(df_new['Marital.Status'])
df_new['Urban'] = le.fit_transform(df_new['Urban'])

X = df_new.iloc[:,0:4]
Y = Y_new

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X = ss.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.4,random_state=(6))

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(max_depth=(4))

dtc.fit(X_train, Y_train)

Y_pred_train = dtc.predict(X_train)
Y_pred_test = dtc.predict(X_test)

from sklearn.metrics import accuracy_score

print('train accuracy', accuracy_score(Y_train,Y_pred_train).round(3))
print('test accuracy', accuracy_score(Y_test,Y_pred_test).round(3))


from sklearn.ensemble import BaggingClassifier
bc = BaggingClassifier(base_estimator=(dtc),n_estimators=300,max_features=0.7,max_samples=0.6)
bc.fit(X_train, Y_train)

Y_pred_train = dtc.predict(X_train)
Y_pred_test = dtc.predict(X_test)

from sklearn.metrics import accuracy_score

print('train accuracy', accuracy_score(Y_train,Y_pred_train).round(3))
print('test accuracy', accuracy_score(Y_test,Y_pred_test).round(3))

train_error=[]
test_error=[]

for i in range(1,350):
    bc = BaggingClassifier(base_estimator=(dtc),n_estimators=300,max_features=0.7,max_samples=0.6)
    bc.fit(X_train, Y_train)
    Y_pred_train = dtc.predict(X_train)
    Y_pred_test = dtc.predict(X_test)
    train_error.append(accuracy_score(Y_train,Y_pred_train).round(3))
    test_error.append(accuracy_score(Y_test,Y_pred_test).round(3))
    
np.mean(train_error)
np.mean(test_error)

# hence after doing multiple iterations also variation among the train and test error is more.


