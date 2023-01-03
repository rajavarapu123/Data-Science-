# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 12:00:34 2022

@author: Anil Kumar
"""

import pandas as pd
import numpy as np
df = pd.read_csv("forestfires.csv")
df.info()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['month'] = le.fit_transform(df['month'])
df['day'] = le.fit_transform(df['day'])

X = df.iloc[:,0:30]
Y = df['size_category']

import matplotlib.pyplot as plt
plt.scatter(X.iloc[:,1],X.iloc[:,3])
plt.scatter(X.iloc[:,8],X.iloc[:,10])
plt.scatter(X.iloc[:,15],X.iloc[:,20])
plt.show()

import seaborn as sns
sns.histplot(X.iloc[:,3]) # Univarite plot
sns.countplot(X['temp'])

plt.figure(figsize=(12,8)) # Bivariate plot
sns.barplot(X.iloc[:,6], Y)
sns.barplot(X.iloc[:,9], Y)


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=(4))

from sklearn.svm import SVC
clf = SVC(kernel='linear',C=1.0) #got the best accuracy
#sv = SVC(kernel='rbf',gamma=1)
#sv = SVC(kernel='poly',gamma=2)

clf.fit(X_train, Y_train)

Y_pred_train = clf.predict(X_train)
Y_pred_test = clf.predict(X_test)

from sklearn.metrics import accuracy_score

print('train accuracy', accuracy_score(Y_train,Y_pred_train))
print('test acuuracy', accuracy_score(Y_test,Y_pred_test))

# plotting the visualisation

pip install mlxtend

from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
Y = LabelEncoder.fit_transform(df['size_category'],Y)
Y = pd.DataFrame(Y)
Y = np.array(Y)
X = np.array(df.iloc[:,0:3])

from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X=X.values, 
                      y=Y.values,
                      clf=clf, 
                      legend=2)










