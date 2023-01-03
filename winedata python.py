# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 17:43:25 2022

@author: Anil Kumar
"""

import pandas as pd
import numpy as np
df = pd.read_csv('wine.csv')
df.info()

from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
X = df.iloc[:,1:]
X_scale = SS.fit_transform(X)

from sklearn.decomposition import PCA
pca = PCA(n_components=(3))
pc = pca.fit_transform(X_scale)
pc_df = pd.DataFrame(data=pc,columns=['pc1','pc2','pc3'])
pc_df

import matplotlib.pyplot as plt
plt.scatter(pc_df['pc1'],pc_df['pc2'])
plt.scatter(pc_df['pc2'],pc_df['pc3'])

### Hierarchy clustering - agglomaritive clustering

import scipy.cluster.hierarchy as shc
dend = shc.dendrogram(shc.linkage(pc_df,method='complete'))

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='single')
cluster = AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='complete')
Y = cluster.fit_predict(pc_df)
Y = pd.DataFrame(Y)
Y.value_counts()

import matplotlib.pyplot as plt
plt.scatter(pc_df['pc1'],pc_df['pc2'],c=cluster.labels_, cmap='rainbow')

### K means clustering

from sklearn.cluster import KMeans
km = KMeans(n_clusters=4,n_init=20)
Y=km.fit_predict(pc_df)
Y = pd.DataFrame(Y)
Y.value_counts()
km.inertia_

kint_list=[]
for i in range(2,10):
    km = KMeans(n_clusters=i,n_init=20)
    Y=km.fit_predict(pc_df)
    kint_list.append(km.inertia_)
print(kint_list)

#elbow cure
import matplotlib.pyplot as plt
plt.scatter(range(2,10),kint_list)
plt.plot(range(2,10),kint_list)

### DB scan clustering

from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=2 , min_samples=3) #eps = epsalon = radius
dbscan.fit(X_scale)
y=dbscan.labels_
Y=pd.DataFrame(y)
Y.value_counts()


# So, KMeans cluster gives the max clusters of 3 with the good varience among them.
