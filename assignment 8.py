# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 14:50:58 2022

@author: hp
"""
#Perform Principal component analysis and perform clustering using first 
#3 principal component scores (both heirarchial and k mean clustering(scree plot or elbow curve) and obtain 
#optimum number of clusters and check whether we have obtained same number of clusters with the original data 
#(class column we have ignored at the begining who shows it has 3 clusters)df


import pandas as pd
df=pd.read_csv("wine.csv")
df

df.head()
df.tail()
df.describe()
df.shape
df.value_counts()

#considering onlu numerical data
wine=df.iloc[:,1:]
wine

#converting to numpy
wine_new=wine.values
wine_new

#normalizing the data
from sklearn.preprocessing import scale
wine_norm=scale(wine_new)
wine_norm

wine.describe()
wine.shape
wine.corr()
wine.info()

#visuaization
import seaborn as sns
sns.pairplot(wine)

#PCA implementation
from sklearn.decomposition import PCA
pca=PCA()
pca_values=pca.fit_transform(wine_norm)
pca_values

#variance
var=pca.explained_variance_ratio_
var
#cummulative variance
import numpy as np
var1=np.cumsum(np.round(var,decimals=4)*100)
var1
#variance plot
import matplotlib.pyplot as plt
plt.plot(var1,color='red')


finaldf=pd.concat([df['Type'],pd.DataFrame(pca_values[:,0:3],columns=['PCA1','PCA2','PCA3'])],axis=1)
finaldf

#visualization od PCA'S
fig=plt.figure(figsize=(16,12))
sns.scatterplot(data=finaldf)

#hierarchical clustering
#as we already normalized the data create dendrogram
import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10,8))
plt.title("PCA dendogram")
dend=shc.dendrogram(shc.linkage(wine_norm,method='complete'))

#create clusters
from sklearn.cluster import AgglomerativeClustering
cluster=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='complete')
h_cluster=cluster.fit_predict(wine_norm)

Y=pd.DataFrame(h_cluster)
Y[0].value_counts()

wine['h_clusters']=cluster.labels_
wine

#KMeans
#as we already normalised the data prform elbow method
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 13):
    kmeans = KMeans(n_clusters=i,random_state=2)
    kmeans.fit(wine)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 13), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

clusters_new=KMeans(3,random_state=32)
clusters_new.fit(wine_norm)

clusters_new.labels_

#assign clusters to the data set
wine['k_clusters']=clusters_new.labels_
wine
