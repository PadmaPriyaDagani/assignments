# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 11:33:55 2022

@author: hp
"""

Perform clustering (hierarchical,K means clustering and DBSCAN) for the airlines data to obtain optimum number of clusters. 
Draw the inferences from the clusters obtained.

import pandas as pd
import numpy as np
df=pd.read_csv("EastWestAirlines.csv")
df

df.head()
df.tail()
df.info()
df.describe()

df.boxplot()
#balance has more outliers
df.plot.hist()
 
df.isnull().sum()

import seaborn as sns
sns.pairplot(df)

df.corr()

df.drop(['ID#','Award?'],axis=1,inplace=True)
array=df.values
array


#Normalization
from sklearn.preprocessing import StandardScaler
stscaler = StandardScaler().fit(array)
X = stscaler.transform(array)
X


#hierarical clustering
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 7))  
plt.title("Dendograms")  
dend = shc.dendrogram(shc.linkage(X, method='complete')) 

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='complete')
cluster_labels = cluster.fit_predict(array)

cluster_labels_new = pd.DataFrame(cluster_labels)
cluster_labels_new[0].value_counts()

df['h_clusters']=cluster.labels_
df

#K-Means
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3,random_state=0)
kmeans.fit(X)
kmeans.inertia_
#27558.765175681914

#elbow method
inertia = []
for i in range(1, 11):
    km = KMeans(n_clusters=i,random_state=0)
    km.fit(X)
    inertia.append(km.inertia_)
    
plt.plot(range(1, 11), inertia)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('inertia')
plt.show()

#Build Cluster algorithm
from sklearn.cluster import KMeans
clusters_new = KMeans(n_clusters=5)
clusters_new.fit(X)


clusters_new.labels_

clusters_new.cluster_centers_
#DBSCAN 
from sklearn.cluster import DBSCAN
DBSCAN()
dbscan = DBSCAN(eps=4, min_samples=20)
dbscan.fit(X)

#Noisy samples are given the label -1.
dbscan.labels_

cl=pd.DataFrame(dbscan.labels_,columns=['cluster'])
cl
cl['cluster'].value_counts()
 #0    3872
#-1     102
 #1      25


clustered = pd.concat([df,cl],axis=1)

noisedata = clustered[clustered['cluster']==-1]
finaldata = clustered[clustered['cluster']==0]

clustered.mean()
finaldata.mean()

#based on the bove clusters, clusters5 for KMeans ans 3 gave the accurate result
#complete linkage is ued to identify the clusters rather than single and average
#we got the final data by dropping the noise data or outliers.
------------------------------------------------------------------------------------------------------
#Perform Clustering(Hierarchical, Kmeans & DBSCAN) for the crime data and identify the number of clusters formed and draw inferences.

import pandas as pd
import numpy as np
df=pd.read_csv("crime_data.csv")

df.head()
df.tail()
df.info()
df.describe()

df.boxplot()
#Rape has outliers.
df.plot.hist()
 
df.isnull().sum()

import seaborn as sns
sns.pairplot(df)

df.corr()
#Normalization
from sklearn.preprocessing import StandardScaler
stscaler = StandardScaler().fit(array)
X1 = stscaler.transform(array)
X1

#hierarical clustering
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 7))  
plt.title("Crime data Dendograms")  
dend_C = shc.dendrogram(shc.linkage(X, method='average')) 

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='average')
cluster_labels = cluster.fit_predict(array)

cluster_labels_new = pd.DataFrame(cluster_labels)
cluster_labels_new[0].value_counts()

cluster.labels_

#KMeans clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3,random_state=0)
kmeans.fit(X1)
kmeans.inertia_
#27558.765175681914

#elbow method
inertia = []
for i in range(1, 11):
    km = KMeans(n_clusters=i,random_state=0)
    km.fit(X1)
    inertia.append(km.inertia_)
    
plt.plot(range(1, 11), inertia)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('inertia')
plt.show()

#Build Cluster algorithm
from sklearn.cluster import KMeans
clusters_new = KMeans(n_clusters=6)
clusters_new.fit(X1)
#KMeans(n_clusters=6)

clusters_new.labels_

clusters_new.cluster_centers_
#DBSCAN
from sklearn.cluster import DBSCAN
DBSCAN()
dbscan = DBSCAN(eps=4, min_samples=20)
dbscan.fit(X1)

#Noisy samples are given the label -1.
dbscan.labels_

cl=pd.DataFrame(dbscan.labels_,columns=['cluster'])
cl
cl['cluster'].value_counts()
 #0    3914
#-1      59
#1      26



clustered = pd.concat([df,cl],axis=1)

noisedata = clustered[clustered['cluster']==-1]
finaldata = clustered[clustered['cluster']==0]

clustered.mean()
finaldata.mean()
#in hierarical 3 clusters gave teh accurate output and i Kmeans 6 clusters gave the accurate output.
#Average linkage is used.
