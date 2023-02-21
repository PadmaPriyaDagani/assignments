# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 11:43:33 2022

@author: hp
"""
#Prepare a model for glass classification using KNN

import pandas as pd
df=pd.read_csv("glass.csv")
df

df.head()
df.tail()
df.shape
df.info()

df.Type.value_counts()

#visualisaton
cor=df.corr()
cor
import seaborn as sns
sns.heatmap(cor)
#We can notice that Ca and K values don't affect Type that much so we can drop them
#Also Ca and RI are highly correlated, this means using only RI is enough.

import matplotlib.pyplot as plt
sns.pairplot(df)
plt.show()

#The pairplot shows that the data is not linear and KNN can be applied to get nearest neighbors and classify the glass types
#Using standard scaler we can scale down to unit variance.

data=df.drop(['Ca','K'],axis=1) # Removing Ca,K
data

#Split the data
X=data.iloc[:,0:7]
Y=data["Type"]

# standardization
from sklearn.preprocessing import StandardScaler
X_scale = StandardScaler().fit_transform(X)
X_scale
type(X_scale)

pd.crosstab(Y,Y)

from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X_scale, Y,stratify=Y,test_size=0.3,random_state=50)  # By default test_size=0.25

pd.crosstab(Y_train,Y_train)

# Install KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3, p=2) # k =5 # p=2 --> Eucledian distance
knn.fit(X_train, Y_train)

# Prediction
Y_pred_train=knn.predict(X_train)
Y_pred_test=knn.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(Y_train, Y_pred_train)
#0.825503355704698
accuracy_score(Y_test, Y_pred_test)
#0.825503355704698

#test accuracy

#KFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

kfold=KFold(n_splits=5,shuffle=True,random_state=5)
kmodel=KNeighborsClassifier(n_neighbors=3)
results=cross_val_score(kmodel,X,Y,cv=kfold,scoring='accuracy')
results

(results.mean()*100.0).round(2)
#66.8

--------------------------------------------------------------------------------------------
#Implement a KNN model to classify the animals in to categorie

zoo=pd.read_csv("Zoo.csv")
zoo

zoo.head()
zoo.tail()
zoo.info()
zoo.shape

zoo.type.value_counts()

#visualisaton
cor=zoo.corr()
cor
import seaborn as sns
sns.heatmap(cor)


import matplotlib.pyplot as plt
sns.pairplot(zoo)
plt.show()

zoo.drop('animal name',axis=1,inplace=True)
zoo


#Split the data
X=zoo.iloc[:,0:16]
Y=zoo["type"]

# standardization
from sklearn.preprocessing import StandardScaler
X_scale = StandardScaler().fit_transform(X)
X_scale
type(X_scale)

pd.crosstab(Y,Y)

from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X_scale, Y,stratify=Y,test_size=0.3,random_state=50)  # By default test_size=0.25

pd.crosstab(Y_train,Y_train)

# Install KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3, p=2) # k =5 # p=2 --> Eucledian distance
knn.fit(X_train, Y_train)

# Prediction
Y_pred_train=knn.predict(X_train)
Y_pred_test=knn.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(Y_train, Y_pred_train)
#0.9714285714285714
accuracy_score(Y_test, Y_pred_test)
#0.967741935483871

#test accuracy

#KFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

kfold=KFold(n_splits=5,shuffle=True,random_state=5)
kmodel=KNeighborsClassifier(n_neighbors=3)
results=cross_val_score(kmodel,X,Y,cv=kfold,scoring='accuracy')
results

(results.mean()*100.0).round(2)
# 92.0



















