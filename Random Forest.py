# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 16:37:40 2022

@author: hp
"""

#Problem Statement:
#A cloth manufacturing company is interested to know about the segment or attributes causes high sale. 
#Approach - A decision tree can be built with target variable Sale (we will first convert it in categorical variable) & all other variable will be independent in the analysis.  

import pandas as pd
df=pd.read_csv("Company_Data.csv")
df

df.head()
df.tail()
df.shape
df.info()

df.isnull().sum()
#No missing values and No ouliers in the dataset
df.corr()


#lABEL ENCODING
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
df['ShelveLoc'] = label_encoder.fit_transform(df['ShelveLoc'])
df['Urban'] = label_encoder.fit_transform(df['Urban'])
df['US'] = label_encoder.fit_transform(df['US'])
df.head()

#Visualisation
import matplotlib.pyplot as plt
df.hist()
plt.figure(figsize =(20,20))

df.boxplot()
plt.figure(figsize =(20,20))

import seaborn as sns
sns.pairplot(data=df,hue='ShelveLoc')
#dropping the unnecessary variables and splitting the data
X=df.iloc[:,0:6]
Y=df['ShelveLoc']

#train test split
from sklearn.model_selection import train_test_split  
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20,random_state=42,stratify=Y) 

#Decision Tree using Entropy
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(criterion='entropy',max_depth=3)
model.fit(X_train,Y_train)

from sklearn import tree
tree.plot_tree(model)

fn=['Sales','CompPrice','Income','Advertising','Population','Price']
cn=['0', '1', '2']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(model,
               feature_names = fn, 
               class_names=cn,
               filled = True);

#Predicting on test data
Y_pred= model.predict(X_test) # predicting on test data set 
pd.Series(Y_pred).value_counts() # getting the count of each category 

#2=Medium    47
#1=Good    17
#0=Bad    16
#accuracy and confusion matrix
from sklearn import metrics
cm=metrics.confusion_matrix(Y_test,Y_pred)
print(cm)
metrics.accuracy_score(Y_test,Y_pred).round(2)
# [[ 8  1 10]
# [ 0 12  5]
# [ 8  4 32]]
#accuracy= 0.65

#Decision tree using gini
model_g=DecisionTreeClassifier(criterion='gini',max_depth=3)
model_g.fit(X_train,Y_train)

from sklearn import tree
tree.plot_tree(model_g)

fn=['Sales','CompPrice','Income','Advertising','Population','Price']
cn=['0', '1', '2']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(model_g,
               feature_names = fn, 
               class_names=cn,
               filled = True);

#Predicting on test data
Y_pred= model_g.predict(X_test) # predicting on test data set 
pd.Series(Y_pred).value_counts() # getting the count of each category 

#2=Medium    56
#1=Good    8
#0=Bad    16
#accuracy and confusion matrix
from sklearn import metrics
cm=metrics.confusion_matrix(Y_test,Y_pred)
print(cm)
metrics.accuracy_score(Y_test,Y_pred).round(2)
# [[ 7  0 12]
# [ 0  7 10]
# [ 9  1 34]]
# accuracy is 0.6

# Entropy gave the best accurate score

#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
RF=RandomForestClassifier(max_features=3,
                       n_estimators=100)

RF.fit(X_train,Y_train)
Y_pred=RF.predict(X_test)
metrics.accuracy_score(Y_test,Y_pred).round(2)
#When we use the model_selection =train_test_split we got the acccuracy as 65% and we try with KFold model_selection

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
kfold =KFold(n_splits=10,shuffle =True,random_state =None)
model1 =RandomForestClassifier(n_estimators=500,max_features =0.6)
results=cross_val_score(model1,X,Y, cv =kfold)

print(results)
#[0.675 0.675 0.5   0.65  0.8   0.7   0.6   0.625 0.675 0.575]

import numpy as np
print(np.mean(results))
#0.6475
-------------------------------------------------------------------------------------------
#Use decision trees to prepare a model on fraud data 
#treating those who have taxable_income <= 30000 as "Risky" and others are "Good"

df=pd.read_csv("Fraud_check.csv")
df

df.head()
df.tail()
df.shape
df.info()

df.isnull().sum()
#no null data or outliers in the data set
df.corr()

##Creating new cols TaxInc and dividing 'Taxable.Income' cols on the basis of [10000,30000,99620] for Risky and Good
df["TaxInc"] = pd.cut(df["Taxable.Income"], bins = [10000,30000,99620], labels = ["Risky", "Good"])

# getting the dummies
df1=pd.get_dummies(df)
df1

df1.drop(df1.columns[[3,5,8,10]],axis=1,inplace=True)
df1

#visualisation
import matplotlib.pyplot as plt
df.hist()
plt.figure(figsize =(20,20))

df.boxplot()
plt.figure(figsize =(20,20))

sns.pairplot(data=df1,hue='TaxInc_Good')

# Normalization function 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)
#normalised data
df_norm=norm_func(df1.iloc[:,1:])
df_norm.tail()

X = df_norm.drop(['TaxInc_Good'], axis=1)
Y = df_norm['TaxInc_Good']

#train test split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20,random_state=42,stratify=Y) 

#Decision Tree using Entropy
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(criterion='entropy',max_depth=3)
model.fit(X_train,Y_train)

from sklearn import tree
tree.plot_tree(model)

fn=['population','experience','Undergrad_YES','Marital.Status_Married','Marital.Status_Single','Urban_YES']
cn=['1', '0']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(model,
               feature_names = fn, 
               class_names=cn,
               filled = True);

#Predicting on test data
Y_pred= model.predict(X_test) # predicting on test data set 

#confusion matrix and accuracy
from sklearn import metrics
cm=metrics.confusion_matrix(Y_test,Y_pred)
print(cm)
metrics.accuracy_score(Y_test,Y_pred).round(2)
#[[ 0 25]
# [ 0 95]]
#accuracy is 0.79

#Decision tree using gini
model_g=DecisionTreeClassifier(criterion='gini',max_depth=3)
model_g.fit(X_train,Y_train)

#plotting the tree
from sklearn import tree
tree.plot_tree(model_g)

fn=['population','experience','Undergrad_YES','Marital.Status_Married','Marital.Status_Single','Urban_YES']
cn=['1', '0']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(model_g,
               feature_names = fn, 
               class_names=cn,
               filled = True);

#Predicting on test data
Y_pred= model_g.predict(X_test) # predicting on test data set 

#accuracy and confusion matrix
from sklearn import metrics
cm=metrics.confusion_matrix(Y_test,Y_pred)
print(cm)
metrics.accuracy_score(Y_test,Y_pred).round(2)
# [[ 0 25]
#[ 0 95]]
# 0.79

# both entropy and gini are equal in accuracy score

#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
RF=RandomForestClassifier(max_features=3,
                       n_estimators=100)

RF.fit(X_train,Y_train)
Y_pred=RF.predict(X_test)
metrics.accuracy_score(Y_test,Y_pred).round(2)
#When we use the model_selection =train_test_split we got the acccuracy as 74% and we try with KFold model_selection

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
kfold =KFold(n_splits=10,shuffle =True,random_state =None)
model1 =RandomForestClassifier(n_estimators=500,max_features =0.6)
results=cross_val_score(model1,X,Y, cv =kfold)

print(results)
#[0.83333333 0.7        0.78333333 0.76666667 0.65       0.81666667
 0.68333333 0.76666667 0.73333333 0.68333333]

import numpy as np
print(np.mean(results))
#0.7416666666666667