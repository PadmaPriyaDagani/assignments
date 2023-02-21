# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 15:24:58 2022

@author: hp
"""

#1)Prepare a prediction model for profit of 50_startups data.
#Do transformations for getting better predictions of profit and
#make a table containing R^2 value for each prepared model.
import pandas as pd
df=pd.read_csv("50_Startups.csv")
df

df.shape
df.head()
list(df)
df.isnull().sum()

#correlation
df.corr()
#boxplot
import seaborn as sns
sns.boxplot(x='State',y='Profit',data=df)
plt.show()
#All the outliers are in New York
# California has the maximum profit and maximum loss

# splitting the data
X=df.iloc[:,:-1].values
X.shape
Y=df["Profit"].values

#label encoding 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder()
X = onehotencoder.fit_transform(X).toarray()
##avoiding dummy variables
X = X[:, 1:]
#train test split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.9,random_state=0)
X_train

from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X_train,Y_train)

LR.intercept_
#100373.64744855961
LR.coef_

#predicting the values
Y_pred=LR.predict(X_test)
Y_pred
#mean squared error, r2 square

from sklearn.metrics import  mean_squared_error,r2_score
mse=mean_squared_error(Y_pred,Y_test)*100
import numpy as np
Rmse=np.sqrt(mse)
print("Root mean squared error is",round(2))

r2Score = r2_score(Y_pred, Y_test)
print("R2 score of model is :" ,r2Score*100)
#Root mean squared error is 2
#R2 score of model is : -284.08709795192345

-------------------------------------------------------------------------------------------
#2)Consider only the below columns and prepare a prediction model for predicting Price.

#Corolla<-Corolla[c("Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight")]

import pandas as pd
df=pd.read_csv("ToyotaCorolla.csv",encoding="ISO-8859-1")
df

df.shape
df.head()
df.columns
list(df)
df.corr()
pd.set_option("display.max_columns",20)
df

data=df[["Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]
data.corr()
#boxplot
data.boxplot(column=["Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"])
#maximum outliers are in KM
#splitting the data
X=df[["Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]
Y=df[["Price"]]
X.head()

#scatterplot
import matplotlib.pyplot as plt
plt.scatter(X.iloc[:,0],Y,color='blue')
plt.show()

import matplotlib.pyplot as plt
plt.scatter(X.iloc[:,1],Y,color='black')
plt.show()

import matplotlib.pyplot as plt
plt.scatter(X.iloc[:,2],Y,color='brown')
plt.show()

import matplotlib.pyplot as plt
plt.scatter(X.iloc[:,3],Y,color='green')
plt.show()

import matplotlib.pyplot as plt
plt.scatter(X.iloc[:,4],Y,color='orange')
plt.show()

import matplotlib.pyplot as plt
plt.scatter(X.iloc[:,5],Y,color='yellow')
plt.show()

import matplotlib.pyplot as plt
plt.scatter(X.iloc[:,6],Y,color='red')
plt.show()

import matplotlib.pyplot as plt
plt.scatter(X.iloc[:,7],Y,color='violet')
plt.show()

#train test split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.5,random_state=0)
X_train


from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X_train,Y_train)

LR.intercept_
#-22346.6111538
LR.coef_
#-1.08807004e+02, -1.60979767e-02,  1.23188806e+01,
 #       -2.52343294e+00, -2.39473757e+02,  5.59125213e+02,
 #       -6.11560300e+00,  3.89727766e+01
 
#predicting the values
Y_pred=LR.predict(X_test)
Y_pred

from sklearn.metrics import  mean_squared_error,r2_score
mse=mean_squared_error(Y_pred,Y_test)*100
import numpy as np
Rmse=np.sqrt(mse)
print("Root mean squared error is",round(2))

r2Score = r2_score(Y_pred, Y_test)
print("R2 score of model is :" ,r2Score*100)
#Root mean squared error is 2
#R2 score of model is : 64.52208906743574
#Vif
Vif=1/(1-r2Score)
print("Vif is:",Vif)
#Vif is: 2.8186552525620265