# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 12:59:51 2022

@author: hp
"""

#1) Delivery_time -> Predict delivery time using sorting time 
import pandas as pd
import seaborn as sns
df=pd.read_csv("delivery_time.csv")
df

X=df["Sorting Time"]
X.ndim

import numpy as np
X=np.c_[X]
X.ndim

Y=df["Delivery Time"]
Y.ndim

import numpy as np
Y=np.c_[Y]
Y.ndim

#boxplot
df.boxplot(column=["Sorting Time"])
#The observations for Sorting Time lies approximately between 4 to 8
#It is symmetric and skewed
# The median is around 6.
sns.distplot(df["Sorting Time"])

df.boxplot(column=["Delivery Time"])
#The observations are nearly between 13 to 20
#The data is left skewed
# The median is approximately between 17 to 19
sns.distplot(df["Delivery Time"])

#scatterplot
import matplotlib.pyplot as plt
plt.scatter(X,Y,color='black')
plt.show()

df.corr()
import numpy as np
Y=np.c_[Y]
Y.ndim

#Linear Regression,model fitting
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X,Y)

LR.intercept_
#B0= array([6.58273397])
LR.coef_
#B1=array([[1.6490199]])
#Y=C+(m)X
#6.58273397+(1.6490199)*Sorting Time

#Predicting the values
Y_pred=LR.predict(X)

import matplotlib.pyplot as plt
plt.scatter(X,Y,color='black')
plt.scatter(X,Y_pred,color='Red')
plt.show()

#calculate error
Y-Y_pred

import matplotlib.pyplot as plt
plt.scatter(X,Y,color='black')
plt.plot(X,Y_pred,color='Red')
plt.show()

#calculating mean squared error
from sklearn.metrics importimport matplotlib.pyplot as plt
plt.scatter(X,Y,color='black')
plt.scatter(X,Y_pred,color='Red')
plt.show()

#calculating mean squared error
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y,Y_pred)
mse
#mse=7.793311548584062
import numpy as np
Rmse=np.sqrt(mse)
print("Root Mean Squared error of the above model is",Rmse.round(2))
#Root Mean Squared error of the above model is 2.79


#2)Salary_hike -> Build a prediction model for Salary_hike
import pandas as pd
import seaborn as sns
df=pd.read_csv("Salary_Data.csv")
df

X=df["YearsExperience"]

Y=df["Salary"]


#boxplot
df.boxplot(column=["YearsExperience"])
# The observations are between 3 to 8
# The data is right skewed
#The median is approximately between 4 to 6
sns.distplot(df["YearsExperience"])

df.boxplot(column=["Salary"])
#The observations are between 5000- 11000
#The data is right skewed
#The median is approximately between 6000-7000
sns.distplot(df["Salary"])

#scatterplot
import matplotlib.pyplot as plt
plt.scatter(X,Y,color='black')
plt.show()
 
df.corr()
import numpy as np
X=np.c_[X]
Y=np.c_[Y]
#model fitting
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X,Y)

LR.intercept_
#B0=25792.20019867
LR.coef_
#B1=9449.96232146
#Y=C+mX
#25792.20019867+(9449.96232146)*YearsExperience

#Predicting values
Y_pred=LR.predict(X)

import matplotlib.pyplot as plt
plt.scatter(X,Y,color='black')
plt.scatter(X,Y_pred,color='Red')
plt.show()

#calculating error
Y-Y_pred

import matplotlib.pyplot as plt
plt.scatter(X,Y,color='black')
plt.plot(X,Y_pred,color='Red')
plt.show()

from sklearn.metrics import mean_squared_error
mse= mean_squared_error(Y,Y_pred)
mse
#mse=31270951.722280957
Rmse=np.sqrt(mse)
print("Root Mean Squared error of above model is",Rmse.round(2))
#Root Mean Squared error of above model is 5592.04


