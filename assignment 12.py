# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 14:22:17 2022

@author: hp
"""
pip install scikit learn

import pandas as pd
salary_train=pd.read_csv("SalaryData_Train.csv")
salary_train.head()

salary_test=pd.read_csv("SalaryData_Test.csv")
salary_test.head()

salary_train.shape
salary_test.shape

salary_train.info()
salary_test.info()

salary_train.describe()
salary_test.describe()


print(salary_train.columns)
print(salary_test.columns)

#Separating categorical from numerical for train data
categorical = [var for var in salary_train.columns if salary_train[var].dtype=='O']
print('There are {} categorical variables\n'.format(len(categorical)))
print('The categorical variables are :\n\n', categorical)
salary_train[categorical].head()
#checking Null variables
salary_train[categorical].isnull().sum()

#Separating categorical from numerical for test data
categorical = [var for var in salary_test.columns if salary_test[var].dtype=='O']
print('There are {} categorical variables\n'.format(len(categorical)))
print('The categorical variables are :\n\n', categorical)
salary_test[categorical].head()
#checking Null variables
salary_test[categorical].isnull().sum()

string_columns=['workclass','education','maritalstatus','occupation','relationship','race','sex','native','Salary']

#label encoding
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
for i in string_columns:
    salary_train[i]=LE.fit_transform(salary_train[i])
    salary_test[i]=LE.fit_transform(salary_test[i])


#Declare feature vector and target variable train data
X = salary_train.drop(['Salary'], axis=1)
Y = salary_train['Salary']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

X_train.shape, X_test.shape
X_train.dtypes
X_test.dtypes
#declare featur vector and target variable for test data
x = salary_test.drop(['Salary'], axis=1)
y = salary_test['Salary']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

x_train.shape, x_test.shape
x_train.dtypes
x_test.dtypes

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
#FOR TRAINING DATA
gnb.fit(X_train, Y_train)

Y_pred = gnb.predict(X_test)
Y_pred

#FOR TEST DATA
gnb.fit(x_train, y_train)

y_pred = gnb.predict(x_test)
y_pred

#confusion matrix for train data
from sklearn.metrics import confusion_matrix,accuracy_score
CM=confusion_matrix(Y_test,Y_pred)
ACC=accuracy_score(Y_test,Y_pred).round(3)
print("naive bayes model accuracy score for train data",ACC)
#naive bayes model accuracy score for train data 0.794

#confusion matrix and accuracy for test data
cm=confusion_matrix(y_test,y_pred)
acc=accuracy_score(y_test,y_pred).round(3)
print("naive bayes model accuracy score for train data",acc)
#naive bayes model accuracy score for train data 0.795

from sklearn.naive_bayes import MultinomialNB
Mnb=MultinomialNB()
#TRAIN DATA
Mnb.fit(X_train, Y_train)

Y_pred = Mnb.predict(X_test)
Y_pred

#FOR TEST DATA
Mnb.fit(x_train, y_train)

y_pred = Mnb.predict(x_test)
y_pred

#confusion matrix for train data
from sklearn.metrics import confusion_matrix,accuracy_score
CM=confusion_matrix(Y_test,Y_pred)
ACC=accuracy_score(Y_test,Y_pred).round(3)
print("naive bayes model accuracy score for train data",ACC)
#naive bayes model accuracy score for train data 0.773

#confusion matrix and accuracy for test data
cm=confusion_matrix(y_test,y_pred)
acc=accuracy_score(y_test,y_pred).round(3)
print("naive bayes model accuracy score for train data",acc)
#naive bayes model accuracy score for train data 0.772
