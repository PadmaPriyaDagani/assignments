# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 17:08:35 2022

@author: hp
"""

#1) Prepare a classification model using SVM for salary data 

pip install scikit learn

import pandas as pd
salary_train=pd.read_csv("SalaryData_Train(1).csv")
salary_train.head()

salary_test=pd.read_csv("SalaryData_Test(1).csv")
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

# view frequency counts of values in categorical variables

for var in categorical: 
    print(salary_train[var].value_counts())

string_columns=['workclass','education','maritalstatus','occupation','relationship','race','sex','native','Salary']

#label encoding
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
for i in string_columns:
    salary_train[i]=LE.fit_transform(salary_train[i])
    salary_test[i]=LE.fit_transform(salary_test[i])

# find numerical variables

numerical = [var for var in salary_train.columns if salary_train[var].dtype!='O']
print('There are {} numerical variables\n'.format(len(numerical)))
print('The numerical variables are :', numerical)

# check missing values in numerical variables
salary_train[numerical].isnull().sum()

#Declare feature vector and target variable 
X = salary_train.drop(['Salary'], axis=1)
Y = salary_train['Salary']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

X_train.shape, X_test.shape
X_train.dtypes
X_test.dtypes

Y_train.dtypes
Y_test.dtypes

# display numerical variables

numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']
numerical

# check missing values in X_train
X_train.isnull().sum()

# check missing values in X_test
X_test.isnull().sum()

#We now have training and testing set ready for model building. Before that, we should map all the feature variables onto the same scale. It is called feature scaling.


df=salary_train
# Importing matlab to plot graphs
import matplotlib as plt
%matplotlib inline

df.groupby('educationno').Salary.mean().plot(kind='bar')
df.groupby('occupation').Salary.mean().plot(kind='bar')
df.groupby('relationship').Salary.mean().plot(kind='bar')
#Higher the education higher the occupation and higher the Salary.


from sklearn.model_selection import train_test_split

# Taking only the features that is important for now
X = df[['educationno', 'occupation']]

# Taking the labels (Income)
Y = df['Salary']

# Spliting into 80% for training set and 20% for testing set so we can see our accuracy
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Importing C-Support Vector Classification from scikit-learn
from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X_train, Y_train)


# Predicting the result and giving the accuracy
score = classifier.score(X_test, Y_test)

print(score)
#0.7763964859937013

#Let's do the correlation map
import seaborn as sns
import matplotlib.pyplot as plt
#correlation matrix
corrmat = df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);

import numpy as np
k = 8 #number of variables for heatmap
cols = corrmat.nlargest(k, 'Salary')['Salary'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
#The first graph show us the relation between each pair of numeric values. 
#The hottier the square, the stronger is the relation between them. 
#The second, show us the 7 features most related to Salary with the relation factor ordered by this factor. 
#We can see by the first graph, for example, that marital status has strong relation with sex but really weak with age.
# We can also see that the classifications we did (race, marital status, occupation) has very little influence in the Salary. 
#That's because it's just classification, not linear values. 
#But these graphs shows us that Age, Hours Per Week and Capital Gain do have some influence in the Salary, 
#let's test that! We will start by switching the occuption by age in our last prediction.

# Taking only the features that is important for now
X = df[['educationno', 'age']]

# Taking the labels (Income)
Y = df['Salary']

# Spliting into 80% for training set and 20% for testing set so we can see our accuracy
X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
# Declaring the SVC with no tunning
classifier = SVC()
classifier.fit(X_train, Y_train)
score = classifier.score(x_test, y_test)

print(score)
#0.7833581965854467
#0.1% Better... Let's see with the three features that the graph show us

# Taking only the features that is important for now
X = df[['educationno', 'age', 'hoursperweek', 'capitalgain']]

# Taking the labels (Income)
Y = df['Salary']

# Spliting into 80% for training set and 20% for testing set so we can see our accuracy
X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

classifier = SVC()
classifier.fit(X_train, Y_train)

score = classifier.score(x_test, y_test)

print(score)
#0.7957898226421349

#What race have the higher income?
#Does women earn less money than men? 
#And in what age do we have more chances to earn more? Let's check it out

df.groupby('race').Salary.mean().plot(kind='bar')
df.groupby('sex').Salary.mean().plot(kind='bar')
df.groupby('age').Salary.mean().plot(kind='bar')

#Conclusion
#Men have more chances to have a higher income
#White and Asian Pacific Islanders have more chances than other races
#Income sort of follows the normal deviation, with a peak at 50 years old
-------------------------------------------------------------------------------------------
#2)classify the Size_Categorie using SVM

import pandas as pd
dataframe = pd.read_csv("forestfires (1).csv")
dataframe

# Encode Data
dataframe.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(1,2,3,4,5,6,7,8,9,10,11,12), inplace=True)
dataframe.day.replace(('mon','tue','wed','thu','fri','sat','sun'),(1,2,3,4,5,6,7), inplace=True)
dataframe.day.replace(('small','large'),(0,1), inplace=True)

dataframe.head()
dataframe.info()
dataframe.shape
dataframe.tail()

#Creating dummy vairables dropping first dummy variable
df=pd.get_dummies(dataframe,columns=['size_category'], drop_first=True)
print(df.head())

dataframe.drop('monthaug',axis='columns', inplace=True)
dataframe.drop('monthdec',axis='columns', inplace=True)
dataframe.drop('monthfeb',axis='columns', inplace=True)
dataframe.drop('monthjan',axis='columns', inplace=True)
dataframe.drop('monthjul',axis='columns', inplace=True)
dataframe.drop('monthjun',axis='columns', inplace=True)
dataframe.drop('monthmar',axis='columns', inplace=True)
dataframe.drop('monthmay',axis='columns', inplace=True)
dataframe.drop('monthnov',axis='columns', inplace=True)
dataframe.drop('monthoct',axis='columns', inplace=True)
dataframe.drop('monthsep',axis='columns', inplace=True)

dataframe.drop('daysat',axis='columns', inplace=True)
dataframe.drop('daysun',axis='columns', inplace=True)
dataframe.drop('daythu',axis='columns', inplace=True)
dataframe.drop('daytue',axis='columns', inplace=True)
dataframe.drop('daywed',axis='columns', inplace=True)
dataframe.drop('monthapr',axis='columns', inplace=True)
dataframe.drop('dayfri',axis='columns', inplace=True)
dataframe.drop('daymon',axis='columns', inplace=True)
#getting information of dataset
dataframe.info()

#Creating dummy vairables dropping first dummy variable
df=pd.get_dummies(dataframe,columns=['size_category'], drop_first=True)
print(df.head())


##Normalising the data as there is scale difference
X=dataframe.iloc[:,0:30]
Y=dataframe['size_category']

def norm_func(i):
    x= (i-i.min())/(i.max()-i.min())
    return (x)

import seaborn as sns
import matplotlib.pyplot as plt
#correlation matrix
corrmat = dataframe.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);

k = 8 #number of variables for heatmap
cols = corrmat.nlargest(k, 'month')['month'].index
cm = np.corrcoef(dataframe[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

#The first graph show us the relation between each pair of numeric values. 
#The second, show us the 7 features most related to month with the relation factor ordered by this factor.
# here FFMC,DMC,DC and temp have more influence in month

# Taking only the features that is important for now
X = dataframe[['FFMC', 'DMC', 'DC', 'temp']]

# Taking the labels (Income)
Y = dataframe['size_category']

# Spliting into 80% for training set and 20% for testing set so we can see our accuracy
X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
model_linear = SVC(kernel = "linear")
model_linear.fit(X_train,Y_train)
pred_test_linear = model_linear.predict(x_test)

np.mean(pred_test_linear==y_test)
# Accuracy = 0.7115384615384616

# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(X_train,Y_train)
pred_test_poly = model_poly.predict(x_test)

np.mean(pred_test_poly==y_test) #Accuacy =  0.7115384615384616

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(X_train,Y_train)
pred_test_rbf = model_rbf.predict(x_test)

np.mean(pred_test_rbf==y_test) #Accuracy =  0.7115384615384616

#'sigmoid'
model_sig = SVC(kernel = "sigmoid")
model_sig.fit(X_train,Y_train)
pred_test_sig = model_rbf.predict(x_test)

np.mean(pred_test_sig==y_test) #Accuracy =  0.7115384615384616
#we got same accurate values for all the types of kernels.