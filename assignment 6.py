# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 22:46:40 2022

@author: hp
"""
 #Whether the client has subscribed a term deposit or not 
#Binomial ("yes" or "no")
import pandas as pd
df=pd.read_csv("bank-full.csv")
df

df.head()
df.shape
list(df)

pd.set_option("display.max_columns",20)
#null entries
df.isnull().sum()

columns=['age','balance','duration','campaign','y']
df_sel=df[columns]
df_sel.info()

pd.crosstab(df_sel.age,df_sel.y).plot()
#from the above graph it is shown that the age group of 20 t0 60 has more rejection of applicationsand 60 to 90 almost everybody is rejeted.

#boxplot
import seaborn as sns
sns.boxplot(data=df_sel)

#outcome
df_sel['outcome']=df_sel.y.map({'no':0,'yes':1})
df_sel

sns.boxplot(data=df_sel) # to see the outcome
#splitting the data
X=df_sel[['age','balance','duration','campaign']]
Y=df_sel['outcome']

from sklearn.linear_model import LogisticRegression
logR=LogisticRegression()
logR.fit(X,Y)

Y_pred=logR.predict(X)
Y_pred

#confusion matix
from sklearn.metrics import confusion_matrix,accuracy_score
CM=confusion_matrix(Y,Y_pred)
#[39342,   580],
#[ 4435,   854]
acc=accuracy_score(Y,Y_pred)*100
print("",acc.round(3))
#88.908

import matplotlib.pyplot as plt
plt.matshow(CM)
plt.title('Confusion matrix')
sns.heatmap(CM)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

from sklearn.metrics import roc_curve,roc_auc_score
logR.predict_proba(X)
logR.predict_proba(X).shape
logR.predict_proba(X)[:,1]

Y_predict_proba=logR.predict_proba(X)[:,1]
fpr,tpr,_=roc_curve(Y,Y_predict_proba)
plt.plot(fpr,tpr)

plt.ylabel('tpr-True positive rate')
plt.xlabel('fpr-False Positive Rate')
plt.show()

auc=roc_auc_score(Y,Y_predict_proba)
(auc*100).round(3)
#81.49
