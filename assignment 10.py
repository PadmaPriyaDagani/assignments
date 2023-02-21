# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 14:14:14 2022

@author: hp
"""
#Problem statement.

#Build a recommender system by using cosine simillarties score.


import pandas as pd
df=pd.read_csv("books.csv",encoding="latin1")
df

df.head()
df.info()
df.shape
df.describe()

df.isnull().sum()
df.duplicated().sum()
df.columns.valuesdf['Book.Title']

df.sort_values('User.ID')

#number of unique users
len(df)

#exclude users with less than 200 ratings and books with less than 100 ratings
user_count=df['User.ID'].value_counts()
book_count=df['Book.Rating'].value_counts()

X=df['User.ID'].value_counts()>200
Y=X[X].index #userID
print(Y.shape)
ratings=df[df['User.ID'].isin(Y)]

#pivot table
book_pivot=df.pivot_table(columns='User.ID',index='Book.Title',values='Book.Rating')
#impute Nans with 0 values
book_pivot.fillna(0,inplace=True) 

#csr matrix
from scipy.sparse import csr_matrix
book_sparse=csr_matrix(book_pivot)

#Nearest Neighbour
from sklearn.neighbors import NearestNeighbors
model=NearestNeighbors(algorithm='brute')
model.fit(book_sparse)

#test algorithm
distance,suggestions=model.kneighbors(book_pivot.iloc[2276,:].values.reshape(1,-1))
                        
for i in range(len(suggestions)):
    print(book_pivot.index[suggestions[i]])
    
# Hence we successfully built a recommend system