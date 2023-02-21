# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 21:59:18 2022

@author: hp
"""

#PREDICT THE BURNED AREA OF FOREST FIRES WITH NEURAL NETWORKS

import pandas as pd
df=pd.read_csv("forestfires.csv")
df

from keras.models import Sequential
from keras.layers import Dense, Activation,Layer,Lambda

import numpy as np 

df.info()


df.drop(["dayfri","daymon","daysat","daysun","daythu","daytue","daywed","monthapr","monthaug",
         "monthdec","monthfeb","monthjan","monthjul","monthjun","monthmar","monthmay",
         "monthoct","monthnov","monthsep",],axis=1,inplace = True)
df

df.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(1,2,3,4,5,6,7,8,9,10,11,12), inplace=True)
df.day.replace(('mon','tue','wed','thu','fri','sat','sun'),(1,2,3,4,5,6,7), inplace=True)

print("Head:", df.head())

df["size_category"].value_counts()
df.isnull().sum()
df.describe()

##I am taking small as 0 and large as 1
df.loc[df["size_category"]=='small','size_category']=0
df.loc[df["size_category"]=='large','size_category']=1
df["size_category"].value_counts()

df.describe()

df.shape

df.corr()

#Data visualisation
import matplotlib.pyplot as plt
import seaborn as sns
 
df.hist()

df.boxplot()

df.skew()

sns.pairplot(df)

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(df.corr(), vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,12,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(df.columns)
ax.set_yticklabels(df.columns)



#The distributions of rain and area are too skewed and have large outliers so we will scale it to even out the distribution.

# natural logarithm scaling (+1 to prevent errors at 0)
import numpy as np
df.loc[:, ['rain', 'area']] = df.loc[:, ['rain', 'area']].apply(lambda x: np.log(x + 1), axis = 1)

fig, ax = plt.subplots(2, figsize = (5, 8))
ax[0].hist(df['rain'])
ax[0].title.set_text('histogram of rain')
ax[1].hist(df['area'])
ax[1].title.set_text('histogram of area')

#The distribution for rain is not good but the distribution for areais highly improved.
# Now we scale the entire dataset.

#Train test Split
from sklearn.model_selection import train_test_split
X = df.drop(['size_category'], axis = 1).astype(np.float32)
Y= df['size_category'].values.reshape(-1, 1).astype(np.float32)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 42)

#Apply the feature scaling : the standardscaler to the data

# fitting scaler
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
# transforming features
X_test = SS.fit_transform(X_test)
X_train = SS.transform(X_train)
# features
X_test = pd.DataFrame(X_test, columns =X.columns)
X_train = pd.DataFrame(X_train, columns = X.columns)
# labels
Y_test = pd.DataFrame(Y_test, columns = ['size_category'])
Y_train = pd.DataFrame(Y_train, columns = ['size_category'])
X_train.head()


# Hyperparameter
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model

model = Sequential([layers.Input((X_train[0].shape))])
# input layer + 1st hidden layer
model.add(Dense(6, input_dim=13, activation='relu'))
# 2nd hidden layer
model.add(Dense(6, activation='relu'))
# output layer
model.add(Dense(6, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(1, activation = 'relu'))
model.summary()


# Compile Model
model.compile(optimizer = 'adam', metrics=['accuracy'], loss ='binary_crossentropy')
# Train Model

history = model.fit(X_train, Y_train, validation_data = (X_test, Y_test), batch_size = 10, epochs = 100)


_,train_acc = model.evaluate(X_train, Y_train, verbose=0)
_, valid_acc = model.evaluate(X_test, Y_test, verbose=0)
print('Train: %.3f, Valid: %.3f' % (train_acc, valid_acc))
#Train: 0.947, Valid: 0.962
#the accuracy score of the train data is 95% and the accuracy score of the valid or the test data is 96.

plt.figure(figsize=[8,5])
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Valid')
plt.legend()
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Accuracy Curves Epoch 100, Batch Size 10', fontsize=16)
plt.show()
#Based on the output of the accuracy graph, the model begins to show the stability at epochs 60 to 100.





