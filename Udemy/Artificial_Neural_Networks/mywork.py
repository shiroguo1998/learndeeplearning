#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 22:43:08 2018

@author: shiro
"""

# preparing data
# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# encode LabeEncoder and OnecodeEncoder dataset (category: geography and gender)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X_geo = LabelEncoder() # for geography
labelEncoder_X_gen = LabelEncoder() # for gender

X[:, 1] = labelEncoder_X_geo.fit_transform(X[:, 1])
X[:, 2] = labelEncoder_X_gen.fit_transform(X[:, 2])

onehotEncoder_X_geo = OneHotEncoder(categorical_features = [1])
X = onehotEncoder_X_geo.fit_transform(X).toarray()
X = X[:, 1:] # remove dummy variable trap

# splitting dataset into train test and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# build ANN
# import libraries
import keras
from keras.models import Sequential
from keras.layers import Dense

# model for ANN
classifier = Sequential()

# adding input layer and 1st hidden layer to ANN
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# adding 2nd hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# adding output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# fitting the ANN to Training set
classifier.fit(x = X_train, y = y_train, batch_size = 10, epochs = 100)

# making predictions and test the model on test_set
# make prediction
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)