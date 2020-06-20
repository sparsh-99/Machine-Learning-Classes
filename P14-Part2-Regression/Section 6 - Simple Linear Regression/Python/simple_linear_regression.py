# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 23:34:12 2020

@author: sparsh
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

"""# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Fitting SLR into Training sets
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color='red')    # that will plot all the real values i.e observation values
plt.plot(X_train, regressor.predict(X_train))  # we want predictions of the training set on the same observations
plt.title('Salary vs Experience: Training Set')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color='red')    # that will plot all the real values i.e observation values in test set
plt.plot(X_train, regressor.predict(X_train))  # we want predictions of the training set on the same observations because this is the unique regression line that our ML model trained on and fits in line 30
plt.title('Salary vs Experience: Test Set')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()