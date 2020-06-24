# -*- coding: utf-8 -*-
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [3])],   # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                                         # Leave the rest of the columns untouched
)
X = ct.fit_transform(X)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Fitting the multiple linear regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


#Predicting Test set results
y_pred = regressor.predict(X_test)


# Building the optimal model using Backward Elimination 
import statsmodels.regression.linear_model as lm
X = np.append(arr=np.ones((50,1)).astype(int), values=X, axis=1)
X_opt = X[:,[0, 1, 2, 3, 4, 5]]
X_opt = np.array(X_opt, dtype=float)
regressor_x = lm.OLS(endog=y, exog=X_opt).fit()
print(regressor_x.summary())
X_opt = X[:,[0, 1, 3, 4, 5]]
X_opt = np.array(X_opt, dtype=float)
regressor_x = lm.OLS(endog=y, exog=X_opt).fit()
print(regressor_x.summary())
X_opt = X[:,[0, 3, 4, 5]]
X_opt = np.array(X_opt, dtype=float)
regressor_x = lm.OLS(endog=y, exog=X_opt).fit()
print(regressor_x.summary())
X_opt = X[:,[0, 3, 5]]
X_opt = np.array(X_opt, dtype=float)
regressor_x = lm.OLS(endog=y, exog=X_opt).fit()
print(regressor_x.summary())
X_opt = X[:,[0, 3]]
X_opt = np.array(X_opt, dtype=float)
regressor_x = lm.OLS(endog=y, exog=X_opt).fit()
print(regressor_x.summary())

# So by the backward elimination, we found that only R&D spend independent variable has a 
# high impact on making the profit
