#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 12:37:22 2023

@author: anson
"""

# IN3062 Introduction to Artificial Intelligence
# Coursework
# Multiple Linear Regression

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Load dataset into original_df
original_df = pd.read_csv('immo_data.csv')

# function to check original_df_info
def original_df_info():
    # overview of original_df
    print('overview of original_df:')
    print(original_df.head(5), '\n')
    
    # data types of original_df
    print('data types of original_df:')
    print(original_df.dtypes, '\n')
    
    # missing values original_df
    print('missing values of original_df:')
    print(original_df.isnull().sum(), '\n')
    
#original_df_info()

# create numerical_df with only numerical and boolean columns
numerical_df = original_df.copy()
numerical_df = pd.concat([original_df.select_dtypes(include=np.number),
                          original_df.select_dtypes(include=bool).astype(int)],
                         axis=1)


# function to check numerical_df_info
def numerical_df_info():
    # overview of numerical_df
    print('overview of numerical_df:')
    print(numerical_df.head(5), '\n')
    
    # data types of numerical_df
    print('data types of numerical_df:')
    print(numerical_df.dtypes, '\n')
    
    # missing values of numerical_df
    print('missing values of numerical_df:')
    print(numerical_df.isnull().sum(), '\n')

# numerical_df_info() before removing columns
#numerical_df_info()

# remove columns with irrelevant features
numerical_df = numerical_df.drop(columns = ['scoutId', 'yearConstructedRange', 'baseRentRange', 'geo_plz',
                                 'noRoomsRange', 'livingSpaceRange'])

# remove columns with too many na values
numerical_df = numerical_df.drop(columns = ['telekomHybridUploadSpeed', 'noParkSpaces', 'thermalChar',
                                 'heatingCosts', 'lastRefurbish', 'electricityBasePrice',
                                 'electricityKwhPrice'])

# numerical_df_info() after removing columns
#numerical_df_info()

# drop na rows
numerical_df = numerical_df.dropna()

# updating the n/a values to the mean of the entire column
# Calculate the mean of each column
#column_means = numerical_df.mean()

# Fill missing values with the mean of their respective columns
#numerical_df = numerical_df.fillna(column_means)

# numerical_df_info() after dropping na rows
#numerical_df_info()

#-----

# Multiple Linear Regression
# define X, baseRent
X = numerical_df[['serviceCharge', 'picturecount', 'pricetrend', 'telekomUploadSpeed', #'yearConstructed',
                 'livingSpace', 'noRooms', 'floor', 'numberOfFloors', 'newlyConst',
                 'balcony', 'hasKitchen', 'cellar', 'lift', 'garden']]
baseRent = numerical_df['baseRent']

# split into training and testing data
X_train, X_test, baseRent_train, baseRent_test = train_test_split(X, baseRent, test_size=0.20, random_state=9)

# fit X_train to LR model
LR_baseRent = LinearRegression().fit(X_train, baseRent_train)

# predict values for X_test
baseRent_pred = LR_baseRent.predict(X_test)

# side by side outputs
df_compare = pd.DataFrame({'Actual': baseRent_test, 'Predicted': baseRent_pred})
df_head = df_compare.head(25)
print(df_head)

# RMSE
print('RMSE =', np.sqrt(mean_squared_error(baseRent_test, baseRent_pred)))

# output
print('Coefficient of determination: %.2f' % r2_score(baseRent_test, baseRent_pred))
print('Correlation: ', stats.pearsonr(baseRent_test, baseRent_pred))

#-----

#plot the outputs
#set figure size
plt.rc('figure', figsize=(8, 8))

#plot line down the middle
x = np.linspace(400,0,100)
plt.plot(x, x, '-r',color='r')

#plot the points, prediction versus actual
plt.scatter(baseRent_test, baseRent_pred, color='black')

plt.xlim(-100, 7000)
plt.ylim(-700, 7000)

plt.xticks(())
plt.yticks(())

plt.show()