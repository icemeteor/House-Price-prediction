# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 21:51:42 2018

@author: yinsh
"""

# Code you have previously used to load data
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt



# Path of the file to read.
# iowa_file_path needs to be changed when running on another computer

#iowa_file_path = "./train.csv"
iowa_file_path = 'D:/coms 3111/train.csv'

home_data = pd.read_csv(iowa_file_path)
# Create target object and call it y
y = home_data.SalePrice
# Create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath',
            'BedroomAbvGr', 'TotRmsAbvGrd']
x = home_data[features]


# To improve accuracy, create a new Random Forest model which you will train on all training data
rf_model_on_full_data = RandomForestRegressor(random_state=1)

# fit rf_model_on_full_data on all data from the 
rf_model_on_full_data.fit(x,y)
y_pred = rf_model_on_full_data.predict(x)

# show the plot of prediction values and real values
plt.scatter(y,y_pred,color="red")
plt.show()
plt.hist(y,bins = 20,color="blue",alpha=0.8)
plt.hist(y_pred,bins=20,color="orange",alpha=0.4)
plt.xlabel("price in US dolloar")
plt.show()

# From the plot, I think that the model fits the train set well.
# Try it on the test file and show the final prediction


# path to file you will use for predictions
# Test_data_path needs to be changed when run on another computer

#test_data_path = "./test.csv"
test_data_path = 'D:/coms 3111/test.csv'

# read test data file using pandas
test_data = pd.read_csv(test_data_path)

# create test_X which comes from test_data but includes only the columns you used for prediction.
test_X = test_data[features]

# make predictions which we will submit. 
test_preds = rf_model_on_full_data.predict(test_X)
# print the first 5 price to see the result
print(test_preds[:5])

# The lines below shows you how to save your data in the format needed to score it in the competition
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})

output.to_csv('final_submission.csv', index=False)