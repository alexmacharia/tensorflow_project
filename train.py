import os
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Turn off tensorflow warning message in program output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load training data
training_data_df = pd.read_csv("sales_data_training.csv", dtype=float)

# Separate X and Y datasets
X_training = training_data_df.drop('total_earnings', axis=1)
y_training = training_data_df[['total_earnings']].values

# Load testing data from csv file
test_data_df = pd.read_csv("sales_data_test.csv", dtype=float)