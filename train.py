import os
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Turn off tensorflow warning message in program output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load training data
training_data_df = pd.read_csv("sales_data_training.csv", dtype=float)

# Separate X and Y datasets
X_training = training_data_df.drop('total_earnings', axis=1).values
y_training = training_data_df[['total_earnings']].values

# Load testing data from csv file
test_data_df = pd.read_csv("sales_data_test.csv", dtype=float)

# Separate X and y columns
X_testing = test_data_df.drop('total_earnings', axis=1).values
y_testing = test_data_df[['total_earnings']].values

# Scale the data to 0 to 1 range
X_scaler = MinMaxScaler(feature_range=(0,1))
y_scaler = MinMaxScaler(feature_range=(0,1))

# Scale both training features and labels
X_scaled_training = X_scaler.fit_transform(X_training)
y_scaled_training = y_scaler.fit_transform(y_training)

# Use same scaler to scale test data
X_scaled_testing = X_scaler.transform(X_testing)
y_scaled_testing = y_scaler.transform(y_testing)

# Define model parameters
learning_rate = 0.001
training_epochs = 100

# Define inputs and outputs
number_of_inputs = 9
number_of_outputs = 1

# Define number of neurons in each layer
layer_1_nodes = 50
layer_2_nodes = 100
layer_3_nodes = 50

