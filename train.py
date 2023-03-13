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

# Define the layers of the neural network

# Input layer
with tf.variable_scope('input'):
    X = tf.placeholder(tf.float32, shape=(None, number_of_inputs))

# Layer1
with tf.variable_scope('layer_1'):
    weights = tf.get_variable("weights", shape=[number_of_inputs, layer_1_nodes],
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases1", shape=[layer_1_nodes],
                             initializer=tf.zeros_initializer())
    layer_1_output = tf.nn.relu(tf.matmul(X, weights) + biases)


# Layer2
with tf.variable_scope('layer_2'):
    weights = tf.get_variable("weights2", shape=[layer_1_nodes, layer_2_nodes],
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases2", shape=[layer_2_nodes],
                             initializer=tf.zeros_initializer())
    layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, weights) + biases)


# Layer3
with tf.variable_scope('layer_3'):
    weights = tf.get_variable("weights3", shape=[layer_2_nodes, layer_3_nodes],
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases3", shape=[layer_3_nodes],
                             initializer=tf.zeros_initializer())
    layer_3_output = tf.nn.relu(tf.matmul(layer_2_output, weights) + biases)


# Output layer
with tf.variable_scope('output'):
    weights = tf.get_variable("weights4", shape=[layer_3_nodes, number_of_outputs],
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases4", shape=[number_of_outputs],
                             initializer=tf.zeros_initializer())

# Define the cost function of the neural network
with tf.variable_scope('cost'):
    Y = tf.placeholder(tf.float32, shape=(None, 1))
    cost = tf.reduce_mean(tf.squared_difference(prediction, Y))

# Define the optimizer function that will be run
with tf.variable_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Initialize a session to run tensorflow operations
with tf.Session() as session:
    # Initialize all variables and layers of the network
    session.run(tf.global_variables_initializer())
    # Run the optimizer over and over to train the network
    for epoch in range(training_epochs):
        # Feed the training data and do one step of newural network training
        session.run(optimizer, feed_dict={X: X_scaled_training, Y: Y_scaled_training})
        print("Training pass: {}".format(epoch))
    print("Training is complete!")