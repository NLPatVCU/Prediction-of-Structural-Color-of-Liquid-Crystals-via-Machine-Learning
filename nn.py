import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
from math import sqrt

import tensorflow as tf

from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score, recall_score, f1_score

def negative_r_squared(y_true, y_pred):
    y_mean = tf.reduce_mean(y_true)
    ss_total = tf.reduce_sum(tf.square(y_true - y_mean))
    ss_residual = tf.reduce_sum(tf.square(y_true - y_pred))
    r_squared = 1.0 - (ss_residual / ss_total)
    return -r_squared
# load dataset
dataset = np.loadtxt('colours_mlr_expanded.csv', delimiter=',')
#dataset = np.loadtxt('colours_mlr_rgb.csv', delimiter=',')

# Randomly shuffle the elements in the dataset
np.random.shuffle(dataset)

# Assuming that the first column is the target variable
y = dataset[:, 3]     # First column is the target
X = dataset[:, 0:3]  # All columns except the first one are features

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Create a neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer for regression
])

# Compile the model
#model.compile(optimizer='adam', loss='mean_squared_error')
model.compile(optimizer='adam', loss=negative_r_squared)

# Train the model on the training data
model.fit(X_train, y_train, epochs=1000, batch_size=32, verbose=2)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE) for evaluation
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

rmse = sqrt(mse)
print("Root Mean Squared Error:", rmse)

r2 = r2_score(y_test, y_pred)
print("R-squared:", r2) #0.7 0.9 0.96 0.9

n_test = [[20,25,55],[30,20,50]]
n_pred = model.predict(n_test)
print(n_pred)
