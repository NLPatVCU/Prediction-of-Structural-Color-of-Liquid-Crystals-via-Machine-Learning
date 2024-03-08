"""
File: regression.py

Author: Bridget McInnes

Date: October 2023

Description: This script performs a regression over the colours data. 

Usage: python regression.py <colours csv file> <fig output file> <Regression model>

Options for Regression modle:
  1. LinearRegression
  2. RandomForestRegression
  3. SVR
  4. DecisionTree Regressor (default)

Example:
  python3 regression.py colours_mlr_expanded.csv figure.png LinearRegression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

from sys import argv

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# load dataset
dataset = np.loadtxt(argv[1], delimiter=',')
#dataset = np.loadtxt('colours_mlr_expanded.csv', delimiter=',')
#dataset = np.loadtxt('colours_mlr_rgb.csv', delimiter=',')

# Randomly shuffle the elements in the dataset
np.random.shuffle(dataset)

# Assuming that the first column is the target variable
y = dataset[:, 3]     # First column is the target
X = dataset[:, 0:3]  # All columns except the first one are features

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Print the shapes of the splits to verify
#print("X_train shape:", X_train.shape)
#print("y_train shape:", y_train.shape)
#print("X_test shape:", X_test.shape)
#print("y_test shape:", y_test.shape)


## Create a Linear Regression model
model = DecisionTreeRegressor()
if len(argv) < 3:
    if(argv[3] == "LinearRegression"):
        model = LinearRegression() 
    if(argv[3] == "SVR"):
        model = SVR(kernel='poly') 
    if(argv[3] == "RandomForestRegression"):
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=0)
    else:
        model = DecisionTreeRegressor()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE) as a performance metric
mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error:", mse)

r2 = r2_score(y_test, y_pred)
print("R-squared:", r2) #0.7 0.9 0.96 0.9

# Pruning using cost-complexity pruning
path = model.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
trees = []

for ccp_alpha in ccp_alphas:
    pruned_tree = DecisionTreeRegressor(random_state=42, ccp_alpha=ccp_alpha)
    pruned_tree.fit(X_train, y_train)
    trees.append(pruned_tree)

# Find the best pruned tree based on cross-validation
validation_mses = [mean_squared_error(y_test, tree.predict(X_test)) for tree in trees]
validation_r2 = [r2_score(y_test, tree.predict(X_test)) for tree in trees]
best_index = np.argmin(validation_mses)
best_pruned_tree = trees[best_index]
best_pruned_mse = validation_mses[best_index]
best_pruned_r2 = validation_r2[best_index]

print(f"Best Pruned Tree MSE: {best_pruned_mse:.2f}")
print(f"Best Pruned Tree R^2: {best_pruned_r2:.2f}")

leaf_nodes = np.unique(best_pruned_tree.apply(X))


#  get the min max
def calculate_min_max(node):
    node_indices = np.where(best_pruned_tree.apply(X) == node)[0]
    node_target_values = y[node_indices]
    return np.min(node_target_values), np.max(node_target_values)
min_max_values = [calculate_min_max(node) for node in leaf_nodes]

# Plot the decision tree
feature_names = ["COC", "CC", "CP"]
plt.figure(figsize=(20, 15))
plot_tree(trees[best_index], feature_names=feature_names, filled=True)

plt.savefig(argv[2], dpi=300)
