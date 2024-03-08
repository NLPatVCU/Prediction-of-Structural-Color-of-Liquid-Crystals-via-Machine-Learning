"""
File: classification.py

Author: Bridget McInnes

Date: October 2023

Description: This script performs a ten-fold cross validaiton using a decision tree over
             the colours data and outputs the best pruned decision tree

Usage: python classification.py <colours csv file> <fig output file>

Output: Classification Accuracy for each fold and the best pruned decision tree. 

Example:
  python3 classification.py colours_mlr_expanded.csv figure.png 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sys import argv

# load dataset
dataset = np.loadtxt(argv[1], delimiter=',')
#dataset = np.loadtxt('colours_mlr_expanded_bagged.csv', delimiter=',')

# Randomly shuffle the elements in the dataset
np.random.shuffle(dataset)

# Assuming that the first column is the target variable
y = dataset[:, 3]     # First column is the target
X = dataset[:, 0:3]  # All columns except the first one are features

# Initialize the KFold cross-validator
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Initialize a variable to keep track of fold number
fold = 1

l_xtrain = []
l_ytrain = []
l_xtest = []
l_ytest = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    #print(X_train)
    # Split data into training and testing sets
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Print the shapes of the splits to verify
    #print("X_train shape:", X_train.shape)
    #print("y_train shape:", y_train.shape)
    #print("X_test shape:", X_test.shape)
    #print("y_test shape:", y_test.shape)


    ## Create a Classification model
    model = DecisionTreeClassifier(max_depth = 4)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = model.predict(X_test)

    # Print fold information
    #print(f'Fold {fold}:')
    #print(f'Train indices: {train_index}')
    #print(f'Test indices: {test_index}')
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    #j = len(X_test)
    #for i in range(j):
        #print(X_test[i], " ", y_pred[i])

    l_xtrain = X_train
    l_ytrain = y_train
    l_xtest  = X_test
    l_ytest = y_test

    # Increment the fold number
    fold += 1

#print(l_xtrain.shape)
#print(l_ytrain.shape)
#print(l_xtest.shape)
#print(l_ytest.shape)

# Pruning using cost-complexity pruning
path = model.cost_complexity_pruning_path(l_xtrain, l_ytrain)
ccp_alphas = path.ccp_alphas

trees = []

for ccp_alpha in ccp_alphas:
    pruned_tree = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    pruned_tree.fit(l_xtrain, l_ytrain)
    trees.append(pruned_tree)


# Find the classifier with the best alpha
acc_scores = [accuracy_score(l_ytest, tree.predict(l_xtest)) for tree in trees]
best_tree = trees[acc_scores.index(max(acc_scores))]
best_alpha = ccp_alphas[acc_scores.index(max(acc_scores))]
    
# Plot the decision tree
feature_names = ["COC", "CC", "CP"]
plt.figure(figsize=(20, 15))
plot_tree(best_tree, feature_names=feature_names, filled=True)
plt.savefig(argv[2], dpi=300)




