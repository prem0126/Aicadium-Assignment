#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 16:15:26 2022

@author: premkumar
"""

import numpy as np
import tensorflow as tf
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import pandas as pd
import json

from sklearn.model_selection import RandomizedSearchCV



def read_dataset(dataset_type):
    assert dataset_type in ['Unbalanced', 'Smote', 'Rbo']
    
    if dataset_type == 'Unbalanced':
        train_data = pd.read_csv('Train_data.csv')
    elif dataset_type == 'Smote':
        train_data = pd.read_csv('balanced_train_data_smote.csv')
    elif dataset_type == 'Rbo':
        train_data = pd.read_csv('balanced_train_data_rbo.csv')
        
    test_data = pd.read_csv('Test_data.csv')
    
    return train_data, test_data

def RandomForest(x_train, y_train, x_test, y_test):
    
    
    with open('MDI_features.json', 'r') as file:
        features_selected = json.load(file)
    file.close()
    
    #Subsetting features selected using MDI method
    x_train = x_train[x_train.columns.intersection(features_selected)]
    x_test = x_test[x_test.columns.intersection(features_selected)]
    
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)] #Number of trees
    max_features = ['sqrt', 'log2'] #Num of features to consider at every split
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)] #Max num of levels in a tree
    max_depth.append(None)
    min_samples_split = [2, 5, 10, 20, 30] #Min num of samples required to split a node
    min_samples_leaf = [1, 2, 4] #Min num of samples required at each leaf
    bootstrap = [True, False] #Selecting samples
    
    random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
    
    model = RandomForestClassifier()
    model_random = RandomizedSearchCV(estimator = model,
                                   param_distributions = random_grid,
                                   n_iter = 100, cv = 3, verbose=2,
                                   random_state=42, n_jobs = -1)
    
    model_random.fit(x_train, y_train)
    
    best_params = model_random.best_params_
    with open('RandomForest_bestparams.json', 'w') as file:
        json.dump(best_params, file)
    file.close()
    
    best_model = model_random.best_estimator_
    best_model.fit(x_train, y_train)
    
    #Evaluate Train set
    y_pred = best_model.predict(x_train)
    cm_train = confusion_matrix(y_train, y_pred)
    print(cm_train)
    
    #Evaluate Test set
    y_pred = best_model.predict(x_test)
    cm_test = confusion_matrix(y_test, y_pred)
    print(cm_test)
    

def main():
    
    train_smote, test_smote = read_dataset('Smote')
    x_train_smote = train_smote.drop(columns=['Revenue'])
    y_train_smote = train_smote['Revenue']
    x_test_smote = test_smote.drop(columns=['Revenue'])
    y_test_smote = test_smote['Revenue']
    
    RandomForest(x_train_smote, y_train_smote,
                 x_test_smote, y_test_smote)
    

if __name__ == '__main__':
    main()