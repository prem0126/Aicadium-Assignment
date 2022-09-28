#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 16:15:26 2022

@author: premkumar
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score
import pandas as pd
import json

#from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV


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

def evaluate(model, x_train, y_train, x_test, y_test):
    
    
    #Evaluate Test set
    y_pred = model.predict(x_test)
    cm_test = confusion_matrix(y_test, y_pred)
    accuracy_test = accuracy_score(y_test, y_pred)
    precision_test = precision_score(y_test, y_pred)
    recall_test = recall_score(y_test, y_pred)
    
    #Evaluate Train set
    y_pred = model.predict(x_train)
    cm_train = confusion_matrix(y_train, y_pred)
    accuracy_train = accuracy_score(y_train, y_pred)
    precision_train = precision_score(y_train, y_pred)
    recall_train = recall_score(y_train, y_pred)
    
    result = {'train' : {'cm' : cm_train.tolist(),
                         'accuracy' : accuracy_train.tolist(),
                         'precision' : precision_train.tolist(),
                         'recall' : recall_train.tolist()},
              'test' : {'cm' : cm_test.tolist(),
                        'accuracy' : accuracy_test.tolist(),
                        'precision' : precision_test.tolist(),
                        'recall' : recall_test.tolist()}}
    
    return result

def RandomForest(x_train, y_train, x_test, y_test):
    '''
    Performs grid search to fine tune Random Forest model, saves the selected parameters
    input : x_train, y_train, x_test, y_test
    output : best result produced by the model
    '''
    
    with open('MDI_features.json', 'r') as file:
        features_selected = json.load(file)
    file.close()
    
    #Subsetting features selected using MDI method
    x_train = x_train[x_train.columns.intersection(features_selected)]
    x_test = x_test[x_test.columns.intersection(features_selected)]

    param_grid = {'bootstrap': [True],
                'max_depth': [80, 90, 100, 110],
                'max_features': [2, 3],
                'min_samples_leaf': [3, 4, 5],
                'min_samples_split': [8, 10, 12],
                'n_estimators': [100, 200, 300, 1000]}
    
    model = RandomForestClassifier()
    model_grid = GridSearchCV(estimator=model,
                                param_grid=param_grid, 
                                cv=3,
                                n_jobs=-1,
                                verbose=2)
    
    model_grid.fit(x_train, y_train)
    
    best_params = model_grid.best_params_
    with open('RandomForest_bestparams.json', 'w') as file:
        json.dump(best_params, file)
    file.close()
    
    best_model = model_grid.best_estimator_
    best_model.fit(x_train, y_train)
    
    result = evaluate(best_model, x_train, y_train, x_test, y_test)
    
    return result
    
def svm(x_train, y_train, x_test, y_test):
    '''
    Performs grid search to fine tune svm model, saves the selected parameters
    input : x_train, y_train, x_test, y_test
    output : best result produced by the model
    '''
    
    with open('mRMR_features.json', 'r') as file:
        features_selected = json.load(file)
    file.close()
    
    #Subsetting features selected using MDI method
    x_train = x_train[x_train.columns.intersection(features_selected)]
    x_test = x_test[x_test.columns.intersection(features_selected)]
    
    param_grid = {'C': [0.1, 1, 10],
                  'gamma': [1, 0.1, 0.01, 0.001],
                  'kernel': ['rbf', 'poly', 'sigmoid']}
    
    model = SVC()
    model_grid = GridSearchCV(estimator=SVC(),
                              param_grid=param_grid,
                              refit=True,verbose=2,
                              n_jobs=-1, cv=3)
    model_grid.fit(x_train, y_train)
    
    best_params = model_grid.best_params_
    with open('SVM_bestparams.json', 'w') as file:
        json.dump(best_params, file)
    file.close()
    
    best_model = model_grid.best_estimator_
    best_model.fit(x_train, y_train)
    
    result = evaluate(best_model, x_train, y_train, x_test, y_test)
    
    return result

def xgboost(x_train, y_train, x_test, y_test):
    '''
    Performs grid search to fine tune xgboost model, saves the selected parameters
    input : x_train, y_train, x_test, y_test
    output : best result produced by the model
    '''
    
    with open('mRMR_features.json', 'r') as file:
        features_selected = json.load(file)
    file.close()
    
    #Subsetting features selected using MDI method
    x_train = x_train[x_train.columns.intersection(features_selected)]
    x_test = x_test[x_test.columns.intersection(features_selected)]
    
    param_grid = {'gamma': [0, 0.2, 0.8, 3.2, 6.4, 25.6, 50],
                  'max_depth' : [3, 6, 10, 15],
                  'learning_rate' : [0.01, 0.05, 0.1, 1],
                  # 'reg_alpha': [0.4, 0.8, 1.6, 3.2, 12.8],
                  # 'reg_lambda': [0.4,0.8,1.6,3.2,6.4,12.8],
                  'n_estimators' : [60, 100, 150, 200]}
        
    model = XGBClassifier()
    model_grid = GridSearchCV(estimator=model,
                            param_grid=param_grid,
                            scoring = 'roc_auc',
                            n_jobs = -1,
                            cv = 3,
                            verbose=2)
    model_grid.fit(x_train, y_train)
    
    best_params = model_grid.best_params_
    with open('XGboost_bestparams.json', 'w') as file:
        json.dump(best_params, file)
    file.close()
    
    best_model = model_grid.best_estimator_
    best_model.fit(x_train, y_train)
    
    result = evaluate(best_model, x_train, y_train, x_test, y_test)
    
    return result

def MLP(x_train, y_train, x_test, y_test):
    
    inp = tf.keras.Input((x_train.shape[1],))
    layer2 = tf.keras.layers.Dense(units=150, activation='tanh')(inp)
    layer3 = tf.keras.layers.Dropout(0.1)(layer2)
    layer4 = tf.keras.layers.Dense(units=200, activation='tanh',
                                   kernel_regularizer=regularizers.L2(1e-4))(layer3)
    layer5 = tf.keras.layers.Dropout(0.1)(layer4)
    layer6 = tf.keras.layers.Dense(units=100, activation='tanh',
                                   kernel_regularizer=regularizers.L2(1e-4))(layer5)
    layer7 = tf.keras.layers.Dropout(0.1)(layer6)
    layer8 = tf.keras.layers.Dense(units=50, activation='tanh',
                                   kernel_regularizer=regularizers.L2(1e-4))(layer7)
    layer9 = tf.keras.layers.Dropout(0.1)(layer8)
    layer10 = tf.keras.layers.Dense(units=20, activation='tanh',
                                    kernel_regularizer=regularizers.L2(1e-4))(layer9)
    layer11 = tf.keras.layers.Dropout(0.1)(layer10)
    
    output = tf.keras.layers.Dense(1, activation = 'sigmoid')(layer11)
    
    model = tf.keras.Model(inputs=inp, outputs=output)
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy', 
                  metrics=[tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Recall(name='recall'),'accuracy'])
    
    history = model.fit(x_train,
                        y_train,
                        batch_size=1000,
                        epochs= 50,
                        validation_data=(x_test,y_test))
    

def main():
    '''
    Main function that calls all the model functions and saves the results
    '''
    
    train_smote, test_smote = read_dataset('Smote')
    x_train_smote = train_smote.drop(columns=['Revenue'])
    y_train_smote = train_smote['Revenue']
    x_test_smote = test_smote.drop(columns=['Revenue'])
    y_test_smote = test_smote['Revenue']
    
    MLP(x_train_smote, y_train_smote, x_test_smote, y_test_smote)
    
    rf_result_smote = RandomForest(x_train_smote, y_train_smote,
                                    x_test_smote, y_test_smote)
    
    with open('Finetued_RandomForest_result_somte.json', 'w') as file:
        json.dump(rf_result_smote, file)
    file.close()
    
    svm_result_smote = svm(x_train_smote, y_train_smote,
                            x_test_smote, y_test_smote)
    
    with open('Finetued_svm_result_somte.json', 'w') as file:
        json.dump(svm_result_smote, file)
    file.close()
    
    xgboost_result_smote = xgboost(x_train_smote, y_train_smote,
                            x_test_smote, y_test_smote)
    
    with open('Finetued_XGboost_result_somte.json', 'w') as file:
        json.dump(xgboost_result_smote, file)
    file.close()
    
    smote_result = {'Random Forest' : rf_result_smote,
                    'Svm' : svm_result_smote,
                    'XGboost' : xgboost_result_smote}
    
    with open('Finetuned_models_smote.json', 'w') as file:
        json.dump(smote_result, file)
    file.close()    
    
    train_rbo, test_rbo = read_dataset('Rbo')
    x_train_rbo = train_rbo.drop(columns=['Revenue'])
    y_train_rbo = train_rbo['Revenue']
    x_test_rbo = test_rbo.drop(columns=['Revenue'])
    y_test_rbo = test_rbo['Revenue']
    
    MLP(x_train_rbo, y_train_rbo, x_test_rbo, y_test_rbo)
    
    rf_result_rbo = RandomForest(x_train_rbo, y_train_rbo,
                                    x_test_rbo, y_test_rbo)
    
    with open('Finetued_RandomForest_result_rbo.json', 'w') as file:
        json.dump(rf_result_rbo, file)
    file.close()
    
    xgboost_result_rbo = xgboost(x_train_rbo, y_train_rbo,
                            x_test_rbo, y_test_rbo)
    
    with open('Finetued_XGboost_result_rbo.json', 'w') as file:
        json.dump(xgboost_result_rbo, file)
    file.close()
    
    svm_result_rbo = svm(x_train_rbo, y_train_rbo,
                            x_test_rbo, y_test_rbo)
    
    with open('Finetued_svm_result_rbo.json', 'w') as file:
        json.dump(svm_result_rbo, file)
    file.close()
    
    rbo_result = {'Random Forest' : rf_result_rbo,
                    'Svm' : svm_result_rbo,
                    'XGboost' : xgboost_result_rbo}
    
    with open('Finetuned_models_rbo.json', 'w') as file:
        json.dump(rbo_result, file)
    file.close()  
    

if __name__ == '__main__':
    main()