#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 19:26:02 2022

@author: premkumar
"""

import numpy as np
import tensorflow as tf
from keras import backend as K
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
import pandas as pd
from tensorflow.keras import regularizers
import json

features = {'numerical' : ['Administrative',
                            'Informational',
                            'ProductRelated',
                            'total_pages_viewed',
                            'Administrative_Duration',
                            'Informational_Duration',
                            'ProductRelated_Duration',
                            'Administrative_%',
                            'Informational_%',
                            'ProductRelated_%',
                            'total_duration',
                            'Administrative_Duration_%',
                            'Informational_Duration_%',
                            'ProductRelated_Duration_%',
                            'Administrative_Duration_avg',
                            'Informational_Duration_avg',
                            'ProductRelated_Duration_avg',
                            'page_values_x_bounce_rate',
                            'BounceRates',
                            'ExitRates',
                            'PageValues',
                            'SpecialDay'],
                
            'categorical' : ['Month',
                              'yearQuarter',
                              'OperatingSystems',
                              'Browser',
                              'Region',
                              'TrafficType',
                              'VisitorType']}


def read_data(dataset_type):
    assert dataset_type in ['Unbalanced', 'Smote', 'Rbo']
    
    if dataset_type == 'Unbalanced':
        train_data = pd.read_csv('Train_data.csv')
    elif dataset_type == 'Smote':
        train_data = pd.read_csv('balanced_train_data_smote.csv')
    elif dataset_type == 'Rbo':
        train_data = pd.read_csv('balanced_train_data_rbo.csv')
        
    test_data = pd.read_csv('Test_data.csv')
    
    return train_data, test_data


    
class NaiveBayes:
    
    def __init__(self, x_train, y_train, categorical_variables, numerical_columns):
        
        self.x_train = x_train
        self.y_train = y_train
        self.categorical_variables = categorical_variables
        self.numerical_columns = numerical_columns
        
        self.categorical_columns = []
        for col in list(x_train.columns):
            prefix = str(col).split('_')[0]
            if prefix in categorical_variables:
                self.categorical_columns.append(col)
        
        #print(self.categorical_columns)
    
    def gaussian_dist(self, x, mean, std):
        
        pdf = (1/(std*np.sqrt(2*np.pi))) * (np.exp(-0.5*((x - mean)/std)**2))
        
        return pdf
    
    def fit(self):
        
        self.means = {}
        self.likelihoods = {}
        self.stds = {}
        self.priors = {}
        for cls in self.y_train.unique():
            self.priors[cls] = len(self.y_train[self.y_train == cls])/len(self.y_train)
            for feature in self.numerical_columns:
                self.means[(feature, cls)] = self.x_train[self.y_train == cls][feature].mean()
                self.stds[(feature, cls)] = self.x_train[self.y_train == cls][feature].std()
            for feature in self.categorical_columns:
                self.likelihoods[(feature, cls)] = (self.x_train[self.y_train == cls][feature].sum() + 1) / self.x_train[feature].sum()

    
    def get_prediction(self, x):
        
        pred = {}
        for cls in self.y_train.unique():
            prior = self.priors[cls]
            pred[cls] = np.log(prior)
            for feature in self.x_train.columns:
                if feature in self.numerical_columns:
                    mean = self.means[(feature, cls)]
                    std = self.stds[(feature, cls)]
                    pred[cls] = pred[cls] + np.log(self.gaussian_dist(x[feature], mean, std))
                elif feature in self.categorical_columns:
                    pred[cls] = pred[cls] + np.log(self.likelihoods[(feature, cls)])
        
        return max(pred, key = pred.get)
    
    def predict(self, samples):
        
        y_pred = [self.get_prediction(row) for _, row in samples.iterrows()]
        return y_pred


class LogisticRegression_C:
    
    def __init__(self, x_train, y_train, learning_rate=1e-6, n_iter=5000,
                 batch_size=500):
        
        self.x_train = x_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_iter = n_iter
    
    def sigmoid(self, x):
        
        x = x.astype(float)
        return 1/(1 + np.exp(-x))
    
    def cost_derivative(self, batch_idx):
        
        lamda = 2 #for l2
        
        x = self.x_train.loc[batch_idx].values
        y = self.y_train.loc[batch_idx]
        derivative = (self.sigmoid(x.dot(self.w)) - y).dot(x)/len(x) + (lamda/len(x))*self.w #l2 regularization
        
        return derivative
    
    def gradient_descent(self):
        
        self.w = np.random.normal(loc = 0, scale = 0.1, size = len(self.x_train.columns))
        index = random.sample(list(self.x_train.index), self.batch_size)
        for i in range(self.n_iter):
            cost_derivative = self.cost_derivative(index)
            self.w -= self.learning_rate * cost_derivative.astype(float)
    
    def train(self):
        
        self.x_train['bias'] = 1 #add bias
        self.gradient_descent()
    
    def predict(self, x):
        
        x['bias'] = 1
        pred = self.sigmoid(x.dot(self.w))
        pred = np.where(pred > 0.5, 1, 0)
        
        return pred


def evaluate(model, x_train, y_train, x_test, y_test):
    
    #Evaluate Test set
    y_pred = model.predict(x_test)
    cm_test = confusion_matrix(y_test, y_pred)
    precision_test = precision_score(y_test, y_pred)
    recall_test = recall_score(y_test, y_pred)
    
    #Evaluate Train set
    y_pred = model.predict(x_train)
    cm_train = confusion_matrix(y_train, y_pred)
    precision_train = precision_score(y_train, y_pred)
    recall_train = recall_score(y_train, y_pred)
    
    result = {'train' : {'cm' : cm_train.tolist(),
                         'precision' : precision_train.tolist(),
                         'recall' : recall_train.tolist()},
              'test' : {'cm' : cm_test.tolist(),
                        'precision' : precision_test.tolist(),
                        'recall' : recall_test.tolist()}}
    
    return result


def logistic_regression(x_train, y_train, x_test, y_test):
    
    model = LogisticRegression_C(x_train, y_train)
    model.train()
    
    result = evaluate(model, x_train, y_train, x_test, y_test)
    
    return result
    # model = LogisticRegression()
    # model.fit(x_train, y_train)
    # y_pred = model.predict(x_test)
    # cm = confusion_matrix(y_test, y_pred)
    # print(cm)

def NB(x_train, y_train, x_test, y_test):
    
    model = NaiveBayes(x_train, y_train,
                       features['categorical'],
                       features['numerical'])
    model.fit()
    
    result = evaluate(model, x_train, y_train, x_test, y_test)
    
    return result

def random_forest(x_train, y_train, x_test, y_test):
    
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    
    result = evaluate(model, x_train, y_train, x_test, y_test)
    
    return result

def svm(x_train, y_train, x_test, y_test):
    
    model = SVC(gamma='auto')
    model.fit(x_train, y_train)
    
    result = evaluate(model, x_train, y_train, x_test, y_test)
    
    return result

def XgBoost(x_train, y_train, x_test, y_test):
    
    model = XGBClassifier()
    model.fit(x_train, y_train)
    
    result = evaluate(model, x_train, y_train, x_test, y_test)
    
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
    
    def focal_loss(y_true, y_pred):
        gamma = 2.0
        alpha = 0.75 #Recheck if right class is weighted
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        loss = -K.sum(alpha * K.pow(1.-pt_1, gamma) * K.log(pt_1)) - K.sum((1-alpha) * K.pow(pt_0, gamma) * K.log(1.-pt_0))
        
        return loss
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy', 
                  metrics=[tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Recall(name='recall'),'accuracy'])
    
    history = model.fit(x_train,
                        y_train,
                        batch_size=1000,
                        epochs= 100,
                        validation_data=(x_test,y_test))
    


def main():
    
    train_data, test_data = read_data(dataset_type='Rbo')
    x_train = train_data.drop(columns=['Revenue'])
    y_train = train_data['Revenue']
    x_test = test_data.drop(columns=['Revenue'])
    y_test = test_data['Revenue']
    
    #print(y_train.value_counts())
    
    rf_result = random_forest(x_train, y_train, x_test, y_test)
    
    svm_result = svm(x_train, y_train, x_test, y_test)
    
    xgboost_result = XgBoost(x_train, y_train, x_test, y_test)
    
    nb_result = NB(x_train, y_train, x_test, y_test)
    
    lr_result = logistic_regression(x_train, y_train, x_test, y_test)
    
    result = {'Logistic Regression' : lr_result,
              'Random Forest' : rf_result,
              'Svm' : svm_result,
              'XgBoost' : xgboost_result,
              'Naive Bayes' : nb_result}
    
    print(result)
    
    with open('Base_Model_Results.json', 'w') as file:
        json.dump(result, file)
    
    file.close()

if __name__ == '__main__':    
    main()