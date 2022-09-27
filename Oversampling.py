#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 16:06:47 2022

@author: premkumar
"""

import pandas as pd
import numpy as np
import random
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import NearMiss
from tqdm import tqdm

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


class RBO:
    
    def __init__(self,
                 dataset,
                 target_variable,
                 categorical_columns,
                 step_size=0.0001,
                 n_iters=10,
                 gamma=0.05,
                 criteria='maximize'):
        assert criteria in ['balance', 'maximize', 'minimize']
        
        self.dataset = dataset
        self.target_variable = target_variable
        self.categorical_columns = categorical_columns
        self.step_size = step_size
        self.n_iters = n_iters
        self.eps = 1/gamma
        self.criteria = criteria
    
    def get_class_data(self):
        
        classes = self.dataset[self.target_variable].unique()
        class_weights = self.dataset[self.target_variable].value_counts() / len(self.dataset)
        self.min_class = np.argmin(class_weights)
        self.max_class = np.argmax(class_weights)
        self.k = self.dataset[self.dataset[self.target_variable] == self.min_class].drop(columns=['Revenue'])
        self.K = self.dataset[self.dataset[self.target_variable] == self.max_class].drop(columns=['Revenue'])
        
    def rbf(self, x, y):

        distance = np.sum(np.abs(x - y))
        rbf = np.exp(-(distance * self.eps) ** 2)
        
        return rbf
    
    def score(self, point):
        
        mutual_density_score = 0
        #print(point)
        for i in self.K.index:
            rbf_ = self.rbf(point.values, self.K.loc[i].values)
            mutual_density_score += rbf_
        
        for i in self.k.index:
            rbf_ = self.rbf(point.values, self.k.loc[i].values)
            mutual_density_score -= rbf_
        
        return mutual_density_score
    
    def generate_samples(self):
        
        min_scores = {}
        for i in tqdm(self.k.index):
            min_scores[i] = self.score(self.k.loc[i])
        
        samples = pd.DataFrame(columns = self.k.columns)
        while len(samples) + len(self.k) < len(self.K):
            print(len(samples) + len(self.k))
            idx = np.random.choice(self.k.index)
            point = self.k.loc[idx].copy()
            score = min_scores[idx]
            
            preserve = False
            for i in range(self.n_iters):
                translated = point.copy()
                direction = np.random.choice(list(point.index))
                if direction in self.categorical_columns:
                    translated_direction = random.choice([0,1])
                    translated_score = self.score(translated)
                else:
                    sign = np.random.choice([-1,1])
                    translated[direction] = translated[direction] + (sign * self.step_size)
                    translated_score = self.score(translated)
                
                if self.criteria == 'balance' and np.abs(translated_score) < np.abs(score):
                    preserve = True
                    point = translated
                    score = translated_score
                
                elif self.criteria == 'maximize' and translated_score > score:
                    preserve = True
                    point = translated
                    score = translated_score
                
                elif self.criteria == 'minimize' and translated_score < score:
                    preserve = True
                    point = translated
                    score = translated_score
            
            if preserve == True:
                samples = samples.append(point, ignore_index = True)
        
        samples[self.target_variable] = self.min_class
        df_upsampled = self.dataset.append(samples, ignore_index = True)
        
        print(df_upsampled['Revenue'].value_counts())
        
        return df_upsampled 
        
    def resample(self):
        
        self.get_class_data()
        df_upsampled = self.generate_samples()
        
        return df_upsampled

def rbo(dataset):
    
    cat_columns = []
    for col in dataset.columns:
        if str(col).split('_')[0] in features['categorical']:
            cat_columns.append(col)
    
    rbo = RBO(dataset, 'Revenue', cat_columns)
    upsampled_df = rbo.resample()
    
    return upsampled_df

def smote(X, y):
    
    cat_index = []
    for idx, col in enumerate(X.columns):
        if str(col).split('_')[0] in features['categorical']:
            cat_index.append(idx)
    
    sm = SMOTENC(random_state=42, categorical_features=cat_index)
    X_res, y_res = sm.fit_resample(X, y)
    
    dataset = pd.DataFrame(X_res)
    dataset['Revenue'] = y_res
    print(dataset)
    return dataset

def undersample(X, y, sampling_strategy='auto'):
    
    nr = NearMiss(version=1, n_neighbors=3,
                  sampling_strategy=sampling_strategy)
    X_res, y_res = nr.fit_resample(X, y)
    
    dataset = pd.DataFrame(X_res)
    dataset['Revenue'] = y_res
    
    return dataset

def main():
    
    dataset = pd.read_csv('Train_data.csv')
    X = dataset.drop(columns=['Revenue'])
    y = dataset['Revenue']
    
    reduced = undersample(X, y, 0.35)
    #print(reduced.columns)
    rbo_dataset = rbo(dataset)
    print(rbo_dataset['Revenue'].value_counts())
    rbo_dataset.to_csv('balanced_train_data_rbo.csv', index=False)
    
    # #smote_dataset = smote(X, y)
    # #smote_dataset.to_csv('balanced_train_data_smote.csv', index=False)
    
    # undersampled_dataset = undersample(X, y)
    # undersampled_dataset.to_csv('Undersampled_data.csv', index=False)

if __name__ == '__main__':
    main()