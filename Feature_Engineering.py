#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 12:44:22 2022

@author: premkumar
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def clean(dataset):
    '''
    Perform data cleaning
    Parameters
    ----------
    dataset : dataset
    Returns
    -------
    dataset : cleaned dataset
    '''
    
    dataset['Month'] = dataset['Month'].replace('June', 'Jun') #to maintain same format
    dataset['Revenue'] = dataset['Revenue'].astype('int64')
    dataset['Weekend'] = dataset['Weekend'].astype('int64')
    
    '''Assuming that exit rates cannot be zero(Very Unlikely)
    imputing 0 value of exit rates with median'''
    dataset['ExitRates'] = dataset['ExitRates'].replace(0, np.NaN)
    dataset['ExitRates'] = dataset['ExitRates'].fillna(dataset['ExitRates'].median())
    
    return dataset

def clean_train(train_set):
    '''
    Remove outliers from the training set using z_score test
    Parameters
    ----------
    train_set : training data
    Returns
    -------
    train_set : training set with outliers removed
    '''
    
    def z_score_outlier(val, mean, std):
        z = (val - mean)/std
        if z > 5 or z < -5:
            return True
        else:
            return False
    
    AD_mean = train_set['Administrative_Duration'].mean()
    AD_std = train_set['Administrative_Duration'].std()
    train_set['AD_outlier'] = train_set['Administrative_Duration'].apply(lambda x: z_score_outlier(x,
                                                                                               AD_mean,
                                                                                               AD_std))
    
    ID_mean = train_set['Informational_Duration'].mean()
    ID_std = train_set['Informational_Duration'].std()
    train_set['ID_outlier'] = train_set['Informational_Duration'].apply(lambda x: z_score_outlier(x,
                                                                                                  ID_mean,
                                                                                                  ID_std))
    
    PD_mean = train_set['ProductRelated_Duration'].mean()
    PD_std = train_set['ProductRelated_Duration'].std()
    train_set['PD_outlier'] = train_set['ProductRelated_Duration'].apply(lambda x: z_score_outlier(x,
                                                                                                   PD_mean,
                                                                                                   PD_std))
    
    train_set = train_set[~((train_set['PD_outlier'] == True) |
                            (train_set['ID_outlier'] == True) |
                            (train_set['AD_outlier'] == True))]
    
    train_set = train_set.drop(columns=['AD_outlier', 'ID_outlier', 'PD_outlier'])
    
    return train_set
    

def add_features(dataset):
    '''
    Engineer Features
    Parameters
    ----------
    dataset : dataset
    Returns
    -------
    dataset : dataset added with new features 
    '''
    
    '''Percentage of type of pages visited'''
    dataset['total_pages_viewed'] = dataset['Administrative'] + dataset['Informational'] + dataset['ProductRelated']
    dataset['Administrative_%'] = dataset['Administrative'] / dataset['total_pages_viewed']
    dataset['Informational_%'] = dataset['Informational'] / dataset['total_pages_viewed']
    dataset['ProductRelated_%'] = dataset['ProductRelated'] / dataset['total_pages_viewed']
    
    '''Percentage of duration on a particular page type'''
    dataset['total_duration'] = dataset['Administrative_Duration'] + dataset['Informational_Duration'] + dataset['ProductRelated_Duration']
    dataset['Administrative_Duration_%'] = dataset['Administrative_Duration'] / dataset['total_duration']
    dataset['Informational_Duration_%'] = dataset['Informational_Duration'] / dataset['total_duration']
    dataset['ProductRelated_Duration_%'] = dataset['ProductRelated_Duration'] / dataset['total_duration']
    
    '''Average duration on types of pages visited'''
    dataset['Administrative_Duration_avg'] = dataset['Administrative_Duration'] / dataset['Administrative']
    dataset['Informational_Duration_avg'] = dataset['Informational_Duration'] / dataset['Informational']
    dataset['ProductRelated_Duration_avg'] = dataset['ProductRelated_Duration'] / dataset['ProductRelated']
    
    dataset['page_values_x_bounce_rate'] = dataset['PageValues'] * (1 - dataset['BounceRates'])
    
    
    '''Replace NaN valued generated due to division by zero'''
    nan_replace = {'Administrative_%' : 0,
                   'Informational_%' : 0,
                   'ProductRelated_%' : 0,
                   'Administrative_Duration_%' : 0,
                   'Informational_Duration_%' : 0,
                   'ProductRelated_Duration_%' : 0,
                   'Administrative_Duration_avg' : 0,
                   'Informational_Duration_avg' : 0,
                   'ProductRelated_Duration_avg' : 0,
                   'page_values_x_bounce_rate' : 0
                   }
    dataset = dataset.fillna(nan_replace)
    
    def quarter_of_year(month):
        
        if month in ['Jan', 'Feb', 'Mar', 'Apr']:
            return '1st'
        elif month in ['May', 'Jun', 'Jul', 'Aug']:
            return '2nd'
        elif month in ['Sep', 'Oct', 'Nov', 'Dec']:
            return '3rd'
    
    dataset['yearQuarter'] = dataset['Month'].apply(lambda x : quarter_of_year(x))
    
    return dataset

def split_dataset(dataset):
    '''
    Perform stratified train test split to preserve the same percentage of 
    each target class as in the complete set
    Parameters
    ----------
    dataset : dataset
    Returns
    -------
    train_set : training set
    test_set : test set
    '''

    train_set, test_set = train_test_split(dataset, test_size=0.20,
                                           random_state=42, stratify=dataset['Revenue'])
    
    return train_set, test_set
    
def transform(dataset):
    '''
    Perform standardization on numerical columns and one hot encoding categorical
    columns.
    Parameters
    ----------
    dataset : dataset
    Returns
    -------
    output : transformed dataset
    '''

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
    
    def standardization(x, mean, std):
        z = (x - mean)/std
        return z
        
    #Applying Standard Scaler for numerical columns
    for col in features['numerical']:
        mean = dataset[col].mean()
        std = dataset[col].std()
        dataset[col] = dataset[col].apply(lambda x : standardization(x, mean, std))
    
    #One hot encoding categorical columns
    dataset = pd.get_dummies(dataset, columns=features['categorical'])
    
    return dataset
    
    
def main():
    '''Main function'''
    
    dataset = pd.read_csv('coding_round_data.csv')
    
    dataset = clean(dataset)
    dataset = add_features(dataset)
    dataset = transform(dataset)
    
    train_set, test_set = split_dataset(dataset)
    train_set.reset_index(inplace=False)
    test_set.reset_index(inplace=False)

    train_set = clean_train(train_set)
    
    train_set.to_csv('Train_data.csv', index=False)
    test_set.to_csv('Test_data.csv', index=False)
    

if __name__ == '__main__':
    main()
    