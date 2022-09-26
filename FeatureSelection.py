#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 22:10:16 2022

@author: premkumar
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from operator import itemgetter
import matplotlib.pyplot as plt
import json



class mRMR:
    
    def __init__(self, x_train, y_train, n_features, k_max):
        '''
        Constructor of the class
        This class performs mRMR feature selection algorithm
        Parameters
        ----------
        x_train : Training dataset
        y_train : Target Variable
        n_features : total number of features to be selected
        k_max : Maximum number of top scoring features to be considered

        Returns
        -------
        None.
        '''
        self.x_train = x_train
        self.y_train = y_train
        self.n_features = n_features
        self.k_max = k_max
    
    def entropy(self, x):
        '''
        Calculate entropy of vector x
        '''
        _, count = np.unique(x, return_counts=True, axis=0)
        prob = count/len(x)
        entropy = np.sum((-1) * prob * np.log2(prob))
        
        return entropy
    
    def conditional_entropy(self, y, x):
        '''
        Calculate H(y|x)
        '''
        yx = np.c_[y, x]
        joint_entropy = self.entropy(yx)
        conditional_entropy = joint_entropy - self.entropy(x)
        
        return conditional_entropy
    
    def mutual_information(self, x, y):
        '''
        Calculated Information gain / mutual information.
        '''
        MI = self.entropy(x) - self.conditional_entropy(x, y)
        
        return MI
    
    def all_mutual_information(self):
        '''
        Calculates mutual information between each feature in x and target variable y.
        '''
        
        mutual_inf = []
        y = np.array(self.y_train.values)
        for col in self.x_train.columns:
            x = np.array(self.x_train[col].values)
            mutual_inf.append(self.mutual_information(x, y))
        
        return sorted(enumerate(mutual_inf), key=itemgetter(1), reverse=True)
    
    def fit(self, threshold=0.8):
        
        ndim = self.x_train.shape[1]
        
        MI_map = self.all_mutual_information()
        features_ordered = [i[0] for i in MI_map]
        #print(features_ordered[:self.k_max])
        subset = self.x_train.iloc[:, features_ordered[:self.k_max]]
        #print(subset)
        
        MI_target = {}
        max_idx, max_rel = MI_map[0]
        mrmr_map = [(max_idx, max_rel)]
        idx_mask = [max_idx]
        
        MI_target[max_idx] = []
                
        for col in self.x_train.columns:
            x = np.array(self.x_train[col].values)
            MI_target[max_idx].append(self.mutual_information(x, subset.iloc[:, max_idx].values))
        
        for i in range(self.n_features - 1):
            phi_vec = []
            for idx, rel in MI_map[1:self.k_max]:
                if idx not in idx_mask:
                    red = sum(MI_target[j][idx] for j,_ in mrmr_map)/len(mrmr_map)
                    phi = red - rel
                    phi_vec.append((idx, phi))
            
            idx, mrmr_val = max(phi_vec, key=itemgetter(1))
            
            MI_target[idx] = [] #Next feature
            for col in self.x_train.columns:
                x = np.array(self.x_train[col].values)
                MI_target[idx].append(self.mutual_information(x, subset.iloc[:, idx].values))
            
            mrmr_map.append((idx, mrmr_val))
            idx_mask.append(idx)
        
        mrmr_map_sorted = sorted(mrmr_map, key=itemgetter(1), reverse=True)
        
        return [x[0] for x in mrmr_map_sorted], [x[1] for x in mrmr_map_sorted]      
         
        

def mrmr(x_train, y_train):
    
    mrmr = mRMR(x_train, y_train, 30, 50)
    features, mrmr_score = mrmr.fit()
    feature_names = [x_train.columns[idx] for idx in features]
    
    with open('mRMR_features.json', 'w') as file:
        json.dump(feature_names, file)
    file.close()
    
    feature_importances = pd.Series(mrmr_score, index=feature_names)
    fig, ax = plt.subplots(figsize=(60,30))
    feature_importances.plot.bar()
    ax.set_title("Feature importances using mRMR algorithm")
    ax.set_ylabel("mRMR Scores")
    fig.tight_layout()
    plt.savefig('mRMR_feature_importance.png', dpi=300)
    
    
def random_forest_importance(x_train, y_train,
                             top_n=30, n_estimators=50):
    '''
    This function implements feature scoring using random forest classifer, 
    utilizing impurity metric and permutation method.
    Parameters
    ----------
    x_train : Training dataset
    y_train : Target variable
    top_n : Total number of features to be selected. The default is 30.
    n_estimators : n_estimators for random forest classifier. The default is 50.

    Returns
    -------
    None.

    '''
    
    forest = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1)
    forest.fit(x_train, y_train)
    
    importances = forest.feature_importances_
    sort_ = np.argsort(importances.copy())[::-1] #Sorting and reversing the order
    top_index = sort_[:30]
    importances = np.sort(importances)[::-1]
    top_MDI = importances[:30]
    
    feature_names = [x_train.columns[idx] for idx in top_index]
    forest_importances = pd.Series(top_MDI, index=feature_names)
    
    with open('MDI_features.json', 'w') as file:
        json.dump(feature_names, file)
    file.close()
    
    fig, ax = plt.subplots(figsize=(60,30))
    forest_importances.plot.bar()
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.savefig('random_forest_feature_importance_MDI.png', dpi=300)
    
    result = permutation_importance(forest, x_train, y_train,
                                    n_repeats=10, random_state=42,
                                    n_jobs=2) #check if test set should be passed
    
    perm_importances = result.importances_mean
    sort_ = np.argsort(perm_importances.copy())[::-1]
    top_index = sort_[:30]
    perm_importances = np.sort(perm_importances)[::-1]
    top_features = perm_importances[:30]
    
    feature_names = [x_train.columns[idx] for idx in top_index]
    forest_importances = pd.Series(top_features, index=feature_names)
    
    with open('Permutation_features.json', 'w') as file:
        json.dump(feature_names, file)
    file.close()
    
    fig, ax = plt.subplots(figsize=(60,30))
    forest_importances.plot.bar()
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.savefig('random_forest_feature_importance_Permutation.png', dpi=300)
    
def logistic_regression_importance(x_train, y_train):
    
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    importances = model.coef_[0]
    importances = [abs(x) for x in importances] #getting absolute value
    sort_ = np.argsort(importances.copy())[::-1]
    top_index = sort_[:30]
    importances.sort(reverse=True)
    top_coef = importances[:30]
    feature_names = [x_train.columns[idx] for idx in top_index]
    
    with open('Logisitic_Regression_features.json', 'w') as file:
        json.dump(feature_names, file)
    file.close()
    
    feature_importance = pd.Series(top_coef, index=feature_names)
    fig, ax = plt.subplots(figsize=(60,30))
    feature_importance.plot.bar()
    ax.set_title("Feature importances by assesing Logistic Regression Coef/weights")
    ax.set_ylabel("Coef")
    fig.tight_layout()
    plt.savefig('LogisticRegression_feature_importance.png', dpi=300)



def main():
    
    dataset = pd.read_csv('Train_data.csv')
    x_train = dataset.drop(columns=['Revenue'])
    y_train = dataset['Revenue']
    
    random_forest_importance(x_train, y_train)
    logistic_regression_importance(x_train, y_train)
    mrmr(x_train, y_train)

if __name__ == '__main__':
    main()
