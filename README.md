# Classification of Online Shoppers Intention

## Summary
This repository contains all the files of my work for the take home assignment. 
The task is to build a machine learning model to predict if a customer will buy a product or not.

## Dataset
This data set contains transactions occurring in an online store (E-commerce).
Each row in the dataset contains a feature vector that contains data corresponding to a visit "session" (period of time spent) of a user on an e-commerce
website.
The total number of sessions in the dataset is 12,330.

The dataset consists of 10 numerical and 8 categorical attributes. The 'Revenue' attribute is used as the class label.

## Data Cleaning and Feature Engineering
The Feature_Engineering.py file contains code to conduct cleaning of the data and add new features to the dataset that helps to provide more information about
each session that are not explicitly provided in the raw dataset.

Z-Score Outlier test is used to identify outliers, and the datapoints are removed from the dataset.
Numerical features are standardized and categorical features are one hot encoded.

Features added:
  - total_pages_viewed : count of pages visited in that session
  - Administrative_% : Percentage of administrative related pages viewed
  - Informational_% : Percentage of informational related pages viewed
  - ProductRelated_% : Percentage of product related pages viewed
  - total_duration : Total time spent in that session
  - Administrative_Duration_% : Percentage of time spent on administrative pages
  - Informational_Duration_% : Percentage of time spent on informational pages
  - ProductRelated_Duration_% : Percentage of time spent on product related pages
  - Administrative_Duration_avg : Average time spent on each administrative pages
  - Informational_Duration_avg : Average time spent on each informational pages
  - ProductRelated_Duration_avg : Average time spent on each product related pages
  - page_values_x_bounce_rate : PageValues  (1 - BounceRates)
  - yearQuarter : Which quarter of the year does the session belong to
 
 ## Balancing the Dataset
 Almost 85% of the datapoints belong to class False ('Revenue'), which indicates class imbalance issue in the dataset. Machine learning models are sensitive
 to class imbalance and predictions will be dominated by giving more importance to majority classes, i.e, it will find it hard to classify minority class 
 samples.
 
 Synthetic datapoints are generated to reduce the bias towards majority class.
 
 ### SMOTE-NC (Synthetic Minority Over-sampling Technique for Nominal and Continuous)
 SMOTE-NC is derived to handle dataset with both numerical and categorical features. It works by fitting a line between a minority sample and k neighbors
 to that sample, and then samples n points from that line
 
 ### RBO (Radial Based Oversampling)
 
 RBO finds regions in which the synthetic objects from minority class should be generated on the basis of the imbalance distribution estimation with
 radial basis functions. A random minority sample is selected and then we move in a random direction with a specified step size from that point. Then
 the potential of that point belonging to minority class is calculated and if it the crieteria that point is added to the minority class.  
 
 ### Near-Miss Undersampling
 
 Undersampling techniques are used to eliminate some of the samples from the majority class to reduce bias. When instances of the two classes are
 very close to each other, we remove the instances of the majority class to increase the spaces between the two classes. This also overcomes the noise
 generated in oversampling techniques.
 
 Near-Miss algorithm is first used to slightly reduce the number of points in the majority class and then SMOTE-NC and RBO techniques are used seperately
 to sample extra minority points.
 
 ## Feature Selection
 
There are a total of 89 features that are passed to the model after one-hot encoding. Feature selection methods are used to remove highly correlated features that impacts the models performance. 30 features are selected after feature selection.

### Tree’s Feature Importance from Mean Decrease in Impurity (MDI)

The impurity-based feature importance ranks the numerical features to be the most important features.
Gini importance (or mean decrease impurity), is computed from the Random Forest structure. For each feature we can collect how on average it decreases the impurity. The average over all trees in the forest is the measure of the feature importance.

### Permutation Based Feature Importance

MDI method has few disadvantages. They are biased towards high cardinality features. They are computed on training set statistics and therefore do not reflect the ability of feature to be useful to make predictions that generalize to the test set (when the model has enough capacity)
Permutation based method overcomes the disadvantages. This method will randomly shuffle each feature and compute the change in the model’s performance. The features which impact the performance the most are the most important one.

### Assesing the co-efficients of Logistic Regression Decision boundary

The weights that are learned during training logistic regression to fit a decision boundary to seprate the classes, can be used to measure the importance
of features in seperating the classes. The features are ranked based on the absolute values of their co-efficients.

### mRMR - Minimum Redundancy - Maximum Relevance

The mRMR is a feature selection approach that tends to select features with a high correlation with the class (output) and a low correlation between themselves. For continuous features, the F-statistic can be used to calculate correlation with the class (relevance) and the Pearson correlation coefficient can be used to calculate correlation between features (redundancy). Thereafter, features are selected one by one by applying a greedy search to maximize the objective function, which is a function of relevance and redundancy.

## Modeling

The task in hand is a binary classification task and the follwing models are trained and evaluated:
  - Logistic Regression
  - SVM
  - Naive Bayes
  - Random Forest
  - XGBoost
  - MLP
The following metrics are reported for both training and test set:
  - Accuracy
  - Precision
  - Recall
 
 ### Performance of models on Un-Balanced Data:
 
 | Model | Train Accuracy | Train Precision | Train Recall |
 | :--   | :--            | :--             | :--          |
 | SVM.  |                |                 |              |
