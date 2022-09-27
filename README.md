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
 
 #### SMOTE-NC (Synthetic Minority Over-sampling Technique for Nominal and Continuous)
 SMOTE-NC is derived to handle dataset with both numerical and categorical features. It works by fitting a line between a minority sample and k neighbors
 to that sample, and then samples n points from that line
 
 #### RBO (Radial Based Oversampling)
 
 RBO finds regions in which the synthetic objects from minority class should be generated on the basis of the imbalance distribution estimation with
 radial basis functions.
 
 #### Near-Miss Undersampling
 
 Undersampling techniques are used to eliminate some of the samples from the majority class to reduce bias. When instances of the two classes are
 very close to each other, we remove the instances of the majority class to increase the spaces between the two classes. This also overcomes the noise
 generated in oversampling techniques.
 
 Near-Miss algorithm is first used to slightly reduce the number of points in the majority class and then SMOTE-NC and RBO techniques are used seperately
 to sample extra minority points.
 
 ## Feature Selection
