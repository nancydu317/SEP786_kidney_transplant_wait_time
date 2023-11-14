# SEP786_kidney_transplant_wait_time   
This repo contains the files used to perform EDA, clean the dataset, and train/test various machine learning models.    

EDA can be found in the file:  
*  __data_EDA__    
*  __CleaningTests__ (within the cleanProcess folder)   

The file __model_training_validation__ contains:  
*  initial linear regression models
*  survival analysis using Cox Proportional Hazard model and Elastic Net penalty

The file __model_and_test__ contains:
*  linear regression
*  linear regression with ridge regularization
*  Lasso Regression
*  Linear Regression with Recursive Feature Elimination (RFECV)
*  Polynomial Regression with GridSearch
*  Support Vector Regression (SVR)
*  SVR with GridSearch
*  K-Nearest Neighbors (KNN)
*  KNN with GridSearch
