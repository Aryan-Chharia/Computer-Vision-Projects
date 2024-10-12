# Diabetes-Prediction

## Overview

The objective of the dataset is to diagnostically predict if a patient has diabetes, established on definite diagnostic quantities incorporated in the dataset.

## Dataset

The datasets is taken from Kaggle it consists of several medical predictor variables and one target variable, Outcome. Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.

## Model Architecture

Multiple models are used, including Decision Tree, Random Forest, Naive Bayes, Logistic Regression, and Support Vector Classifier.
Feature selection is based on correlation and Random Forest feature importance.
Ensemble methods, like soft-voting with SVM and Logistic Regression, are used to combine models for improved performance.
The final ensemble classifier is built with Linear SVM and Logistic Regression, combining their predictions through a soft-voting strategy.


## Data Processing

The dataset is split into train and test sets, stratified by the outcome variable to maintain the class distribution.
Features (Glucose, BMI, Age, and DiabetesPedigreeFunction) are standardized using StandardScaler.
Correlation analysis and feature importance are used to reduce the dimensionality of input variables.

## Training

Models are trained on the training set using default hyperparameters for simplicity, including Logistic Regression, Naive Bayes, Random Forest, and SVM.
Calibration curves are plotted to assess the reliability of the predicted probabilities.
The accuracy and evaluation metrics such as precision, recall, and F1
score are calculated and compared across different models.

![img2](https://raw.githubusercontent.com/Aryan-Chharia/Computer-Vision-Projects/140a77c50db1fce9d3d6c0daadb10212c2f1a838/Diabetes-Prediction/images/1.PNG)


## Result 
![img](https://raw.githubusercontent.com/Aryan-Chharia/Computer-Vision-Projects/140a77c50db1fce9d3d6c0daadb10212c2f1a838/Diabetes-Prediction/images/4.PNG)
![img1](https://raw.githubusercontent.com/Aryan-Chharia/Computer-Vision-Projects/140a77c50db1fce9d3d6c0daadb10212c2f1a838/Diabetes-Prediction/images/2.PNG))


## Evaluation


![Description of Image](https://raw.githubusercontent.com/Aryan-Chharia/Computer-Vision-Projects/140a77c50db1fce9d3d6c0daadb10212c2f1a838/Diabetes-Prediction/images/3.PNG)






