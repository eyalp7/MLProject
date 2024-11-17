# Diabetes Prediction Using Machine Learning

This project focuses on predicting the likelihood of diabetes in individuals based on several health-related parameters using machine learning techniques. The dataset used includes various factors like age, gender, smoking history, blood glucose levels, and more. The goal is to train different machine learning models and evaluate their performance in predicting whether a person has diabetes.

## Table of Contents

- [Project Overview](#project-overview)
- [Libraries Used](#libraries-used)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Models and Evaluation](#models-and-evaluation)
- [Results](#results)
- [How to Run](#how-to-run)

## Project Overview

This project aims to classify individuals into two categories:
1. **Diabetic (1)** – The individual has diabetes.
2. **Non-diabetic (0)** – The individual does not have diabetes.

We utilize multiple machine learning algorithms to train a predictive model and assess their performance on a dataset. 

## Libraries Used

This project uses the following libraries:

- **matplotlib**: For data visualization and plotting graphs.
- **numpy**: For numerical operations.
- **pandas**: For data manipulation and analysis.
- **sklearn**: For machine learning models, data preprocessing, and evaluation metrics.
- **google.colab**: For working with Google Drive in Colab.

```python
import matplotlib.pyplot as plt
import numpy as np
from google.colab import drive
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_score, accuracy_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm
```

## Dataset

The dataset used in this project is the **Diabetes Prediction Dataset**. It contains several health-related features for each individual, including:

- **Gender** (Male/Female)
- **Age** (Age of the individual)
- **Hypertension** (Whether the individual has hypertension)
- **Heart Disease** (Whether the individual has heart disease)
- **Smoking History** (Whether the individual has a smoking history)
- **BMI** (Body Mass Index)
- **HbA1c Level** (A marker of long-term blood glucose control)
- **Blood Glucose Level** (Blood sugar levels)

### Sample Data

| gender  | age  | hypertension | heart_disease | smoking_history | bmi   | HbA1c_level | blood_glucose_level | diabetes |
|---------|------|--------------|---------------|-----------------|-------|-------------|---------------------|----------|
| Female  | 80.0 | 0            | 1             | never           | 25.19 | 6.6         | 140                 | 0        |
| Female  | 54.0 | 0            | 0             | No Info         | 27.32 | 6.6         | 80                  | 0        |
| Male    | 28.0 | 0            | 0             | never           | 27.32 | 5.7         | 158                 | 0        |
| Female  | 36.0 | 0            | 0             | current         | 23.45 | 5.0         | 155                 | 0        |

## Data Preprocessing

In this project, we perform the following preprocessing steps:

1. **Removing Duplicate Entries**: Duplicate rows in the dataset are dropped.
2. **Encoding Categorical Variables**: Categorical columns such as `gender` and `smoking_history` are encoded into numerical values using `LabelEncoder`.
3. **Normalization**: Feature scaling is done by normalizing the data to a range of 0-1.
4. **Splitting the Dataset**: The dataset is split into training and testing sets for model evaluation.

## Models and Evaluation

We evaluate the performance of the following models:

- **K-Nearest Neighbors (KNN)**: A simple and effective classification algorithm.
- **Logistic Regression**: A linear model for binary classification.
- **Support Vector Machine (SVM)**: A powerful model that works well for classification tasks.

Each model is trained multiple times, and performance metrics such as **precision**, **accuracy**, and **recall** are computed.

### Example of model evaluation output:

```text
model name: knn 
Average Precision:  0.9651282051282051 
Average Accuracy:  0.9576703068122724 
Average Recall:  0.546775130737943 
Confusion Matrix: 
[[17475    34]
 [  780   941]]
Classification Report: 
               precision    recall  f1-score   support
         0.0       0.96      1.00      0.98     17509
         1.0       0.97      0.55      0.70      1721
```

## Results

The models were evaluated based on their ability to predict whether an individual has diabetes or not. The **KNN model** performed with high accuracy but had a relatively low recall, indicating that it could miss some diabetic cases. Below are the results for each model:

### K-Nearest Neighbors (KNN)
- **Precision**: 96.5%
- **Accuracy**: 95.8%
- **Recall**: 54.7%

Confusion Matrix:
```text
[[17475    34]
 [  780   941]]
```
Classification report:
               precision    recall  f1-score   support
         0.0       0.96      1.00      0.98     17509
         1.0       0.97      0.55      0.70      1721


### Logistic Regression
- **Precision**: 95.2%
- **Accuracy**: 94.7%
- **Recall**: 62.3%

Confusion Matrix:
```text
[[17470    39]
 [  650  1073]]
```

Classification Report:

```text
               precision    recall  f1-score   support
         0.0       0.96      1.00      0.98     17509
         1.0       0.96      0.55      0.70      1721
```
### Support Vector Machine (SVM)
- **Precision**: 97.0%
- **Accuracy**: 96.3%
- **Recall**: 55.2%

Confusion Matrix:
```text
[[17483    26]
 [  774   947]]
```
Classification Report:
```text
               precision    recall  f1-score   support
         0.0       0.96      1.00      0.98     17509
         1.0       0.97      0.55      0.70      1721
```
## How to Run

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/yourusername/diabetes-prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd diabetes-prediction
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Script:
   ```bash
   python diabetes_prediction.py
   ```
  
This will train the models and output the results, including metrics like accuracy, precision, recall, and confusion matrices for each classifier.

5. Alternatively, you can run the notebook in Google Colab by uploading it to your Google Drive. Once uploaded, run the cells sequentially to execute the entire workflow.
