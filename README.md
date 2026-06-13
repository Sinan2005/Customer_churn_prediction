For your **Customer Churn Prediction** project, you can create a README in the same detailed style as the Hybrid RAG project.

Also, since this is based on the uploaded notebook/code, I'll reference it. 

# Customer Churn Prediction using Machine Learning

## Overview

Customer churn is one of the most important business problems faced by subscription-based companies. Customer churn occurs when a customer stops using a company's products or services.

Acquiring new customers is often significantly more expensive than retaining existing ones. Therefore, being able to identify customers who are likely to leave can help organizations take preventive actions and improve customer retention.

This project develops a machine learning system capable of predicting customer churn using customer demographic information, service subscriptions, billing details, and contract information.

The project involves:

* Data Cleaning
* Exploratory Data Analysis
* Feature Engineering
* Handling Class Imbalance
* Model Training
* Model Evaluation
* Predictive System Development

The final model was trained using a Random Forest Classifier and achieved approximately **77.85% accuracy** on unseen test data. 

---

# Problem Statement

Telecommunication companies often lose customers due to:

* Better competitor offers
* High service costs
* Poor customer support
* Lack of engagement
* Contract expiration

Predicting churn enables businesses to:

* Identify at-risk customers
* Launch targeted retention campaigns
* Improve customer satisfaction
* Reduce revenue loss

The goal of this project is to predict whether a customer will churn based on historical customer data.

---

# Dataset Information

Dataset:

**Telco Customer Churn Dataset**

Number of Records:

```text
7043 Customers
```

Number of Features:

```text
20 Features
```

Target Variable:

```text
Churn
```

Classes:

```text
Yes
No
```

---

# Technology Stack

## Programming Language

* Python

## Libraries

### Data Processing

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-Learn
* XGBoost
* Imbalanced-Learn (SMOTE)

### Model Serialization

* Pickle

---

# Project Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Encoding
   ↓
Train-Test Split
   ↓
SMOTE Oversampling
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Saving
   ↓
Prediction System
```

---

# Data Preprocessing

## Removing Irrelevant Features

The dataset contains:

```text
customerID
```

This feature uniquely identifies customers but does not contribute to churn prediction.

Therefore it was removed before training. 

```python
df = df.drop(columns=["customerID"])
```

---

## Handling Missing Values

The `TotalCharges` column contained blank values.

Example:

```text
TotalCharges = " "
```

A total of:

```text
11 rows
```

contained missing values. 

These values were replaced with:

```python
0.0
```

and converted into numerical format.

```python
df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0"})
df["TotalCharges"] = df["TotalCharges"].astype(float)
```

---

# Exploratory Data Analysis

EDA was performed to understand:

* Data distribution
* Customer characteristics
* Class imbalance
* Feature relationships

---

## Numerical Features

The following numerical features were analyzed:

```text
tenure
MonthlyCharges
TotalCharges
```

Visualization techniques:

* Histograms
* KDE Plots
* Boxplots
* Correlation Heatmaps

---

## Tenure Analysis

Tenure represents the number of months a customer has stayed with the company.

Observations:

* Many customers have low tenure
* Long-term customers are also present
* Useful predictor of churn

---

## Monthly Charges Analysis

Monthly charges indicate customer spending.

Observations:

* Values range approximately from:

```text
18 → 119
```

* Higher charges may increase churn probability

---

## Total Charges Analysis

Represents cumulative spending.

Observations:

* Strongly correlated with tenure
* Useful indicator of customer value

---

## Correlation Analysis

A correlation heatmap was generated for:

```text
tenure
MonthlyCharges
TotalCharges
```

Key finding:

```text
tenure ↔ TotalCharges
```

showed strong positive correlation because customers who stay longer generally pay more over time. 

---

# Class Imbalance Problem

Target distribution:

```text
No Churn : 5174
Churn    : 1869
```

The dataset is significantly imbalanced. 

Problems caused by imbalance:

* Model becomes biased toward majority class
* Poor minority class detection
* Misleading accuracy scores

---

# Handling Class Imbalance using SMOTE

To address class imbalance, the project uses:

## SMOTE

Synthetic Minority Oversampling Technique

SMOTE generates synthetic examples for the minority class.

Before SMOTE:

```text
No Churn : 4138
Churn    : 1496
```

After SMOTE:

```text
No Churn : 4138
Churn    : 4138
```

The training dataset became perfectly balanced. 

---

# Feature Encoding

Machine learning models require numerical input.

Categorical features were encoded using:

```python
LabelEncoder()
```

Features encoded:

* Gender
* Partner
* Dependents
* InternetService
* Contract
* PaymentMethod
* PhoneService
* OnlineSecurity
* TechSupport
* StreamingTV
* StreamingMovies

and others. 

The encoders were saved using:

```python
pickle.dump()
```

allowing consistent preprocessing during inference.

---

# Model Training

Three machine learning models were evaluated.

---

## Decision Tree Classifier

Cross-validation accuracy:

```text
78%
```

---

## Random Forest Classifier

Cross-validation accuracy:

```text
84%
```

Highest performing model. 

---

## XGBoost Classifier

Cross-validation accuracy:

```text
83%
```

---

# Model Selection

Comparison:

| Model         | CV Accuracy |
| ------------- | ----------- |
| Decision Tree | 78%         |
| Random Forest | 84%         |
| XGBoost       | 83%         |

Random Forest achieved the highest performance and was selected as the final model. 

---

# Model Evaluation

The final Random Forest model was evaluated on the test set.

## Accuracy

```text
77.85%
```

---

## Confusion Matrix

```text
[[878 158]
 [154 219]]
```

Meaning:

* 878 customers correctly predicted as non-churn
* 219 customers correctly predicted as churn
* Some false positives and false negatives remain

---

## Classification Report

### Non-Churn

Precision:

```text
0.85
```

Recall:

```text
0.85
```

---

### Churn

Precision:

```text
0.58
```

Recall:

```text
0.59
```

This indicates the model is reasonably effective at identifying churn customers but still has room for improvement. 

---

# Model Deployment Preparation

The trained model was serialized using:

```python
pickle.dump()
```

Saved file:

```text
customer_churn_model.pkl
```

The saved object contains:

* Trained Random Forest Model
* Feature Names

This enables future deployment without retraining.

---

# Predictive System

A prediction pipeline was built to classify new customers.

Workflow:

```text
Customer Details
      ↓
Feature Encoding
      ↓
Random Forest Model
      ↓
Prediction
      ↓
Churn / No Churn
```

Example Prediction:

```text
Prediction:
No Churn

Probability:
78%
```

The system also returns churn probabilities using:

```python
predict_proba()
```

allowing businesses to rank customers by churn risk. 

---

# Challenges Faced

## Class Imbalance

The dataset contained far more non-churn customers than churn customers.

Solution:

```text
SMOTE Oversampling
```

---

## Missing Values

Blank entries existed in:

```text
TotalCharges
```

Solution:

```text
Replace with 0 and convert to float
```

---

## Categorical Data

Most features were non-numeric.

Solution:

```text
Label Encoding
```

---

# Future Improvements

Potential enhancements include:

* Hyperparameter Tuning
* Grid Search CV
* Random Search CV
* Stratified K-Fold Validation
* Feature Selection
* Model Explainability using SHAP
* Deep Learning Models
* Customer Churn Dashboard
* Streamlit Deployment
* Real-Time Prediction API

---

# Key Learnings

Through this project I gained practical experience in:

* Data Cleaning
* Exploratory Data Analysis
* Feature Engineering
* Class Imbalance Handling
* SMOTE
* Machine Learning Classification
* Random Forest
* XGBoost
* Cross Validation
* Model Evaluation
* Model Serialization
* Predictive System Development

---

# Conclusion

This project successfully developed a customer churn prediction system using machine learning techniques. Through extensive preprocessing, exploratory analysis, class imbalance handling, and model comparison, a Random Forest model was selected as the final solution.

The model achieved approximately **77.85% test accuracy** and demonstrated the ability to identify customers at risk of churning. Such systems can help businesses improve customer retention, reduce revenue loss, and make data-driven decisions.

---

## Author

**Sinan Tamake**

Areas of Interest:

* Machine Learning
* Data Science
* Generative AI
* MLOps
* Predictive Analytics
* Deep Learning
