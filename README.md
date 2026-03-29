# 📊 Customer Churn Prediction using Machine Learning

## 🚀 Overview

This project predicts whether a customer will churn (leave a telecom service) using machine learning models.
The goal is to help businesses identify at-risk customers and improve retention strategies.

---

## 📁 Dataset

* **Dataset:** Telco Customer Churn Dataset
* Total samples: **7043 rows, 21 features**
* Target variable: **Churn (0 = No, 1 = Yes)**

---

## ⚙️ Technologies Used

* Python 🐍
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Matplotlib / Seaborn

---

## 🔍 Project Workflow

### 1. Data Preprocessing

* Handled missing values
* Encoded categorical variables
* Applied feature scaling
* Addressed class imbalance using **SMOTE**

---

### 2. Model Training

Trained and compared multiple models:

* Decision Tree
* Random Forest
* XGBoost

Cross-validation results:

* Decision Tree: ~0.68 – 0.83
* Random Forest: ~0.72 – 0.91 ✅
* XGBoost: ~0.70 – 0.90

👉 **Random Forest performed best overall**

---

### 3. Model Evaluation

Final model: **Random Forest Classifier**

#### 📈 Accuracy

* **Accuracy:** 77.85%

#### 📊 Confusion Matrix

```
[[878 158]
 [154 219]]
```

#### 📋 Classification Report

| Class        | Precision | Recall | F1-score |
| ------------ | --------- | ------ | -------- |
| 0 (No Churn) | 0.85      | 0.85   | 0.85     |
| 1 (Churn)    | 0.58      | 0.59   | 0.58     |

---

## 🧠 Key Insights

* Model performs well for **non-churn customers (Class 0)**
* Performance on **churn class is lower**, indicating class imbalance challenge
* SMOTE helped improve recall for churn prediction

---

## ⚠️ Challenges

* Imbalanced dataset
* Lower recall for churn class
* Need for further optimization

---

## 🚀 Future Improvements

* Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
* Use advanced models (LightGBM, CatBoost)
* Feature engineering
* Deploy using **Streamlit**

---

## ▶️ How to Run

1. Clone the repository:

```bash
git clone https://github.com/your-username/customer-churn-prediction.git
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the notebook:

* Open `.ipynb` file in Jupyter Notebook or Google Colab

---

## 📌 Project Structure

```
customer-churn-prediction/
│
├── customer_churn_prediction.ipynb
├── churn.csv
├── README.md
```

---

## 🎯 Conclusion

The project demonstrates how machine learning can be applied to predict customer churn.
Random Forest achieved the best performance, but further improvements can enhance churn detection.

---

## ⭐ If you like this project

Give it a star ⭐ on GitHub!

---
