# 🚗 Road Traffic Accident Severity Prediction

## 📌 Overview
This project focuses on predicting the **severity of road traffic accidents** using machine learning techniques. The objective is to analyze multiple influencing factors such as driver details, road conditions, weather, and vehicle characteristics to classify accident severity.

The system combines **data preprocessing, exploratory data analysis (EDA), feature engineering, and ensemble learning models** to build an accurate and reliable prediction pipeline.

---

## 📊 Dataset Information

- 📁 Dataset: Road Traffic Accident (RTA)
- 📌 Total Samples: ~12,000
- 📌 Total Features: 32
- 🎯 Target Variable: `accident_severity`

### 🔑 Key Feature Categories:
- Driver Information (age, sex, experience)
- Road & Environment (surface, alignment, weather, light)
- Vehicle Information
- Casualty Details
- Time-based Features (hour)

---

## 🔍 Exploratory Data Analysis (EDA)

### 📈 Key Insights:

- Most accidents:
  - Occur between **3 PM – 7 PM**
  - Involve **2 vehicles and 2 casualties**
  - Happen on **two-way roads under normal weather**

- Driver trends:
  - Majority are **male drivers aged 18–30**
  - Have **moderate driving experience (5–10 years)**

- Severity patterns:
  - Most accidents result in **slight injuries**
  - Severe and fatal cases increase in **low-light conditions**

---

## ⚙️ Data Preprocessing

- Missing values handled using **mode imputation**
- Removed irrelevant columns
- Extracted `hour` from time feature
- Applied **ordinal encoding** for categorical variables
- Handled class imbalance using **SMOTE**

---

## 🤖 Models Used

### 🌳 Random Forest
- Ensemble of decision trees
- Reduces overfitting
- Provides feature importance

---

### 🚀 XGBoost
- Gradient boosting algorithm
- Learns from previous errors
- High performance on structured data

---

### 🌲 Extra Trees
- Highly randomized trees
- Faster than Random Forest
- Reduces variance

---

### 🔁 AdaBoost (with Random Forest)
- Boosting approach
- Improves weak learners iteratively

---

## 📈 Model Evaluation

### 📊 Confusion Matrix

The confusion matrix helps visualize model performance by comparing actual vs predicted values.

![Confusion Matrix](images/newplot(1).png)

---

### 📏 Evaluation Metrics

- **Accuracy** → Overall correctness  
- **Precision** → Correct positive predictions  
- **Recall** → Ability to detect actual positives  
- **F1-score** → Balance between precision & recall  

---

## 🏆 Results

- Random Forest achieved the best overall performance
- Ensemble methods significantly improved prediction accuracy to 82%
- Feature selection using importance improved efficiency

---

## 🔍 Feature Importance

Top contributing features:
- Number of vehicles involved
- Number of casualties
- Road conditions
- Driver age
- Cause of accident

---

## 💾 Model Saving

```python
import joblib
joblib.dump(rf, 'models/Baseline_RandomForest_final.joblib')
