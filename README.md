# StreamlitAppimplentation
### 📌 a) Problem Statement

The objective of this project is to predict the presence of heart disease using multiple machine learning classification models and compare their performance using various evaluation metrics.

### 📌 b) Dataset Description

Dataset: Alternative UCI Heart Disease Dataset
Source: Kaggle
Number of Instances: 1000+
Number of Features: 13
Target Variable:

0 → No Heart Disease

1 → Heart Disease

The dataset contains patient medical attributes such as age, cholesterol, resting blood pressure, maximum heart rate, etc.
### 📌 c) Models Used

Logistic Regression

Decision Tree

K-Nearest Neighbors

Naive Bayes
### 📌 Comparison Table

| ML Model            | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
| ------------------- | -------- | ------ | --------- | ------ | ------ | ------ |
| Logistic Regression | 0.7951   | 0.7949 | 0.7699    | 0.8447 | 0.8056 | 0.5929 |
| Decision Tree       | 0.9854   | 0.9854 | 1.0000    | 0.9709 | 0.9852 | 0.9712 |
| KNN                 | 0.8634   | 0.8634 | 0.8571    | 0.8738 | 0.8654 | 0.7269 |
| Naive Bayes         | 0.6780   | 0.6769 | 0.6209    | 0.9223 | 0.7422 | 0.4065 |
| Random Forest       | 0.9854   | 0.9854 | 1.0000    | 0.9709 | 0.9852 | 0.9712 |
| XGBoost             | 0.9854   | 0.9854 | 1.0000    | 0.9709 | 0.9852 | 0.9712 |


### 📌 Observations Section
Logistic Regression performed moderately well with balanced precision and recall.

Decision Tree, Random Forest, and XGBoost achieved the highest performance with MCC ≈ 0.97.

Naive Bayes showed lower MCC (0.40), indicating weaker overall correlation.

Ensemble models (Random Forest & XGBoost) provided the best generalization.

High MCC score for ensemble models indicates strong predictive stability.
