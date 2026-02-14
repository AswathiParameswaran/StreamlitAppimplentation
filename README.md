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

The following six Machine Learning classification models were implemented and evaluated on the selected heart disease dataset:

Logistic Regression – A linear classification model used as a baseline for comparison.

Decision Tree Classifier – A tree-based model that splits data using feature-based decision rules.

K-Nearest Neighbors (KNN) – A distance-based classifier that predicts based on nearest data points.

Naive Bayes (Gaussian) – A probabilistic classifier based on Bayes’ theorem with independence assumptions.

Random Forest – An ensemble learning method that combines multiple decision trees for improved performance.

XGBoost (Extreme Gradient Boosting) – An advanced boosting ensemble model that optimizes performance using gradient boosting techniques.

All models were trained and evaluated using the same dataset and preprocessing steps to ensure fair comparison.
### 📌 Comparison Table

| ML Model            | Accuracy | AUC    | Precision | Recall | F1 Score | MCC    |
| ------------------- | -------- | ------ | --------- | ------ | -------- | ------ |
| Logistic Regression | 0.7951   | 0.7949 | 0.7699    | 0.8447 | 0.8056   | 0.5929 |
| Decision Tree       | 0.9854   | 0.9854 | 1.0000    | 0.9709 | 0.9852   | 0.9712 |
| KNN                 | 0.8634   | 0.8634 | 0.8571    | 0.8738 | 0.8654   | 0.7269 |
| Naive Bayes         | 0.6780   | 0.6769 | 0.6209    | 0.9223 | 0.7422   | 0.4065 |
| Random Forest       | 0.9854   | 0.9854 | 1.0000    | 0.9709 | 0.9852   | 0.9712 |
| XGBoost             | 0.9854   | 0.9854 | 1.0000    | 0.9709 | 0.9852   | 0.9712 |

Evaluation metrics used:

1.Accuracy

2.AUC (Area Under ROC Curve)

3.Precision

4.Recall

5.F1 Score

6.Matthews Correlation Coefficient (MCC)


### 📌 Observations Section
1.Logistic Regression demonstrated moderate performance with balanced precision and recall. As a linear model, it provided a strong baseline but was outperformed by tree-based models.

2.Decision Tree, Random Forest, and XGBoost achieved the highest performance across all evaluation metrics. These models recorded an MCC value of approximately 0.97, indicating excellent classification quality and strong correlation between predicted and actual classes.

3.K-Nearest Neighbors (KNN) showed good performance but was slightly less effective compared to ensemble methods, with an MCC of 0.73.

4.Naive Bayes achieved high recall but lower precision and MCC (0.40), suggesting weaker overall predictive balance and higher false positives.

5.Ensemble models (Random Forest and XGBoost) demonstrated superior generalization capability, benefiting from combining multiple decision trees and boosting strategies.

6.The high MCC scores for ensemble models indicate strong predictive stability and balanced performance, making them the most suitable models for this heart disease classification task.
