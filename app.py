import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# -------------------------------
# Title & Description
# -------------------------------
st.title("❤️ Heart Disease Prediction System")

st.markdown("""
This application predicts the presence of heart disease using **six Machine Learning models**.
Upload a dataset to evaluate model performance.
""")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("📌 About This Project")

st.sidebar.info("""
This project implements:

• Logistic Regression  
• Decision Tree  
• K-Nearest Neighbors  
• Naive Bayes  
• Random Forest  
• XGBoost  

Evaluation Metrics Used:

• Accuracy  
• AUC  
• Precision  
• Recall  
• F1 Score  
• MCC  
""")

# -------------------------------
# Model Comparison Table
# -------------------------------
st.subheader("📊 Model Performance Comparison")

try:
    results_df = pd.read_csv("models/model_results.csv")
    st.dataframe(results_df)
except:
    st.warning("Model comparison file not found.")

# -------------------------------
# File Upload Section
# -------------------------------
st.subheader("📂 Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV file containing heart disease data",
    type=["csv"]
)

# -------------------------------
# Model Selection
# -------------------------------
st.subheader("🤖 Choose Machine Learning Model")

model_choice = st.selectbox(
    "Select a model for prediction",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

# -------------------------------
# Prediction Section
# -------------------------------
if uploaded_file:
    data = pd.read_csv(uploaded_file)

    st.write("### 🔎 Uploaded Data Preview")
    st.dataframe(data.head())

    # Load model, scaler, and columns
    model = joblib.load(f"model/{model_choice}.pkl")
    scaler = joblib.load("model/scaler.pkl")
    train_columns = joblib.load("model/columns.pkl")

    # Split features & target
    X = data.drop("target", axis=1)
    y = data["target"]

    # Apply same encoding
    X = pd.get_dummies(X, drop_first=True)

    # Add missing columns
    for col in train_columns:
        if col not in X.columns:
            X[col] = 0

    # Ensure same column order
    X = X[train_columns]

    # Apply scaling
    X = scaler.transform(X)

    # Make predictions
    predictions = model.predict(X)

    st.success("✅ Prediction completed successfully!")

    # -------------------------------
    # Classification Report
    # -------------------------------
    st.subheader("📄 Classification Report")

    report = classification_report(y, predictions, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    st.dataframe(report_df)

    # -------------------------------
    # Confusion Matrix
    # -------------------------------
    st.subheader("📌 Confusion Matrix")

    cm = confusion_matrix(y, predictions)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Disease", "Disease"],
        yticklabels=["No Disease", "Disease"]
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
