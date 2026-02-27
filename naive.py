import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    r2_score,
    mean_squared_error,
    roc_curve,
    auc
)

st.set_page_config(page_title="Universal ML Dashboard", layout="wide")
st.title("🤖 Universal Machine Learning Dashboard")

uploaded_file = st.file_uploader("Upload Your CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Overview")
    st.write("Shape:", df.shape)
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    # ===============================
    # DATA VISUALIZATION SECTION
    # ===============================

    st.subheader("📈 Data Visualization")

    col1, col2 = st.columns(2)

    with col1:
        if numeric_cols:
            selected_hist = st.selectbox("Histogram for:", numeric_cols)
            fig = px.histogram(df, x=selected_hist)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if numeric_cols:
            fig, ax = plt.subplots()
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

    if categorical_cols:
        st.subheader("📊 Categorical Distribution")
        cat_col = st.selectbox("Select categorical column:", categorical_cols)
        fig = px.histogram(df, x=cat_col, color=cat_col)
        st.plotly_chart(fig, use_container_width=True)

    # ===============================
    # TASK SELECTION
    # ===============================

    st.subheader("⚙️ Model Configuration")

    task = st.radio("Select Task Type", ["Classification", "Regression"])

    target = st.selectbox("Select Target Variable", df.columns)

    features = st.multiselect(
        "Select Feature Columns",
        [col for col in df.columns if col != target]
    )

    test_size = st.slider("Test Size (%)", 10, 50, 20) / 100

    scale_option = st.checkbox("Apply Feature Scaling (Recommended for Regression)")

    if st.button("Train & Evaluate"):

        if not features:
            st.warning("Please select at least one feature.")
        else:

            X = df[features]
            y = df[target]

            # Encode categorical features
            X = pd.get_dummies(X, drop_first=True)

            # Encode categorical target if classification
            if task == "Classification" and y.dtype == "object":
                le = LabelEncoder()
                y = le.fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            if scale_option:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            # =====================================
            # CLASSIFICATION
            # =====================================

            if task == "Classification":

                model_choice = st.selectbox(
                    "Choose Classification Model",
                    ["Gaussian Naive Bayes", "Logistic Regression"]
                )

                if model_choice == "Gaussian Naive Bayes":
                    model = GaussianNB()
                else:
                    model = LogisticRegression(max_iter=1000)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                acc = accuracy_score(y_test, y_pred)

                st.success(f"Accuracy: {round(acc * 100, 2)}%")

                # Confusion Matrix
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)

                fig, ax = plt.subplots()
                ConfusionMatrixDisplay(cm).plot(ax=ax)
                st.pyplot(fig)

                # Classification Report
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())

                # ROC Curve (Binary only)
                if len(np.unique(y)) == 2:
                    y_probs = model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_probs)
                    roc_auc = auc(fpr, tpr)

                    fig, ax = plt.subplots()
                    ax.plot(fpr, tpr, label=f"AUC = {round(roc_auc, 2)}")
                    ax.plot([0, 1], [0, 1], linestyle="--")
                    ax.set_xlabel("False Positive Rate")
                    ax.set_ylabel("True Positive Rate")
                    ax.set_title("ROC Curve")
                    ax.legend()
                    st.pyplot(fig)

            # =====================================
            # REGRESSION
            # =====================================

            else:

                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)

                st.success(f"R² Score: {round(r2, 4)}")
                st.write(f"Mean Squared Error: {round(mse, 4)}")

                # Actual vs Predicted
                fig, ax = plt.subplots()
                ax.scatter(y_test, y_pred)
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                ax.set_title("Actual vs Predicted")
                st.pyplot(fig)

                # Residual Plot
                residuals = y_test - y_pred
                fig2, ax2 = plt.subplots()
                ax2.scatter(y_pred, residuals)
                ax2.axhline(0, linestyle="--")
                ax2.set_xlabel("Predicted")
                ax2.set_ylabel("Residuals")
                ax2.set_title("Residual Plot")
                st.pyplot(fig2)
