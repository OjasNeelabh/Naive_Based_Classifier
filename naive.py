import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    r2_score,
    mean_squared_error
)

st.set_page_config(page_title="ML Playground App", layout="centered")

st.title("🤖 Machine Learning Playground")

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    columns = df.columns.tolist()

    # Task selection
    task = st.selectbox("Select Task Type", ["Classification", "Regression"])

    # Target variable
    target = st.selectbox("Select Target Variable", columns)

    # Feature selection
    features = st.multiselect(
        "Select Feature Columns",
        [col for col in columns if col != target]
    )

    # Train/Test split
    test_size = st.slider("Select Test Size (%)", 10, 50, 20) / 100

    if st.button("Train & Evaluate"):

        if len(features) == 0:
            st.warning("Please select at least one feature.")
        else:
            X = df[features]
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            # =========================
            # CLASSIFICATION
            # =========================
            if task == "Classification":

                model = GaussianNB()
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                acc = accuracy_score(y_test, y_pred)

                st.subheader("📊 Classification Results")
                st.write(f"Accuracy: {round(acc * 100, 2)}%")

                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)

                fig, ax = plt.subplots()
                ConfusionMatrixDisplay(cm).plot(ax=ax)
                st.pyplot(fig)

            # =========================
            # REGRESSION
            # =========================
            else:

                model = LinearRegression()
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)

                st.subheader("📈 Regression Results")
                st.write(f"R² Score: {round(r2, 4)}")
                st.write(f"Mean Squared Error: {round(mse, 4)}")

                # Plot actual vs predicted
                fig, ax = plt.subplots()
                ax.scatter(y_test, y_pred)
                ax.set_xlabel("Actual Values")
                ax.set_ylabel("Predicted Values")
                ax.set_title("Actual vs Predicted")
                st.pyplot(fig)
