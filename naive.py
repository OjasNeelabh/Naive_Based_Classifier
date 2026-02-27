import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

st.set_page_config(page_title="Naive Bayes Classifier", layout="centered")

st.title("📊 Naive Bayes Classifier App")

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    columns = df.columns.tolist()

    # Target variable
    target = st.selectbox("Select Target Variable", columns)

    # Feature selection
    features = st.multiselect("Select Feature Columns", 
                              [col for col in columns if col != target])

    # Train test split
    test_size = st.slider("Select Test Size (%)", 10, 50, 20)
    test_size = test_size / 100

    if st.button("Evaluate Model"):

        if len(features) == 0:
            st.warning("Please select at least one feature.")
        else:
            X = df[features]
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            model = GaussianNB()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)

            st.subheader("Model Accuracy")
            st.write(f"Accuracy: {round(acc * 100, 2)}%")

            # Confusion Matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)

            fig, ax = plt.subplots()
            ConfusionMatrixDisplay(cm).plot(ax=ax)
            st.pyplot(fig)
