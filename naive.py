import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    r2_score,
    mean_squared_error,
    roc_curve,
    auc
)

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="ML Studio", layout="wide")

st.markdown("""
<style>
.main-title {
    font-size:40px !important;
    font-weight:700;
}
.metric-box {
    padding:15px;
    border-radius:10px;
    background-color:#1f2937;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🚀 ML Studio</p>', unsafe_allow_html=True)
st.caption("Clean. Interactive. Powerful.")

uploaded_file = st.file_uploader("", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    # Sidebar Controls
    st.sidebar.header("⚙️ Configuration")

    task = st.sidebar.selectbox("Task", ["Classification", "Regression"])

    target = st.sidebar.selectbox("Target Variable", df.columns)

    features = st.sidebar.multiselect(
        "Feature Variables",
        [col for col in df.columns if col != target]
    )

    test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)

    scale = st.sidebar.checkbox("Apply Scaling")

    tabs = st.tabs(["📊 Visuals", "🤖 Model", "📈 Insights"])

    # -----------------------------------
    # VISUAL TAB
    # -----------------------------------

    with tabs[0]:

        if numeric_cols:

            col1, col2 = st.columns(2)

            with col1:
                fig = px.scatter_matrix(
                    df,
                    dimensions=numeric_cols,
                    color=target if target in categorical_cols else None,
                    title="Feature Relationships"
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig2 = px.box(
                    df,
                    x=target if target in categorical_cols else None,
                    y=numeric_cols[0],
                    title="Distribution Overview"
                )
                st.plotly_chart(fig2, use_container_width=True)

    # -----------------------------------
    # MODEL TAB
    # -----------------------------------

    with tabs[1]:

        if features:

            X = df[features]
            y = df[target]

            X = pd.get_dummies(X, drop_first=True)

            if task == "Classification" and y.dtype == "object":
                le = LabelEncoder()
                y = le.fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            if scale:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            if task == "Classification":

                model_choice = st.selectbox(
                    "Model",
                    ["Naive Bayes", "Logistic Regression"]
                )

                model = GaussianNB() if model_choice == "Naive Bayes" else LogisticRegression(max_iter=1000)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                acc = accuracy_score(y_test, y_pred)

                st.metric("Accuracy", f"{round(acc*100,2)}%")

                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                st.pyplot(fig)

                if len(np.unique(y)) == 2:
                    y_probs = model.predict_proba(X_test)[:,1]
                    fpr, tpr, _ = roc_curve(y_test, y_probs)
                    roc_auc = auc(fpr, tpr)

                    fig2 = px.area(
                        x=fpr, y=tpr,
                        title=f"ROC Curve (AUC = {round(roc_auc,2)})"
                    )
                    st.plotly_chart(fig2, use_container_width=True)

            else:

                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)

                col1, col2 = st.columns(2)
                col1.metric("R² Score", round(r2,4))
                col2.metric("MSE", round(mse,4))

                fig = px.scatter(
                    x=y_test,
                    y=y_pred,
                    labels={"x":"Actual", "y":"Predicted"},
                    title="Actual vs Predicted"
                )
                st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------
    # INSIGHTS TAB
    # -----------------------------------

    with tabs[2]:

        if numeric_cols:
            corr = df[numeric_cols].corr()

            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(corr, cmap="coolwarm", annot=True)
            st.pyplot(fig)

            st.markdown("### Feature Correlation Insight")
            st.write(
                "Higher absolute correlation values indicate stronger relationships between features."
            )
