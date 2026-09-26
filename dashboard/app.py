import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import joblib


# =============================
# Project Paths
# =============================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "student_performance_cleaned.csv"
FEATURE_PATH = BASE_DIR / "data" / "processed" / "student_performance_features.csv"
EVALUATION_PATH = BASE_DIR / "reports" / "model_evaluation.csv"
PREDICTIONS_PATH = BASE_DIR / "reports" / "test_predictions.csv"
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"


# =============================
# Page Configuration
# =============================

st.set_page_config(
    page_title="Student Performance AI Dashboard",
    page_icon="🎓",
    layout="wide"
)


# =============================
# Load Data
# =============================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_features():
    return pd.read_csv(FEATURE_PATH)


@st.cache_data
def load_evaluation():
    return pd.read_csv(EVALUATION_PATH)


@st.cache_data
def load_predictions():
    return pd.read_csv(PREDICTIONS_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


df = load_data()
features_df = load_features()
evaluation = load_evaluation()
predictions = load_predictions()
model = load_model()


# =============================
# Header
# =============================

st.title("🎓 AI-Assisted Student Performance Dashboard")

st.markdown(
    """
    **National Online Education Platform — Student Performance Analysis**

    This dashboard demonstrates an AI-assisted data pipeline for
    preprocessing student data, evaluating machine learning models,
    and predicting academic performance.
    """
)


# =============================
# Dataset Overview
# =============================

st.header("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Students", len(df))

with col2:
    st.metric("Model Features", 39)

with col3:
    st.metric("Average Final Grade", round(df["G3"].mean(), 2))

with col4:
    st.metric("Maximum Grade", int(df["G3"].max()))


# =============================
# Dataset Preview
# =============================

st.subheader("Student Dataset Preview")

st.dataframe(
    df.head(10),
    width="stretch"
)


# =============================
# Grade Distribution
# =============================

st.subheader("Final Grade Distribution")

fig_grade = px.histogram(
    df,
    x="G3",
    nbins=20,
    title="Distribution of Final Grades"
)

st.plotly_chart(fig_grade, width="stretch")


# =============================
# Model Evaluation
# =============================

st.header("🤖 Model Evaluation")

st.dataframe(
    evaluation.round(4),
    width="stretch"
)


# =============================
# Model Error Comparison
# =============================

fig_metrics = px.bar(
    evaluation,
    x="Model",
    y=["MAE", "RMSE"],
    barmode="group",
    title="Model Error Comparison"
)

st.plotly_chart(fig_metrics, width="stretch")


# =============================
# R2 Comparison
# =============================

fig_r2 = px.bar(
    evaluation,
    x="Model",
    y="R2",
    title="R² Model Comparison"
)

st.plotly_chart(fig_r2, width="stretch")


# =============================
# Actual vs Predicted
# =============================

st.header("🎯 Actual vs Predicted Grades")

fig_prediction = px.scatter(
    predictions,
    x="Actual_G3",
    y="Predicted_G3",
    title="Actual vs Predicted Final Grades"
)

fig_prediction.add_shape(
    type="line",
    x0=0,
    y0=0,
    x1=20,
    y1=20
)

st.plotly_chart(fig_prediction, width="stretch")


# =============================
# Feature Importance
# =============================

st.header("🔍 Feature Importance")

if hasattr(model, "feature_importances_"):

    importance_df = pd.DataFrame({
        "Feature": features_df.drop(columns=["G3"]).columns,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    ).head(15)

    fig_importance = px.bar(
        importance_df.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 15 Important Features"
    )

    st.plotly_chart(fig_importance, width="stretch")


# =============================
# Student Prediction
# =============================

st.header("🔮 Student Performance Prediction")

st.info(
    "Enter basic student information to generate an estimated final grade."
)

col1, col2, col3 = st.columns(3)

with col1:
    studytime = st.slider(
        "Study Time",
        min_value=1,
        max_value=4,
        value=2
    )

    failures = st.slider(
        "Past Failures",
        min_value=0,
        max_value=3,
        value=0
    )

    absences = st.slider(
        "Absences",
        min_value=0,
        max_value=75,
        value=5
    )

with col2:
    age = st.slider(
        "Age",
        min_value=15,
        max_value=22,
        value=17
    )

    traveltime = st.slider(
        "Travel Time",
        min_value=1,
        max_value=4,
        value=1
    )

    freetime = st.slider(
        "Free Time",
        min_value=1,
        max_value=5,
        value=3
    )

with col3:
    health = st.slider(
        "Health",
        min_value=1,
        max_value=5,
        value=3
    )

    goout = st.slider(
        "Going Out",
        min_value=1,
        max_value=5,
        value=3
    )

    study_support = st.selectbox(
        "School Support",
        ["No", "Yes"]
    )


if st.button("Predict Final Grade"):

    # Create default feature row
    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=features_df.drop(columns=["G3"]).columns
    )

    # Numeric features
    numeric_values = {
        "age": age,
        "traveltime": traveltime,
        "studytime": studytime,
        "failures": failures,
        "freetime": freetime,
        "goout": goout,
        "health": health,
        "absences": absences
    }

    for feature, value in numeric_values.items():
        if feature in input_data.columns:
            input_data[feature] = value

    # School support
    support_column = "schoolsup_yes"

    if support_column in input_data.columns:
        input_data[support_column] = (
            1 if study_support == "Yes" else 0
        )

    prediction = model.predict(input_data)[0]

    prediction = max(0, min(20, prediction))

    st.success(
        f"Estimated Final Grade: **{prediction:.2f} / 20**"
    )


# =============================
# Footer
# =============================

st.markdown("---")

st.caption(
    "AI-Assisted Data Pipeline & Model Evaluation Dashboard | "
    "Week 3 SDC Internship Project"
)