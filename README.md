# AI-Assisted Data Pipeline & Model Evaluation Dashboard

## Overview

This project is an AI-assisted student performance analysis system developed as part of an SDC internship project.

The system processes student academic and demographic data, performs data cleaning and feature engineering, trains multiple machine learning regression models, evaluates their performance, and presents the results through an interactive Streamlit dashboard.

## Objectives

* Build a complete data preprocessing pipeline
* Perform feature engineering
* Predict students' final academic performance
* Compare multiple machine learning regression models
* Evaluate models using MAE, RMSE, and R²
* Visualize model performance
* Provide an interactive student performance prediction interface

## Dataset

The project uses the UCI Student Performance dataset.

The Portuguese student dataset (`student-por.csv`) is used as the primary modeling dataset.

* Students: 649
* Original attributes: 33
* Model features: 39
* Target variable: Final grade (G3)

`G1` and `G2` were excluded from the model to avoid data leakage and ensure that the system predicts final performance without using later-period grades.

## Machine Learning Models

Three regression models were trained and evaluated:

1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

### Evaluation Metrics

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

### Model Results

| Model             |    MAE |   RMSE |     R² |
| ----------------- | -----: | -----: | -----: |
| Linear Regression | 2.1564 | 2.8618 | 0.1602 |
| Random Forest     | 2.0648 | 2.8274 | 0.1802 |
| Gradient Boosting | 2.0697 | 2.7881 | 0.2029 |

Gradient Boosting was saved as the selected model based on the evaluation results.

## Dashboard

The interactive Streamlit dashboard provides:

* Dataset overview
* Dataset preview
* Final grade distribution
* Model comparison
* MAE and RMSE visualization
* R² comparison
* Actual vs. predicted grades
* Feature importance
* Interactive student performance prediction

## Project Structure

```text
3rd project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   └── model_training.py
│
├── dashboard/
│   └── app.py
│
├── tests/
│   └── test_pipeline.py
│
├── reports/
│   ├── model_evaluation.csv
│   └── test_predictions.csv
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Plotly
* Streamlit
* Matplotlib
* Seaborn
* Git & GitHub

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Run the Dashboard

```bash
streamlit run dashboard/app.py
```

## Run Tests

```bash
pytest tests/test_pipeline.py
```

The current test suite contains 7 automated tests, all of which pass successfully.

## Data Pipeline

```text
Raw Student Data
       ↓
Data Cleaning
       ↓
Feature Engineering
       ↓
Train/Test Split
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Prediction Results
       ↓
Streamlit Dashboard
```

## Limitations

* The dataset is relatively small.
* The model does not use G1 and G2 to avoid data leakage.
* Model performance may vary on different student populations.
* The system is intended as an analytical and educational tool rather than a definitive assessment of student outcomes.

## Future Improvements

* Use larger and more recent education datasets
* Perform hyperparameter tuning
* Add cross-validation
* Evaluate additional machine learning models
* Add explainable AI techniques
* Expand prediction inputs
* Deploy the dashboard to a cloud platform

## Project Status

Completed as a Week 3 SDC internship project.

The project includes a complete data pipeline, machine learning model evaluation, automated testing, and an interactive Streamlit dashboard.
