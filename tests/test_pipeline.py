import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "student_performance_cleaned.csv"
FEATURE_PATH = BASE_DIR / "data" / "processed" / "student_performance_features.csv"
EVALUATION_PATH = BASE_DIR / "reports" / "model_evaluation.csv"
PREDICTIONS_PATH = BASE_DIR / "reports" / "test_predictions.csv"
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"


def test_processed_data_exists():
    assert DATA_PATH.exists()


def test_feature_data_exists():
    assert FEATURE_PATH.exists()


def test_evaluation_results_exist():
    assert EVALUATION_PATH.exists()


def test_predictions_exist():
    assert PREDICTIONS_PATH.exists()


def test_model_exists():
    assert MODEL_PATH.exists()


def test_processed_data_has_no_missing_values():
    df = pd.read_csv(DATA_PATH)
    assert df.isnull().sum().sum() == 0


def test_prediction_columns():
    df = pd.read_csv(PREDICTIONS_PATH)
    assert "Actual_G3" in df.columns
    assert "Predicted_G3" in df.columns