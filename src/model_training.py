import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "data" / "processed" / "student_performance_features.csv"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"

RESULTS_PATH = REPORTS_DIR / "model_evaluation.csv"
MODEL_PATH = MODELS_DIR / "best_model.pkl"
PREDICTIONS_PATH = REPORTS_DIR / "test_predictions.csv"


def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    return mae, rmse, r2, y_pred


def main():
    print("Loading feature dataset...")

    df = pd.read_csv(INPUT_PATH)

    X = df.drop(columns=["G3"])
    y = df["G3"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            random_state=42
        )
    }

    results = []
    predictions = {}

    print("\nTraining models...\n")

    for name, model in models.items():

        print(f"Training {name}...")

        mae, rmse, r2, y_pred = evaluate_model(
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        predictions[name] = y_pred

    results_df = pd.DataFrame(results)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save evaluation results
    results_df.to_csv(RESULTS_PATH, index=False)

    # Select Gradient Boosting based on current evaluation
    best_model = models["Gradient Boosting"]
    best_model.fit(X_train, y_train)

    # Save trained model
    joblib.dump(best_model, MODEL_PATH)

    # Save predictions
    prediction_df = pd.DataFrame({
        "Actual_G3": y_test.values,
        "Predicted_G3": predictions["Gradient Boosting"]
    })

    prediction_df.to_csv(PREDICTIONS_PATH, index=False)

    print("\nModel Evaluation:")
    print(results_df.to_string(index=False))

    print(f"\nEvaluation saved: {RESULTS_PATH}")
    print(f"Best model saved: {MODEL_PATH}")
    print(f"Predictions saved: {PREDICTIONS_PATH}")

    print("\nModel training pipeline completed successfully.")


if __name__ == "__main__":
    main()