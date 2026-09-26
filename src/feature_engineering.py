import pandas as pd
from pathlib import Path


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "data" / "processed" / "student_performance_cleaned.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "student_performance_features.csv"


def load_data():
    """Load the cleaned dataset."""
    return pd.read_csv(INPUT_PATH)


def create_features(df):
    """Create model-ready features and target."""

    # Separate target variable
    y = df["G3"]

    # Remove target and G1/G2 to avoid using later-period grades
    # for early student performance prediction.
    X = df.drop(columns=["G3", "G1", "G2"])

    # Convert categorical variables into numerical dummy variables
    X = pd.get_dummies(X, drop_first=True)

    return X, y


def save_features(X, y):
    """Save model-ready features and target."""

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    features = X.copy()
    features["G3"] = y.values

    features.to_csv(OUTPUT_PATH, index=False)

    print(f"Feature dataset saved to: {OUTPUT_PATH}")


def main():
    print("Loading cleaned dataset...")

    df = load_data()

    print(f"Original shape: {df.shape}")

    X, y = create_features(df)

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    save_features(X, y)

    print("Feature engineering completed successfully.")


if __name__ == "__main__":
    main()