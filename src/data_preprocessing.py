import pandas as pd
from pathlib import Path


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "student-por.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "student_performance_cleaned.csv"


def load_data():
    """Load the raw Portuguese student performance dataset."""
    df = pd.read_csv(RAW_DATA_PATH, sep=";")
    return df


def clean_data(df):
    """Perform basic data quality checks and cleaning."""

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Check for missing values
    missing_values = df.isnull().sum().sum()

    print(f"Missing values: {missing_values}")
    print(f"Rows after cleaning: {len(df)}")

    return df


def save_data(df):
    """Save the cleaned dataset to the processed folder."""

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"Processed dataset saved to: {PROCESSED_DATA_PATH}")


def main():
    print("Loading raw dataset...")

    df = load_data()

    print(f"Original dataset shape: {df.shape}")

    df = clean_data(df)

    save_data(df)

    print("Data preprocessing completed successfully.")


if __name__ == "__main__":
    main()