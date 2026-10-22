 
import pandas as pd

NUMERIC_COLS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]


def preprocess(input_path, output_path):
    df = pd.read_csv(input_path)

    required_cols = NUMERIC_COLS + ["species"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Remove rows with missing target labels
    df = df.dropna(subset=["species"])

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Convert numeric columns to numeric values
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill missing numeric values with the median
    for col in NUMERIC_COLS:
        if not df.empty and df[col].isnull().any():
            median_value = df[col].median()
            if pd.notna(median_value):
                df[col] = df[col].fillna(median_value)

    # Save the processed data
    df.to_csv(output_path, index=False)

    return df
