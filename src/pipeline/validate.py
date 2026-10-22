
import pandas as pd


class DataValidationError(Exception):
    pass


EXPECTED_COLUMNS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "species",
    "sepal_area",
    "petal_area",
    "sepal_to_petal_length_ratio",
    "petal_length_bin",
]

VALID_SPECIES = {"setosa", "versicolor", "virginica"}


def validate(path):
    df = pd.read_csv(path)
    errors = []

    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        errors.append(f"Missing expected columns: {sorted(missing)}")

    if errors:
        raise DataValidationError("; ".join(errors))

    if not df["species"].isin(VALID_SPECIES).all():
        errors.append("Unexpected species values")

    numeric_ranges = {
        "sepal length (cm)": (3.0, 9.0),
        "sepal width (cm)": (1.0, 6.0),
        "petal length (cm)": (0.0, 8.0),
        "petal width (cm)": (0.0, 3.0),
    }

    for column, (low, high) in numeric_ranges.items():
        values = df[column]
        if not values.between(low, high).all():
            errors.append(f"{column} out of expected range")

    if errors:
        raise DataValidationError("; ".join(errors))

    return df
