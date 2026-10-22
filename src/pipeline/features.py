
import pandas as pd


def engineer_features(input_path, output_path):
    """Create additional features for the Iris dataset."""
    df = pd.read_csv(input_path)

    df["sepal_area"] = (
        df["sepal length (cm)"] * df["sepal width (cm)"]
    )

    df["petal_area"] = (
        df["petal length (cm)"] * df["petal width (cm)"]
    )

    df["sepal_to_petal_length_ratio"] = (
        df["sepal length (cm)"]
        / df["petal length (cm)"].replace(0, float("nan"))
    )

    df["petal_length_bin"] = pd.cut(
        df["petal length (cm)"],
        bins=[-float("inf"), 2, 5, float("inf")],
        labels=["short", "medium", "long"],
        right=False,
    )

    df.to_csv(output_path, index=False)
    return df
