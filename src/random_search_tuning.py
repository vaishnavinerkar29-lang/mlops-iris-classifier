import mlflow
import pandas as pd

from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder


FEATURE_COLS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "sepal_area",
    "petal_area",
    "sepal_to_petal_length_ratio",
]

PARAM_DIST = {
    "n_estimators": randint(50, 300),
    "max_depth": [3, 5, 10, 15, None],
    "min_samples_split": randint(2, 15),
    "max_features": ["sqrt", "log2"],
}

N_ITER = 30


def load_dataset(path):
    df = pd.read_csv(path)

    le = LabelEncoder()
    y = le.fit_transform(df["species"])

    X = df[FEATURE_COLS].fillna(
        df[FEATURE_COLS].median()
    )

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )


def run_random_search(data_path):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("iris-hyperparameter-tuning")

    X_train, X_test, y_train, y_test = load_dataset(data_path)

    model = RandomForestClassifier(
        random_state=42
    )

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=PARAM_DIST,
        n_iter=N_ITER,
        cv=5,
        scoring="f1_macro",
        random_state=42,
        n_jobs=-1,
        return_train_score=False,
    )

    with mlflow.start_run(
        run_name="random_search_random_forest"
    ):

        random_search.fit(X_train, y_train)

        best_model = random_search.best_estimator_

        test_accuracy = best_model.score(
            X_test,
            y_test
        )

        total_fits = N_ITER * 5

        mlflow.log_param(
            "model_type",
            "RandomForestClassifier"
        )

        mlflow.log_param(
            "cv_folds",
            5
        )

        mlflow.log_param(
            "scoring",
            "f1_macro"
        )

        mlflow.log_param(
            "search_type",
            "RandomizedSearchCV"
        )

        mlflow.log_param(
            "n_iter",
            N_ITER
        )

        mlflow.log_param(
            "total_fits",
            total_fits
        )

        mlflow.log_metric(
            "best_cv_f1_macro",
            random_search.best_score_
        )

        mlflow.log_metric(
            "test_accuracy",
            test_accuracy
        )

        for param_name, param_value in random_search.best_params_.items():
            mlflow.log_param(
                f"best_{param_name}",
                param_value
            )

        results = pd.DataFrame(
            random_search.cv_results_
        )

        results.to_csv(
            "random_search_all_candidates.csv",
            index=False
        )

        mlflow.log_artifact(
            "random_search_all_candidates.csv"
        )

        print(
            f"Random Search best CV f1_macro: "
            f"{random_search.best_score_:.4f}"
        )

        print(
            f"Random Search test accuracy: "
            f"{test_accuracy:.4f}"
        )

        print("Best parameters:")

        print(
            random_search.best_params_
        )

        print(
            f"Total iterations: {N_ITER}"
        )

        print(
            f"Total fits: {total_fits}"
        )


if __name__ == "__main__":
    run_random_search(
        "data/processed/iris_features.csv"
    )