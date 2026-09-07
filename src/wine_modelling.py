"""Reusable modelling utilities for the white-wine quality project.

The binary target is:
    0 = lower quality (original quality scores 3, 4, or 5)
    1 = acceptable/higher quality (original quality scores 6, 7, 8, or 9)

All learned preprocessing is fitted on training data only.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
TEST_SIZE = 0.20
QUALITY_THRESHOLD = 6
CLASS_NAMES = ["Lower quality (0)", "Acceptable/higher quality (1)"]


def load_and_clean_data(csv_path: str | Path) -> pd.DataFrame:
    """Load the semicolon-separated UCI file and remove duplicate rows."""
    wine_df = pd.read_csv(csv_path, sep=";")

    if "quality" not in wine_df.columns:
        raise ValueError("The dataset must contain a 'quality' column.")

    wine_df = wine_df.drop_duplicates().reset_index(drop=True)
    wine_df["quality_class"] = np.where(
        wine_df["quality"] >= QUALITY_THRESHOLD,
        1,
        0,
    )
    return wine_df


def create_features_and_target(
    wine_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the 11 predictors from the engineered binary target."""
    required_columns = {"quality", "quality_class"}
    missing = required_columns.difference(wine_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    X = wine_df.drop(columns=["quality", "quality_class"])
    y = wine_df["quality_class"].astype(int)
    return X, y


def make_stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create the untouched 80/20 test split while preserving class balance."""
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def make_cross_validator() -> StratifiedKFold:
    """Return the five-fold cross-validator used throughout the project."""
    return StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, Any]:
    """Return overall and lower-quality-specific classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "lower_quality_precision": precision_score(
            y_true, y_pred, pos_label=0, zero_division=0
        ),
        "lower_quality_recall": recall_score(
            y_true, y_pred, pos_label=0, zero_division=0
        ),
        "lower_quality_f1": f1_score(
            y_true, y_pred, pos_label=0, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        ),
    }


def fit_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> DummyClassifier:
    """Fit the majority-class reference model."""
    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)
    return model


def fit_unscaled_knn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_neighbors: int = 10,
) -> KNeighborsClassifier:
    """Fit the initial KNN model without scaling."""
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model


def fit_normalized_knn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_neighbors: int = 5,
) -> Pipeline:
    """Fit MinMaxScaler and KNN together without leaking test information."""
    model = Pipeline(
        steps=[
            ("normalizer", MinMaxScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=n_neighbors)),
        ]
    )
    model.fit(X_train, y_train)
    return model


def tune_knn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold | None = None,
) -> GridSearchCV:
    """Tune scaled KNN across the 116 combinations used in the notebook."""
    pipeline = Pipeline(
        steps=[
            ("normalizer", MinMaxScaler()),
            ("knn", KNeighborsClassifier()),
        ]
    )
    parameter_grid = {
        "knn__n_neighbors": list(range(3, 32)),
        "knn__weights": ["uniform", "distance"],
        "knn__p": [1, 2],
    }
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="accuracy",
        cv=cv or make_cross_validator(),
        n_jobs=-1,
        return_train_score=True,
    )
    search.fit(X_train, y_train)
    return search


def fit_initial_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:
    """Fit the initial balanced Random Forest model."""
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold | None = None,
) -> GridSearchCV:
    """Tune Random Forest across the 72 combinations used in the notebook."""
    parameter_grid = {
        "n_estimators": [150, 250],
        "max_depth": [8, 12, 16],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }
    search = GridSearchCV(
        estimator=RandomForestClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        ),
        param_grid=parameter_grid,
        scoring="accuracy",
        cv=cv or make_cross_validator(),
        n_jobs=-1,
        return_train_score=True,
    )
    search.fit(X_train, y_train)
    return search


def fit_initial_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> GradientBoostingClassifier:
    """Fit the initial Gradient Boosting model."""
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def tune_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold | None = None,
) -> GridSearchCV:
    """Tune Gradient Boosting across the 72 combinations used in the notebook."""
    parameter_grid = {
        "n_estimators": [100, 150, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [1, 2, 3],
        "min_samples_leaf": [1, 3],
        "subsample": [0.8, 1.0],
    }
    search = GridSearchCV(
        estimator=GradientBoostingClassifier(random_state=RANDOM_STATE),
        param_grid=parameter_grid,
        scoring="accuracy",
        cv=cv or make_cross_validator(),
        n_jobs=-1,
        return_train_score=True,
    )
    search.fit(X_train, y_train)
    return search


def select_probability_threshold(
    model: GradientBoostingClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold | None = None,
) -> tuple[float, pd.DataFrame]:
    """Choose a threshold from out-of-fold training probabilities only."""
    probabilities = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv or make_cross_validator(),
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    rows = []
    for threshold in np.arange(0.40, 0.71, 0.01):
        predictions = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "balanced_accuracy": balanced_accuracy_score(
                    y_train, predictions
                ),
                "lower_quality_precision": precision_score(
                    y_train,
                    predictions,
                    pos_label=0,
                    zero_division=0,
                ),
                "lower_quality_recall": recall_score(
                    y_train,
                    predictions,
                    pos_label=0,
                    zero_division=0,
                ),
                "lower_quality_f1": f1_score(
                    y_train,
                    predictions,
                    pos_label=0,
                    zero_division=0,
                ),
            }
        )

    results = pd.DataFrame(rows)
    best_row = results.loc[results["balanced_accuracy"].idxmax()]
    return float(best_row["threshold"]), results


def predict_with_threshold(
    model: GradientBoostingClassifier,
    X: pd.DataFrame,
    threshold: float,
) -> np.ndarray:
    """Convert class-1 probabilities into labels using a chosen threshold."""
    probabilities = model.predict_proba(X)[:, 1]
    return (probabilities >= threshold).astype(int)


def bootstrap_confidence_intervals(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    n_bootstrap: int = 2000,
) -> dict[str, tuple[float, float]]:
    """Estimate 95% intervals for accuracy and lower-quality recall."""
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    generator = np.random.default_rng(RANDOM_STATE)
    accuracies: list[float] = []
    recalls: list[float] = []

    for _ in range(n_bootstrap):
        indices = generator.choice(
            len(y_true_array),
            size=len(y_true_array),
            replace=True,
        )
        sampled_true = y_true_array[indices]
        sampled_pred = y_pred_array[indices]
        accuracies.append(accuracy_score(sampled_true, sampled_pred))
        if 0 in sampled_true:
            recalls.append(
                recall_score(sampled_true, sampled_pred, pos_label=0)
            )

    return {
        "accuracy_95_ci": tuple(np.percentile(accuracies, [2.5, 97.5])),
        "lower_quality_recall_95_ci": tuple(
            np.percentile(recalls, [2.5, 97.5])
        ),
    }


def fit_surrogate_model(
    final_model: GradientBoostingClassifier,
    X_train: pd.DataFrame,
) -> DecisionTreeClassifier:
    """Fit a small tree that approximates the final model's predictions."""
    surrogate = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=30,
        random_state=RANDOM_STATE,
    )
    surrogate.fit(X_train, final_model.predict(X_train))
    return surrogate


def run_complete_modelling(csv_path: str | Path) -> dict[str, Any]:
    """Run the full project modelling workflow and return fitted results."""
    wine_df = load_and_clean_data(csv_path)
    X, y = create_features_and_target(wine_df)
    X_train, X_test, y_train, y_test = make_stratified_split(X, y)
    cv = make_cross_validator()

    baseline = fit_baseline(X_train, y_train)
    unscaled_knn = fit_unscaled_knn(X_train, y_train)
    normalized_knn = fit_normalized_knn(X_train, y_train)
    tuned_knn_search = tune_knn(X_train, y_train, cv)

    initial_rf = fit_initial_random_forest(X_train, y_train)
    initial_rf_cv = cross_val_score(
        initial_rf, X_train, y_train, scoring="accuracy", cv=cv, n_jobs=-1
    )
    tuned_rf_search = tune_random_forest(X_train, y_train, cv)

    initial_gb = fit_initial_gradient_boosting(X_train, y_train)
    initial_gb_cv = cross_val_score(
        initial_gb, X_train, y_train, scoring="accuracy", cv=cv, n_jobs=-1
    )
    tuned_gb_search = tune_gradient_boosting(X_train, y_train, cv)
    final_model = tuned_gb_search.best_estimator_

    default_predictions = final_model.predict(X_test)
    threshold, threshold_table = select_probability_threshold(
        final_model, X_train, y_train, cv
    )
    threshold_predictions = predict_with_threshold(
        final_model, X_test, threshold
    )
    intervals = bootstrap_confidence_intervals(
        y_test, default_predictions
    )
    surrogate = fit_surrogate_model(final_model, X_train)

    return {
        "wine_df": wine_df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "baseline": baseline,
        "unscaled_knn": unscaled_knn,
        "normalized_knn": normalized_knn,
        "tuned_knn_search": tuned_knn_search,
        "initial_random_forest": initial_rf,
        "initial_random_forest_cv_scores": initial_rf_cv,
        "tuned_random_forest_search": tuned_rf_search,
        "initial_gradient_boosting": initial_gb,
        "initial_gradient_boosting_cv_scores": initial_gb_cv,
        "tuned_gradient_boosting_search": tuned_gb_search,
        "final_model": final_model,
        "default_test_metrics": classification_metrics(
            y_test, default_predictions
        ),
        "selected_threshold": threshold,
        "threshold_results": threshold_table,
        "threshold_test_metrics": classification_metrics(
            y_test, threshold_predictions
        ),
        "confidence_intervals": intervals,
        "surrogate_model": surrogate,
        "surrogate_test_fidelity": accuracy_score(
            final_model.predict(X_test), surrogate.predict(X_test)
        ),
    }


def print_summary(results: dict[str, Any]) -> None:
    """Print the central results from the completed modelling workflow."""
    final_search = results["tuned_gradient_boosting_search"]
    metrics = results["default_test_metrics"]
    threshold_metrics = results["threshold_test_metrics"]
    accuracy_interval = results["confidence_intervals"]["accuracy_95_ci"]
    recall_interval = results["confidence_intervals"][
        "lower_quality_recall_95_ci"
    ]

    print("Best Gradient Boosting parameters:", final_search.best_params_)
    print(f"Best cross-validation accuracy: {final_search.best_score_:.2%}")
    print(f"Final test accuracy: {metrics['accuracy']:.2%}")
    print(
        "Lower-quality precision / recall / F1: "
        f"{metrics['lower_quality_precision']:.2%} / "
        f"{metrics['lower_quality_recall']:.2%} / "
        f"{metrics['lower_quality_f1']:.2%}"
    )
    print(
        f"Selected quality-control threshold: "
        f"{results['selected_threshold']:.2f}"
    )
    print(
        "Threshold-adjusted accuracy / lower-quality recall: "
        f"{threshold_metrics['accuracy']:.2%} / "
        f"{threshold_metrics['lower_quality_recall']:.2%}"
    )
    print(
        "95% accuracy confidence interval: "
        f"{accuracy_interval[0]:.2%} to {accuracy_interval[1]:.2%}"
    )
    print(
        "95% lower-quality recall confidence interval: "
        f"{recall_interval[0]:.2%} to {recall_interval[1]:.2%}"
    )
    print(
        f"Surrogate test fidelity: "
        f"{results['surrogate_test_fidelity']:.2%}"
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the white-wine quality classification workflow."
    )
    parser.add_argument(
        "data_path",
        type=Path,
        help="Path to winequality-white.csv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    project_results = run_complete_modelling(arguments.data_path)
    print_summary(project_results)
