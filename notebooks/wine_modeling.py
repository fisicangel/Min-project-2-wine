"""Reusable utilities for the white-wine quality classification project."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]


def load_and_clean_wine_data(csv_path: str | Path) -> pd.DataFrame:
    """Load the semicolon-separated UCI white-wine file and remove duplicates."""
    wine_df = pd.read_csv(csv_path, sep=";")
    return wine_df.drop_duplicates().reset_index(drop=True)


def create_binary_target(wine_df: pd.DataFrame) -> pd.DataFrame:
    """Create class 0 for scores <=5 and class 1 for scores >=6."""
    result = wine_df.copy()
    result["quality_class"] = (result["quality"] >= 6).astype(int)
    return result


def split_features_target(
    wine_df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
):
    """Return a reproducible stratified train-test split."""
    X = wine_df[FEATURE_COLUMNS]
    y = wine_df["quality_class"]
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_final_gradient_boosting() -> GradientBoostingClassifier:
    """Return the best Gradient Boosting configuration found by Grid Search."""
    return GradientBoostingClassifier(
        learning_rate=0.1,
        max_depth=3,
        min_samples_leaf=1,
        n_estimators=200,
        subsample=1.0,
        random_state=42,
    )


def predict_with_threshold(model, X: pd.DataFrame, threshold: float = 0.50):
    """Convert class-1 probabilities into predictions at a chosen threshold."""
    probabilities = model.predict_proba(X)[:, 1]
    return (probabilities >= threshold).astype(int)


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    """Return overall accuracy and metrics focused on lower-quality class 0."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "lower_quality_precision": precision_score(y_true, y_pred, pos_label=0),
        "lower_quality_recall": recall_score(y_true, y_pred, pos_label=0),
        "lower_quality_f1": f1_score(y_true, y_pred, pos_label=0),
    }
