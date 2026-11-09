"""ML validation with small worked examples and student TODOs."""
import pandas as pd
import numpy as np
from sklearn.metrics import recall_score, f1_score, brier_score_loss
from ml_config import FEATURES, TARGET


def evaluate_model(model, df: pd.DataFrame) -> dict[str, float]:
    X = df[FEATURES]
    y = df[TARGET]
    predictions = model.predict(X)
    # WORKED EXAMPLE: recall for malignant cases (positive label = 1).
    results = {"recall": float(recall_score(y, predictions, zero_division=0))}
    results["f1"] = float(f1_score(y, predictions, zero_division=0))
    # Hint: model.predict_proba(X)[:, 1] is the probability of malignant=1
    # for both supplied models (classes_ are [0, 1]). Lower Brier is better.
    results["brier"] = float(brier_score_loss(y, model.predict_proba(X)[:, 1]))
    return results


def performance_gate(metrics: dict[str, float]) -> bool:
    """Accept only a model that meets all three project requirements."""
    return (metrics['f1'] >= 0.85 and metrics['recall'] >= 0.80 and metrics['brier'] <= 0.15)


def drift_detected(reference: pd.DataFrame, current: pd.DataFrame) -> bool:
    """Simple educational drift warning, not a statistical proof of drift."""
    return abs(reference['mean radius'].mean() - current['mean radius'].mean()) > 2.0
    


def subgroup_recalls(model, df: pd.DataFrame) -> dict[str, float]:
    """Recall for smaller/larger mean-radius observations in THIS data.

    This is descriptive slice analysis, not a clinical subgroup comparison.
    """
    median = df["mean radius"].median()
    masks = {"small_radius": df["mean radius"] < median,
             "large_radius": df["mean radius"] >= median}
    result = {}
    for label, mask in masks.items():
        subset = df.loc[mask]
        result[label] = float(recall_score(subset[TARGET], model.predict(subset[FEATURES]), zero_division=0))
    return result
