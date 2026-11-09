"""Optional cross-validation, calibration, and deployment exercises.

The required core tests are in tests/. You can run this file directly.
"""
import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from ml_config import ROOT, FEATURES, TARGET


def cross_validate():
    train = pd.read_csv(ROOT / 'data/train.csv')
    X, y = train[FEATURES], train[TARGET]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))
    # WORKED EXAMPLE: construct reproducible stratified 3-fold splitting.
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    # TODO E1: use cross_val_score(model, X, y, cv=folds, scoring='f1').
    # Print the three fold scores and their average.
    pass


def calibration():
    evaluation = pd.read_csv(ROOT / 'data/evaluation.csv')
    model = joblib.load(ROOT / 'models/candidate.joblib')
    y = evaluation[TARGET]
    probs = model.predict_proba(evaluation[FEATURES])[:, 1]
    # TODO E2: use calibration_curve(y, probs, n_bins=5) and print
    # fraction_of_positives and mean_predicted_value (2 returned arrays).
    # A calibration plot is optional; compare values near the diagonal.
    pass


if __name__ == '__main__':
    cross_validate()
    calibration()
