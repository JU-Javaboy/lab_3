"""Model checks: M1-M5 live in model_checks.py."""
import joblib
import pandas as pd
from ml_config import ROOT
from model_checks import evaluate_model, performance_gate, drift_detected, subgroup_recalls


def load_data():
    return pd.read_csv(ROOT / 'data/evaluation.csv')


def test_candidate_meets_quality_gate():
    candidate = joblib.load(ROOT / 'models/candidate.joblib')
    metrics = evaluate_model(candidate, load_data())
    print("metrics for candidate model:", metrics)
    assert {'f1','recall','brier'} <= metrics.keys(), metrics
    assert performance_gate(metrics), metrics


def test_weak_model_is_rejected():
    weak = joblib.load(ROOT / 'models/weak.joblib')
    metrics = evaluate_model(weak, load_data())
    print("metrics for weak model:", metrics)
    assert not performance_gate(metrics), metrics


def test_shifted_dataset_triggers_drift_warning():
    evaluation = load_data()
    shifted = pd.read_csv(ROOT / 'data/evaluation_shifted.csv')
    assert not drift_detected(evaluation, evaluation)
    assert drift_detected(evaluation, shifted)


def test_subgroup_recall_is_computed():
    model = joblib.load(ROOT / 'models/candidate.joblib')
    scores = subgroup_recalls(model, load_data())
    assert set(scores) == {'small_radius', 'large_radius'}
    assert all(0 <= value <= 1 for value in scores.values())
