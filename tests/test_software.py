"""Small traditional software tests to contrast with ML quality checks."""
import pandas as pd
from ml_config import ROOT, FEATURES


def test_data_files_exist():
    assert (ROOT / 'data/train.csv').exists()
    assert (ROOT / 'models/candidate.joblib').exists()


def test_evaluation_has_expected_features():
    df = pd.read_csv(ROOT / 'data/evaluation.csv')
    assert all(feature in df for feature in FEATURES)
