"""Data checks: initially red until you implement D1-D5 in validation.py."""
import pandas as pd
from ml_config import ROOT
from validation import validate_data, assert_valid_data


def test_clean_data_passes():
    assert validate_data(pd.read_csv(ROOT / 'data/train.csv')) == []


def test_missing_required_column_is_detected():
    clean = pd.read_csv(ROOT / 'data/train.csv').drop(columns=['mean radius'])
    assert any('column' in item.lower() for item in validate_data(clean))


def test_faulty_dataset_is_rejected():
    faulty = pd.read_csv(ROOT / 'data/train_faulty.csv')
    errors = validate_data(faulty)
    # TODO D6: inspect the printed messages; an accepted faulty file is a bug.
    assert len(errors) >= 4, f'Expected multiple problems, got {errors}'


def test_missing_value_is_rejected():
    df = pd.read_csv(ROOT / 'data/train.csv')
    df.loc[0, 'mean radius'] = float('nan')
    assert validate_data(df)


def test_negative_measurement_is_rejected():
    df = pd.read_csv(ROOT / 'data/train.csv')
    df.loc[0, 'mean radius'] = -1
    assert validate_data(df)


def test_duplicate_id_is_rejected():
    df = pd.read_csv(ROOT / 'data/train.csv')
    df.loc[0, 'sample_id'] = df.loc[1, 'sample_id']
    assert validate_data(df)


def test_invalid_label_is_rejected():
    df = pd.read_csv(ROOT / 'data/train.csv')
    df.loc[0, 'malignant'] = 2
    assert validate_data(df)
