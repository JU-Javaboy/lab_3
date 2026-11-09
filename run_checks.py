"""Small CI entrypoint; nonzero exit code blocks the pipeline."""
import sys
import joblib
import pandas as pd
from ml_config import ROOT
from validation import assert_valid_data
from model_checks import evaluate_model, performance_gate, drift_detected, subgroup_recalls


def main():
    train = pd.read_csv(ROOT / 'data/train.csv')
    evaluation = pd.read_csv(ROOT / 'data/evaluation.csv')
    for name, df in [('train', train), ('evaluation', evaluation)]:
        assert_valid_data(df)
        print(f'Validated {name}: {len(df)} rows')
    model = joblib.load(ROOT / 'models/candidate.joblib')  # Load only trusted files.
    scores = evaluate_model(model, evaluation)
    print('Evaluation metrics:', scores)
    if not performance_gate(scores):
        raise ValueError('Model quality gate failed')
    print('Slice recall:', subgroup_recalls(model, evaluation))
    shifted = pd.read_csv(ROOT / 'data/evaluation_shifted.csv')
    print('Shifted data drift warning:', drift_detected(evaluation, shifted))
    print('All required gates passed')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, NotImplementedError) as exc:
        print(f'Validation failure: {exc}', file=sys.stderr)
        sys.exit(1)
