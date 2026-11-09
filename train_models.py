"""Recreate both committed models deterministically; no download needed."""
import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ml_config import ROOT, FEATURES, TARGET
from validation import assert_valid_data


def main():
    df = pd.read_csv(ROOT / 'data/train.csv')
    assert_valid_data(df)
    X, y = df[FEATURES], df[TARGET]
    good = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))
    good.fit(X, y)
    weak = DummyClassifier(strategy='most_frequent').fit(X, y)
    (ROOT / 'models').mkdir(exist_ok=True)
    joblib.dump(good, ROOT / 'models/candidate.joblib')
    joblib.dump(weak, ROOT / 'models/weak.joblib')
    print('Wrote candidate.joblib and weak.joblib')


if __name__ == '__main__':
    main()
