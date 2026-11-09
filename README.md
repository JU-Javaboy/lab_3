# Lab 3 — CI/CD and ML Testing (student starter)

**Optional, individual lab.** This is an educational ML exercise, not a clinical system.

## Setup

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest tests/test_software.py -v
python -m pytest -v  # Some tests SHOULD FAIL until you complete TODOs.
```

## Your files

- `data/train.csv`, `data/evaluation.csv`: clean educational datasets from Wisconsin Diagnostic Breast Cancer data; **never tune repeatedly on the evaluation file** in real research.
- `data/train_faulty.csv`: intentionally corrupted **copy** of training data: negative values, a missing value, a nonnumeric string, a duplicate ID, and an invalid label.
- `data/evaluation_shifted.csv`: constructed copy with increased mean radius to illustrate a drift alert; it is NOT a naturally observed population shift.
- `models/candidate.joblib`: pretrained logistic regression classifier.
- `models/weak.joblib`: deliberately weak majority-class classifier.
- `validation.py`: data validation TODOs D1-D5 (D6 is an inspection task).
- `model_checks.py`: model evaluation, quality gate, drift, and slice TODOs M1-M5.
- `.github/workflows/ml-ci.yml`: complete GitHub Actions TODOs C1-C4.
- `optional_extensions.py`: cross-validation and calibration TODOs E1-E2.
- `serve_model.py`, `Dockerfile`: optional deployment TODOs E3-E4.
- `train_models.py`: regenerate models after D1-D5; the pre-trained versions are already included.

## Useful commands

```bash
python -m pytest tests/test_data.py -v
python -m pytest tests/test_model.py -v
python run_checks.py
python train_models.py   # After data validation works; regenerates models
python optional_extensions.py
```

## Model labels and source

Original scikit-learn target: 0=malignant, 1=benign. **In this teaching project we deliberately remap to 1=malignant, 0=benign**, to simplify recall of the positive class. Real Wisconsin Diagnostic Breast Cancer observations, 569 samples and 30 source features, from the UCI dataset bundled with scikit-learn. We select five numeric measurements to simplify teaching. No synthetic patients have been created. The intentionally corrupt and shifted files are synthetic **modifications of real dataset rows**.

Sources: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html and https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic

**Do not use these models for medical decisions.** Metrics and thresholds are classroom demonstrations, not clinical approval criteria.

## Reference material

- pandas missing values: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.isna.html
- pandas to_numeric: https://pandas.pydata.org/docs/reference/api/pandas.to_numeric.html
- scikit-learn classification metrics: https://scikit-learn.org/stable/modules/model_evaluation.html
- sklearn cross-validation: https://scikit-learn.org/stable/modules/cross_validation.html
- sklearn calibration: https://scikit-learn.org/stable/modules/calibration.html
- GitHub Actions Python: https://docs.github.com/en/actions/tutorials/build-and-test-code/python
