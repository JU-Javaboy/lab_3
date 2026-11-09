# Lab 3 reflections

## Part A: Code tests versus ML-specific tests

Software smoke tests can confirm that a script starts, but they may not inspect the data or model quality. For example, a smoke test could miss a missing feature value or duplicate `sample_id` in the training CSV. It could also miss a weak majority-class model that runs successfully but has poor recall and F1-score for malignant cases. ML-specific tests are therefore needed in addition to ordinary software tests.

## Part B: Faulty dataset findings

`train_faulty.csv` contains missing values, a nonnumeric value in `mean perimeter`, negative values in `mean radius` and `mean area`, a duplicate `sample_id`, and a label outside the permitted values 0 and 1. The validation function reports these problems instead of silently removing or repairing rows. Imputing a missing value is not the same as validating data: imputation changes the data, while validation checks whether the data is acceptable. Invalid data should stop training because it can produce misleading metrics and an unreliable model.

## Part C: Model-quality findings

The candidate model achieved recall 0.849, F1-score 0.874, and Brier score 0.057, so it passed the defined gate. Recall measures how many malignant cases were detected, F1 balances precision and recall, and the Brier score evaluates predicted probabilities, with lower values being better. The weak majority-class model can run and may appear accurate because one class is more common, but it does not meet the positive-class requirements. One metric is not enough: a model can have good accuracy but poor recall or poorly calibrated probabilities.

## Part D: Drift and subgroup performance

The shifted evaluation data increased `mean radius`, and the simple mean-difference rule detected drift. This is an alert, not proof that model accuracy decreased or that the relationship between features and labels changed. Subgroup recall was 0.667 for the small-radius group and 0.860 for the large-radius group. However, the small-radius group contained only three positive cases, so its recall estimate is very uncertain.

## Part E: CI checks and failure demonstration

The GitHub Actions workflow runs software tests, data tests, model tests, and `run_checks.py`. This makes validation repeatable whenever the project is checked by CI. A deliberately failing version that loads `weak.joblib` would fail the model quality gate, demonstrating that CI can prevent an inadequate model from being accepted. The workflow currently targets the `main` branch, so the repository branch configuration should remain consistent with that trigger.

## Conclusion (150–250 words)

This lab showed that an ML system can fail even when its Python code runs correctly. The faulty training file contained missing and nonnumeric values, negative measurements, a duplicate identifier, and an invalid label. These issues should stop the pipeline rather than being silently ignored, because training on corrupted data can produce misleading results. The weak majority-class model was another important failure: it executed successfully but did not satisfy the required positive-class performance criteria. The candidate model passed the classroom gate with recall 0.849, F1 0.874, and Brier 0.057, but these results do not prove clinical reliability. The shifted evaluation file triggered the simple drift rule, although measured feature drift does not necessarily mean concept drift or reduced accuracy. Automated CI helps by running the same data, model, and software checks consistently for every change. Before trusting an ML system in production, I would need independent external validation, representative and sufficiently large data, confidence intervals, monitoring for performance and drift, calibration analysis, security and privacy review, reproducible training, and human oversight. This project is educational and must not be used for medical decisions.
