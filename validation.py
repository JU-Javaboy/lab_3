"""Data-quality checks. Complete the marked TODOs (do not silently clean data)."""
import pandas as pd
from ml_config import FEATURES, REQUIRED, TARGET


def validate_data(df: pd.DataFrame) -> list[str]:
    """Return descriptions of problems; return [] for a valid dataset.

    We return errors instead of raising immediately so a CI log can explain
    several problems at once. A failed check should stop model training.
    """
    problems = []
    # WORKED EXAMPLE: required columns are present.
    missing_columns = sorted(set(REQUIRED) - set(df.columns))
    if missing_columns:
        problems.append(f"Missing columns: {missing_columns}")
        return problems  # Cannot safely inspect absent columns.

    # TODO D1: reject missing values in FEATURES and TARGET.
    # Hint: df[FEATURES + [TARGET]].isna().any().any()
    df_subset = df[FEATURES + [TARGET]]
    if df_subset.isna().any().any():
        problems.append("Missing values found in FEATURES or TARGET")

    # TODO D2: reject nonnumeric feature values, including strings such as
    # 'not-a-number'. Use pd.to_numeric(..., errors='coerce') in a loop.
    # Important: a nonnumeric string must not crash this function.
   
    # D2: nonnumeric values (coerce, don't crash)
    numeric = {}
    for col in FEATURES:
        converted = pd.Series(pd.to_numeric(df[col], errors="coerce"))
        bad = converted.isna() & df[col].notna()
        if bad.any():
            problems.append(f"Nonnumeric values in '{col}'")
        numeric[col] = converted

    # TODO D3: reject negative feature values. All selected measurements are
    # nonnegative. The numeric conversion from D2 may help here.

    # D3: negative values
    for col, values in numeric.items():
        if (values < 0).any():
            problems.append(f"Negative values in '{col}'")

    # TODO D4: sample_id must be unique. Hint: df['sample_id'].duplicated().any()

    # D4: unique IDs
    if df["sample_id"].duplicated().any():
        problems.append("Duplicate sample_id values")

    # TODO D5: TARGET must contain only 0 and 1 (and no missing values).
    # Hint: df[TARGET].isin([0, 1]).all()

    # D5: labels only 0/1
    if not df[TARGET].isin([0, 1]).all():
        problems.append(f"{TARGET} must contain only 0 and 1")

    return problems


def assert_valid_data(df: pd.DataFrame) -> None:
    """Raise a clear failure for CI, rather than training on invalid data."""
    problems = validate_data(df)
    if problems:
        raise ValueError("Data validation failed: " + "; ".join(problems))
