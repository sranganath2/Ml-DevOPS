import pandas as pd
import os

DATA_PATH = "data/raw/customers.csv"

def load_dataset():
    assert os.path.exists(DATA_PATH), f"Dataset not found: {DATA_PATH}"
    return pd.read_csv(DATA_PATH)

def test_dataset_exists():
    assert os.path.exists(DATA_PATH)

def test_expected_columns_present():
    df = load_dataset()
    expected = {"customer_id", "tenure_months", "monthly_charges", "contract_type", "churned"}
    missing = expected - set(df.columns)
    assert not missing, f"Missing columns: {missing}"

def test_target_values_are_binary_strings():
    df = load_dataset()
    actual = set(df["churned"].dropna().unique())
    assert actual.issubset({"0", "1", 0, 1}), f"Unexpected churned values: {actual}"

def test_tenure_is_non_negative():
    df = load_dataset()
    assert (pd.to_numeric(df["tenure_months"], errors="coerce").dropna() >= 0).all()

def test_monthly_charges_is_non_negative():
    df = load_dataset()
    assert (pd.to_numeric(df["monthly_charges"], errors="coerce").dropna() >= 0).all()
