import os
import json
import sys
sys.path.insert(0, "src")

from train import load_data, simple_train_test_split, train_simple_model, predict
from evaluate import evaluate

DATA_PATH = "data/raw/customers.csv"

def test_load_data_returns_rows():
    rows = load_data(DATA_PATH)
    assert isinstance(rows, list)
    assert len(rows) > 0
    assert isinstance(rows[0], dict)

def test_train_test_split_preserves_all_rows():
    rows = load_data(DATA_PATH)
    train_rows, test_rows = simple_train_test_split(rows, test_ratio=0.2, seed=42)
    assert len(train_rows) + len(test_rows) == len(rows)

def test_model_can_be_trained():
    rows = load_data(DATA_PATH)
    train_rows, _ = simple_train_test_split(rows, test_ratio=0.2, seed=42)
    model = train_simple_model(train_rows)
    assert isinstance(model, dict)
    assert "churn_rates_by_contract" in model
    assert "tenure_threshold" in model

def test_prediction_is_binary():
    rows = load_data(DATA_PATH)
    train_rows, test_rows = simple_train_test_split(rows, test_ratio=0.2, seed=42)
    model = train_simple_model(train_rows)
    pred = predict(model, test_rows[0])
    assert pred in [0, 1]

def test_evaluate_returns_expected_metrics():
    rows = load_data(DATA_PATH)
    train_rows, test_rows = simple_train_test_split(rows, test_ratio=0.2, seed=42)
    model = train_simple_model(train_rows)
    metrics = evaluate(model, test_rows, predict)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "test_size" in metrics

def test_accuracy_meets_minimum_threshold():
    rows = load_data(DATA_PATH)
    train_rows, test_rows = simple_train_test_split(rows, test_ratio=0.2, seed=42)
    model = train_simple_model(train_rows)
    metrics = evaluate(model, test_rows, predict)
    assert metrics["accuracy"] >= 0.60, f"Accuracy too low: {metrics['accuracy']}"
