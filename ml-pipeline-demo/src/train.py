import csv
import os
import sys
import random
import pickle
import json
import yaml
import mlflow

from evaluate import evaluate


def load_config(path="configs/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_data(path):
    """Load CSV data and return features and labels."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def simple_train_test_split(rows, test_ratio=0.2, seed=42):
    """Split data into train and test sets."""
    rows = rows.copy()
    random.seed(seed)
    random.shuffle(rows)
    split_idx = int(len(rows) * (1 - test_ratio))
    return rows[:split_idx], rows[split_idx:]


def train_simple_model(train_rows, charge_threshold=80.0):
    """
    Train a simple rule-based model.
    """
    churn_rates = {}
    for contract_type in ["month-to-month", "one-year", "two-year"]:
        matching = [r for r in train_rows if r["contract_type"] == contract_type]
        if matching:
            churned = sum(1 for r in matching if r["churned"] == "1")
            churn_rates[contract_type] = churned / len(matching)
        else:
            churn_rates[contract_type] = 0.5

    churned_tenure = [int(r["tenure_months"]) for r in train_rows if r["churned"] == "1"]
    avg_churned_tenure = sum(churned_tenure) / len(churned_tenure) if churned_tenure else 12

    model = {
        "churn_rates_by_contract": churn_rates,
        "tenure_threshold": avg_churned_tenure,
        "charge_threshold": charge_threshold,
    }
    return model


def predict(model, row, score_threshold=0.4):
    """Predict churn for a single row."""
    score = model["churn_rates_by_contract"].get(row["contract_type"], 0.5)
    if int(row["tenure_months"]) < model["tenure_threshold"]:
        score += 0.1
    if float(row["monthly_charges"]) > model["charge_threshold"]:
        score += 0.1
    return 1 if score > score_threshold else 0


if __name__ == "__main__":
    config = load_config()

    data_path = config["data"]["raw_data_path"]
    test_ratio = config["training"]["test_ratio"]
    random_seed = config["training"]["random_seed"]
    charge_threshold = config["model"]["charge_threshold"]
    score_threshold = config["model"]["score_threshold"]
    min_accuracy = config["thresholds"]["min_accuracy"]
    experiment_name = config["mlflow"]["experiment_name"]

    print(f"Loading data from {data_path}...")
    rows = load_data(data_path)
    print(f"Loaded {len(rows)} rows")

    train_rows, test_rows = simple_train_test_split(
        rows, test_ratio=test_ratio, seed=random_seed
    )
    print(f"Train: {len(train_rows)} rows, Test: {len(test_rows)} rows")

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("test_ratio", test_ratio)
        mlflow.log_param("random_seed", random_seed)
        mlflow.log_param("charge_threshold", charge_threshold)
        mlflow.log_param("score_threshold", score_threshold)

        print("Training model...")
        model = train_simple_model(train_rows, charge_threshold=charge_threshold)

        print("Evaluating...")
        metrics = evaluate(
            model,
            test_rows,
            lambda m, row: predict(m, row, score_threshold=score_threshold),
        )
        print(f"Accuracy:  {metrics['accuracy']}")
        print(f"Precision: {metrics['precision']}")
        print(f"Recall:    {metrics['recall']}")

        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall", metrics["recall"])

        os.makedirs("models", exist_ok=True)
        model_path = "models/churn_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print("Model saved to models/churn_model.pkl")

        os.makedirs("metrics", exist_ok=True)
        metrics_path = "metrics/results.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print("Metrics saved to metrics/results.json")

        mlflow.log_artifact(model_path)
        mlflow.log_artifact(metrics_path)
        mlflow.log_artifact("configs/config.yaml")

        if metrics["accuracy"] < min_accuracy:
            print(
                f"ERROR: Accuracy {metrics['accuracy']} is below minimum threshold {min_accuracy}"
            )
            sys.exit(1)
