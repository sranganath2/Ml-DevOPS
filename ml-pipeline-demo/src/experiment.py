import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import sys
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

config = {
    "model_type": "logistic_regression",
    "test_size": 0.2,
    "random_state": 42,
    "handle_missing": "median",
    "scale_features": True,
    "features_to_drop": [],
    "lr_C": 1.0,                           # logistic regression regularization
    "rf_n_estimators": 100,                # random forest number of trees
    "rf_max_depth": None,                  # random forest max depth (None = unlimited)
    "gb_n_estimators": 100,                # gradient boosting number of trees
    "gb_learning_rate": 0.1,               # gradient boosting learning rate
    "gb_max_depth": 3,                     # gradient boosting max depth
}

def load_and_prepare_data(config):
    url = "https://raw.githubusercontent.com/TripleTen-DS/Dataset/refs/heads/main/student_dropout_dataset.csv"
    print(f"Loading data from URL...")
    df = pd.read_csv(url)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    df = df.drop(columns=["Student_ID"])
    if config["features_to_drop"]:
        df = df.drop(columns=config["features_to_drop"], errors="ignore")
        print(f"Dropped features: {config['features_to_drop']}")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if config["handle_missing"] == "median":
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        print("Filled missing values with median")
    elif config["handle_missing"] == "drop":
        before = len(df)
        df = df.dropna()
        print(f"Dropped rows with missing values: {before} -> {len(df)}")
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    X = df.drop(columns=["Dropout"])
    y = df["Dropout"]
    return X, y, len(df), numeric_cols, categorical_cols

def build_model(config):
    if config["model_type"] == "logistic_regression":
        return LogisticRegression(
            C=config["lr_C"],
            random_state=config["random_state"],
            max_iter=1000
        )
    elif config["model_type"] == "random_forest":
        return RandomForestClassifier(
            n_estimators=config["rf_n_estimators"],
            max_depth=config["rf_max_depth"],
            random_state=config["random_state"]
        )
    elif config["model_type"] == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=config["gb_n_estimators"],
            learning_rate=config["gb_learning_rate"],
            max_depth=config["gb_max_depth"],
            random_state=config["random_state"]
        )
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")

def run_experiment(config):
    mlflow.set_experiment("student-dropout-prediction")
    with mlflow.start_run():
        mlflow.log_param("model_type", config["model_type"])
        mlflow.log_param("test_size", config["test_size"])
        mlflow.log_param("random_state", config["random_state"])
        mlflow.log_param("handle_missing", config["handle_missing"])
        mlflow.log_param("scale_features", config["scale_features"])
        mlflow.log_param("features_dropped", str(config["features_to_drop"]))
        if config["model_type"] == "logistic_regression":
            mlflow.log_param("C", config["lr_C"])
        elif config["model_type"] == "random_forest":
            mlflow.log_param("n_estimators", config["rf_n_estimators"])
            mlflow.log_param("max_depth", str(config["rf_max_depth"]))
        elif config["model_type"] == "gradient_boosting":
            mlflow.log_param("n_estimators", config["gb_n_estimators"])
            mlflow.log_param("learning_rate", config["gb_learning_rate"])
            mlflow.log_param("max_depth", config["gb_max_depth"])
        X, y, n_rows, numeric_cols, categorical_cols = load_and_prepare_data(config)
        mlflow.log_param("n_rows", n_rows)
        mlflow.log_param("n_features", X.shape[1])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config["test_size"],
            random_state=config["random_state"],
            stratify=y
        )
        if config["scale_features"]:
            scaler = StandardScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        model = build_model(config)
        print(f"\nTraining {config['model_type']}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        mlflow.log_metric("accuracy", round(accuracy, 4))
        mlflow.log_metric("precision", round(precision, 4))
        mlflow.log_metric("recall", round(recall, 4))
        mlflow.log_metric("f1_score", round(f1, 4))
        mlflow.log_metric("auc_roc", round(auc, 4))
        mlflow.sklearn.log_model(model, "model")
        config_path = "config_snapshot.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)
        mlflow.log_artifact(config_path)
        os.remove(config_path)  # clean up temp file
        print(f"\n{'='*50}")
        print(f"Model:     {config['model_type']}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"AUC-ROC:   {auc:.4f}")
        print(f"{'='*50}")
        run_id = mlflow.active_run().info.run_id
        print(f"\nMLflow Run ID: {run_id}")
        print("View this run in the UI: mlflow ui")
    return run_id

if __name__ == "__main__":
    run_experiment(config)
