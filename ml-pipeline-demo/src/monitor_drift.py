import os
import sys
import json
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

DRIFT_SHARE_WARNING = 0.20
DRIFT_SHARE_CRITICAL = 0.40

def check_drift(reference_path, current_path):
    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)
    report = Report([DataDriftPreset()])
    my_eval = report.run(current_data=current, reference_data=reference)
    result = my_eval.dict()
    metrics = result["metrics"] 
    summary_metric = metrics[0]
    summary_value = summary_metric["value"]
    drifted = int(summary_value["count"])
    share = float(summary_value["share"])
    drifted_features = []
    total = 0
    for metric in metrics[1:]:
        metric_name = metric.get("metric_name", "")
        if metric_name.startswith("ValueDrift("):
            total += 1
            column_name = metric["config"]["column"]
            threshold = metric["config"]["threshold"]
            value = metric["value"]
            if value > threshold:
                drifted_features.append(column_name)
    check_result = {
        "total_features": total,
        "drifted_features": drifted,
        "drift_share": round(share, 3),
        "dataset_drift": share >= 0.5,
        "status": "ok",
        "drifted_feature_names": drifted_features,
    }
    if share >= DRIFT_SHARE_CRITICAL:
        check_result["status"] = "critical"
    elif share >= DRIFT_SHARE_WARNING:
        check_result["status"] = "warning"
    return check_result

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python drift_check.py <reference_data.csv> <current_data.csv>")
        sys.exit(1)
    reference_path = sys.argv[1]
    current_path = sys.argv[2]
    print(f"Checking drift: {current_path} vs {reference_path}")
    print("=" * 60)
    result = check_drift(reference_path, current_path)
    print(
        f"Features drifted: {result['drifted_features']}/{result['total_features']} "
        f"({result['drift_share'] * 100:.1f}%)"
    )
    print(f"Dataset drift:    {result['dataset_drift']}")
    print(f"Status:           {result['status'].upper()}")
    if result["drifted_feature_names"]:
        print(f"\nDrifted features: {', '.join(result['drifted_feature_names'])}")
    os.makedirs("reports", exist_ok=True)
    with open("reports/drift_check_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nFull result saved to reports/drift_check_result.json")
    if result["status"] == "critical":
        print(
            f"\nCRITICAL: {result['drift_share'] * 100:.1f}% of features drifted "
            f"(threshold: {DRIFT_SHARE_CRITICAL * 100:.0f}%)"
        )
        print("Action required: investigate and consider retraining.")
        sys.exit(1)
    elif result["status"] == "warning":
        print(
            f"\nWARNING: {result['drift_share'] * 100:.1f}% of features drifted "
            f"(threshold: {DRIFT_SHARE_WARNING * 100:.0f}%)"
        )
        print("Monitor closely. Retraining may be needed soon.")
        sys.exit(0)
    else:
        print("\nAll clear. Feature distributions are stable.")
        sys.exit(0)
