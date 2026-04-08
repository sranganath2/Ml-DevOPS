# Drift Monitoring Analysis

## Summary
The drift monitoring script compares a reference dataset against simulated production datasets using Evidently.

## Observed Drift
For the strongest simulated drift scenario, the following features showed drift:
- Age
- Family_Income
- Study_Hours_per_Day
- Attendance_Rate
- Department

These features likely drifted because the production-like dataset was intentionally modified to simulate population and behavior changes over time.

## Expected Impact on Model Performance
Drift in demographic and behavioral features can affect model performance because these inputs influence the learned decision patterns. If the relationship between these features and the target changes, prediction quality may decline.

## Recommended Action
The current drift level should trigger investigation and continued monitoring. If drift continues to increase or model performance drops in production, retraining should be considered.

## Run Command
```bash
PYTHONPATH=src python src/monitor_drift.py data/drift/reference_data.csv data/drift/month1_data.csv && python src/monitor_drift.py data/drift/reference_data.csv data/drift/month2_data.csv && python src/monitor_drift.py data/drift/reference_data.csv data/drift/month3_data.csv   
