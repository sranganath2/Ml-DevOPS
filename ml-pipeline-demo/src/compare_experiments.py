import mlflow

experiment = mlflow.get_experiment_by_name("ml-pipeline-demo")
if experiment is None:
    raise ValueError("Experiment 'ml-pipeline-demo' not found.")

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="status = 'FINISHED'",
    order_by=["metrics.accuracy DESC"]
)

print("Top Runs by Accuracy:")
print("=" * 80)

for _, row in runs.head(5).iterrows():
    print(f"\nRun: {row['run_id'][:8]}...")
    print(f"  Accuracy:  {row.get('metrics.accuracy', float('nan')):.4f}")
    print(f"  Precision: {row.get('metrics.precision', float('nan')):.4f}")
    print(f"  Recall:    {row.get('metrics.recall', float('nan')):.4f}")
    print(f"  Test ratio: {row.get('params.test_ratio', 'N/A')}")
    print(f"  Charge threshold: {row.get('params.charge_threshold', 'N/A')}")
    print(f"  Score threshold: {row.get('params.score_threshold', 'N/A')}")

best_run = runs.iloc[0]
print(f"\n{'=' * 80}")
print("BEST RUN")
print("=" * 80)
print(f"Run ID:     {best_run['run_id']}")
print(f"Accuracy:   {best_run.get('metrics.accuracy', float('nan')):.4f}")
print(f"Precision:  {best_run.get('metrics.precision', float('nan')):.4f}")
print(f"Recall:     {best_run.get('metrics.recall', float('nan')):.4f}")

print(f"\n{'=' * 80}")
print("Average Accuracy Across Runs")
print("=" * 80)
print(runs[["metrics.accuracy", "metrics.precision", "metrics.recall"]].mean().to_string())
