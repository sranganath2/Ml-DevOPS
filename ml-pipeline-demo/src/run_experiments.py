import subprocess
import sys
import yaml
from pathlib import Path

CONFIG_PATH = Path("configs/config.yaml")

EXPERIMENTS = [
    {"random_seed": 42, "charge_threshold": 80.0, "score_threshold": 0.40},
    {"random_seed": 42, "charge_threshold": 80.0, "score_threshold": 0.35},
    {"random_seed": 42, "charge_threshold": 70.0, "score_threshold": 0.45},
    {"random_seed": 7,  "charge_threshold": 70.0, "score_threshold": 0.45},
    {"random_seed": 7,  "charge_threshold": 80.0, "score_threshold": 0.50},
]

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def save_config(cfg):
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

def main():
    base_cfg = load_config()

    for i, exp in enumerate(EXPERIMENTS, start=1):
        cfg = load_config()
        cfg["training"]["random_seed"] = exp["random_seed"]
        cfg["model"]["charge_threshold"] = exp["charge_threshold"]
        cfg["model"]["score_threshold"] = exp["score_threshold"]
        save_config(cfg)

        print(f"\n=== Running experiment {i}/{len(EXPERIMENTS)} ===")
        print(exp)

        result = subprocess.run([sys.executable, "src/train.py"])
        if result.returncode != 0:
            print(f"Experiment {i} finished with non-zero exit code: {result.returncode}")

    save_config(base_cfg)
    print("\nAll experiments completed and original config restored.")

if __name__ == "__main__":
    main()
