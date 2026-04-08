# ML Pipeline Demo

## Project Overview
This project is an end-to-end MLOps pipeline for a customer churn prediction task. It includes dataset versioning with DVC, experiment tracking with MLflow, automated testing with pytest, CI/CD with GitHub Actions, and drift monitoring with Evidently.

## Dataset
The dataset used in this project is a customer churn dataset stored at:

`data/raw/customers.csv`

The prediction target is:

`churned`

## Project Structure
- `src/` - source code for training, preprocessing, evaluation, experiment comparison, and drift monitoring
- `tests/` - pytest test suite
- `configs/` - YAML configuration file
- `data/` - raw and drift-monitoring data
- `reports/` - generated drift reports
- `.github/workflows/` - GitHub Actions pipeline

## Setup
Install dependencies:

```bash
pip install -r requirements.txt
