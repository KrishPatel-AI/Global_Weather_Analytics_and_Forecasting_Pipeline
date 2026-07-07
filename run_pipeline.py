"""
run_pipeline.py
----------------
Master script -- runs the ENTIRE project end-to-end in one command:

    python run_pipeline.py

Steps:
  1. Generate synthetic dataset (skipped automatically if data/GlobalWeatherRepository.csv
     already exists -- e.g. because you dropped in the real Kaggle CSV).
  2. Data cleaning & preprocessing            -> outputs/cleaned_weather_data.csv
  3. Exploratory Data Analysis                -> outputs/figures/01-05_*.png
  4. Basic forecasting model                  -> outputs/figures/06_*.png, outputs/basic_model_metrics.csv
  5. Advanced analyses (anomaly detection,
     multi-model forecasting + ensemble,
     climate/environmental/spatial analysis)  -> outputs/figures/07-13_*.png, outputs/advanced_model_comparison.csv
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
SRC_DIR = os.path.join(ROOT, "src")
RAW_CSV = os.path.join(DATA_DIR, "GlobalWeatherRepository.csv")


def run(script_name, cwd):
    print(f"\n{'='*70}\nRUNNING: {script_name}\n{'='*70}")
    result = subprocess.run([sys.executable, script_name], cwd=cwd)
    if result.returncode != 0:
        print(f"\n!! {script_name} exited with an error. Stopping pipeline. !!")
        sys.exit(1)


def main():
    if not os.path.exists(RAW_CSV):
        run("generate_synthetic_data.py", cwd=DATA_DIR)
    else:
        print(f"Found existing dataset at {RAW_CSV} -- skipping synthetic data generation.")
        print("(If you want to regenerate the synthetic sample, delete this file and re-run.)")

    run("data_cleaning.py", cwd=SRC_DIR)
    run("eda.py", cwd=SRC_DIR)
    run("forecasting_basic.py", cwd=SRC_DIR)
    run("advanced_analysis.py", cwd=SRC_DIR)

    print(f"\n{'='*70}\nPIPELINE COMPLETE")
    print(f"{'='*70}")
    print("Check the following folders for results:")
    print("  outputs/figures/        -> all 13 charts")
    print("  outputs/*.csv           -> cleaned data + model metrics")
    print("  report/REPORT.md        -> full written report")


if __name__ == "__main__":
    main()
