"""
data_cleaning.py
-----------------
Basic Assessment requirement: "Handle missing values, outliers, and normalize data."

Steps:
1. Load raw CSV, parse dates, drop exact duplicate rows.
2. Report missing values per column.
3. Impute missing numeric values with per-city median (falls back to global median).
4. Detect & cap outliers using the IQR method (winsorization) rather than deleting rows,
   so we don't throw away otherwise-valid days.
5. Add a Min-Max normalized version of key numeric columns (kept in separate `_norm`
   columns so the original units are still available for modeling/interpretation).
6. Save the cleaned dataset to outputs/cleaned_weather_data.csv
"""
import pandas as pd
import numpy as np
from utils import RAW_CSV_PATH, CLEAN_CSV_PATH, NUMERIC_COLS_FOR_CLEANING


def load_raw(path=RAW_CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["last_updated"] = pd.to_datetime(df["last_updated"])
    return df


def report_missing(df: pd.DataFrame) -> pd.Series:
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    return missing


def impute_missing(df: pd.DataFrame, cols=NUMERIC_COLS_FOR_CLEANING) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
       
        df[col] = df.groupby("location_name")[col].transform(lambda s: s.fillna(s.median()))
        
        df[col] = df[col].fillna(df[col].median())
    return df


def cap_outliers_iqr(df: pd.DataFrame, cols=NUMERIC_COLS_FOR_CLEANING, k=1.5,
                      percentile_cap_cols=("precip_mm", "precip_in")) -> tuple[pd.DataFrame, dict]:
    """
    Winsorize outliers instead of dropping rows.
    - Most columns: classic IQR fence (good for roughly symmetric distributions).
    - Zero-inflated/skewed columns (rainfall): IQR fires on almost every non-zero
      day because Q1==Q3==0 for many locations, so we cap those at the 99th
      percentile instead, which only trims genuine extreme downpours.
    """
    df = df.copy()
    outlier_counts = {}
    for col in cols:
        if col not in df.columns:
            continue
        if col in percentile_cap_cols:
            upper = df[col].quantile(0.99)
            lower = 0  
        else:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - k * iqr, q3 + k * iqr
        n_out = int(((df[col] < lower) | (df[col] > upper)).sum())
        outlier_counts[col] = n_out
        df[col] = df[col].clip(lower, upper)
    return df, outlier_counts


def normalize_minmax(df: pd.DataFrame, cols=("temperature_celsius", "humidity", "precip_mm",
                                              "wind_kph", "air_quality_PM2.5")) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        c_min, c_max = df[col].min(), df[col].max()
        df[f"{col}_norm"] = (df[col] - c_min) / (c_max - c_min) if c_max > c_min else 0.0
    return df


def clean_pipeline(save=True) -> pd.DataFrame:
    df = load_raw()
    n_before = len(df)

    df = df.drop_duplicates()
    n_after_dedup = len(df)

    missing_report = report_missing(df)

    df = impute_missing(df)
    df, outlier_counts = cap_outliers_iqr(df)
    df = normalize_minmax(df)

    if save:
        df.to_csv(CLEAN_CSV_PATH, index=False)

    print("=== DATA CLEANING SUMMARY ===")
    print(f"Rows before dedup:  {n_before}")
    print(f"Rows after dedup:   {n_after_dedup}  ({n_before - n_after_dedup} duplicates removed)")
    print("\nMissing values found (pre-imputation):")
    print(missing_report.to_string() if len(missing_report) else "  none")
    print("\nOutliers capped per column (IQR method):")
    for col, n in outlier_counts.items():
        if n > 0:
            print(f"  {col}: {n}")
    print(f"\nCleaned dataset saved to: {CLEAN_CSV_PATH}")
    print(f"Final shape: {df.shape}")
    return df


if __name__ == "__main__":
    clean_pipeline()
