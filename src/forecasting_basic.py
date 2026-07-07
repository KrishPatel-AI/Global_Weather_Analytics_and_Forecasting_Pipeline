"""
forecasting_basic.py
---------------------
Basic Assessment requirement: "Build a basic forecasting model and evaluate its
performance using different metrics. Use the `last_updated` feature for the
time series analysis."

Approach:
- Build a global daily average temperature series from `last_updated`.
- Engineer simple time features (day-of-year sin/cos, trend index) -- this
  keeps the model fully self-contained (no statsmodels/Prophet dependency).
- Train/test split chronologically (last 20% of days = test set) -- this is
  the correct way to validate a forecasting model, since a random split would
  leak future information into training.
- Fit a Linear Regression baseline forecaster.
- Evaluate with MAE, RMSE, MAPE, R^2.
- Plot actual vs. predicted.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils import CLEAN_CSV_PATH, FIGURES_DIR


def build_daily_series(df: pd.DataFrame) -> pd.DataFrame:
    daily = (df.set_index("last_updated")
               .resample("D")["temperature_celsius"]
               .mean()
               .reset_index()
               .rename(columns={"last_updated": "date"}))
    daily["day_of_year"] = daily["date"].dt.dayofyear
    daily["t_index"] = np.arange(len(daily))  # linear trend index
    daily["sin_doy"] = np.sin(2 * np.pi * daily["day_of_year"] / 365.25)
    daily["cos_doy"] = np.cos(2 * np.pi * daily["day_of_year"] / 365.25)
    return daily.dropna(subset=["temperature_celsius"])


def evaluate(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape, "R2": r2}


def run_basic_forecast():
    df = pd.read_csv(CLEAN_CSV_PATH, parse_dates=["last_updated"])
    daily = build_daily_series(df)

    split = int(len(daily) * 0.8)
    train, test = daily.iloc[:split], daily.iloc[split:]

    features = ["t_index", "sin_doy", "cos_doy"]
    model = LinearRegression()
    model.fit(train[features], train["temperature_celsius"])

    preds = model.predict(test[features])
    metrics = evaluate(test["temperature_celsius"].values, preds)

  
    naive_preds = daily["temperature_celsius"].shift(1).iloc[split:]
    naive_metrics = evaluate(test["temperature_celsius"].values, naive_preds.values)

    print("=== BASIC FORECASTING MODEL (Linear Regression: trend + seasonal terms) ===")
    print(f"Train days: {len(train)}  |  Test days: {len(test)}")
    print("\nModel performance on held-out test period:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")
    print("\nNaive persistence baseline (for comparison):")
    for k, v in naive_metrics.items():
        print(f"  {k}: {v:.3f}")

    plt.figure(figsize=(12, 5))
    plt.plot(train["date"], train["temperature_celsius"], label="Train (actual)", color="#888", alpha=0.6)
    plt.plot(test["date"], test["temperature_celsius"], label="Test (actual)", color="#1d3557", linewidth=2)
    plt.plot(test["date"], preds, label="Forecast (Linear Reg.)", color="#e63946", linewidth=2, linestyle="--")
    plt.title("Basic Forecast: Global Daily Avg Temperature")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/06_basic_forecast.png", dpi=150)
    plt.close()

    metrics_df = pd.DataFrame([
        {"model": "Linear Regression (trend+seasonality)", **metrics},
        {"model": "Naive Persistence Baseline", **naive_metrics},
    ])
    metrics_df.to_csv("../outputs/basic_model_metrics.csv", index=False)
    print(f"\nMetrics saved to outputs/basic_model_metrics.csv")
    print(f"Figure saved to {FIGURES_DIR}/06_basic_forecast.png")
    return metrics_df


if __name__ == "__main__":
    run_basic_forecast()
