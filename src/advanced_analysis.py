import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import (
    IsolationForest,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils import CLEAN_CSV_PATH, FIGURES_DIR, OUTPUTS_DIR, RANDOM_STATE

sns.set_theme(style="whitegrid")


def load_clean():
    df = pd.read_csv(CLEAN_CSV_PATH, parse_dates=["last_updated"])
    df["month"] = df["last_updated"].dt.month
    return df


def detect_anomalies(df):
    features = [
        "temperature_celsius",
        "humidity",
        "precip_mm",
        "wind_kph",
        "pressure_mb",
    ]
    X = df[features].fillna(df[features].median())

    iso = IsolationForest(
        contamination=0.02, random_state=RANDOM_STATE, n_estimators=200
    )
    df = df.copy()
    df["anomaly"] = iso.fit_predict(X)  # -1 = anomaly, 1 = normal
    n_anom = (df["anomaly"] == -1).sum()

    plt.figure(figsize=(9, 6))
    sample = df.sample(min(4000, len(df)), random_state=RANDOM_STATE)
    sns.scatterplot(
        data=sample[sample["anomaly"] == 1],
        x="temperature_celsius",
        y="wind_kph",
        alpha=0.35,
        s=15,
        color="#457b9d",
        label="Normal",
    )
    sns.scatterplot(
        data=sample[sample["anomaly"] == -1],
        x="temperature_celsius",
        y="wind_kph",
        alpha=0.9,
        s=45,
        color="#e63946",
        label="Anomaly",
        marker="X",
    )
    plt.title(
        f"Isolation Forest Anomaly Detection ({n_anom} anomalies flagged, {n_anom/len(df)*100:.1f}%)"
    )
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Wind Speed (kph)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/07_anomaly_detection.png", dpi=150)
    plt.close()

    print(
        f"[Anomaly Detection] Flagged {n_anom} anomalous records ({n_anom/len(df)*100:.2f}% of data)"
    )
    top_anom = df[df["anomaly"] == -1].nlargest(5, "temperature_celsius")[
        [
            "location_name",
            "last_updated",
            "temperature_celsius",
            "wind_kph",
            "precip_mm",
        ]
    ]
    print("Top 5 anomalies by temperature:")
    print(top_anom.to_string(index=False))
    return df


def build_daily_features(df):
    daily = (
        df.set_index("last_updated")
        .resample("D")
        .agg(
            {
                "temperature_celsius": "mean",
                "humidity": "mean",
                "precip_mm": "mean",
                "wind_kph": "mean",
                "pressure_mb": "mean",
            }
        )
        .dropna()
        .reset_index()
        .rename(columns={"last_updated": "date"})
    )
    daily["day_of_year"] = daily["date"].dt.dayofyear
    daily["t_index"] = np.arange(len(daily))
    daily["sin_doy"] = np.sin(2 * np.pi * daily["day_of_year"] / 365.25)
    daily["cos_doy"] = np.cos(2 * np.pi * daily["day_of_year"] / 365.25)
    # lag features (yesterday's readings help predict today's temperature)
    for lag in (1, 2, 3, 7):
        daily[f"temp_lag{lag}"] = daily["temperature_celsius"].shift(lag)
    daily = daily.dropna().reset_index(drop=True)
    return daily


def multi_model_forecast(daily):
    feature_cols = [
        "t_index",
        "sin_doy",
        "cos_doy",
        "humidity",
        "wind_kph",
        "pressure_mb",
        "temp_lag1",
        "temp_lag2",
        "temp_lag3",
        "temp_lag7",
    ]
    target = "temperature_celsius"

    split = int(len(daily) * 0.8)
    train, test = daily.iloc[:split], daily.iloc[split:]
    X_train, y_train = train[feature_cols], train[target]
    X_test, y_test = test[feature_cols], test[target]

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=6, random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE
        ),
    }

    results = {}
    preds_all = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        preds_all[name] = preds
        results[name] = {
            "MAE": mean_absolute_error(y_test, preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
            "MAPE (%)": np.mean(np.abs((y_test.values - preds) / y_test.values)) * 100,
            "R2": r2_score(y_test, preds),
        }

    ensemble_preds = np.mean([preds_all[n] for n in models], axis=0)
    results["Ensemble (avg of 3)"] = {
        "MAE": mean_absolute_error(y_test, ensemble_preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, ensemble_preds)),
        "MAPE (%)": np.mean(np.abs((y_test.values - ensemble_preds) / y_test.values))
        * 100,
        "R2": r2_score(y_test, ensemble_preds),
    }
    preds_all["Ensemble (avg of 3)"] = ensemble_preds

    results_df = (
        pd.DataFrame(results).T.reset_index().rename(columns={"index": "model"})
    )
    results_df.to_csv(f"{OUTPUTS_DIR}/advanced_model_comparison.csv", index=False)

    plt.figure(figsize=(12, 5))
    plt.plot(test["date"], y_test, label="Actual", color="black", linewidth=2)
    colors = {
        "Linear Regression": "#457b9d",
        "Random Forest": "#2a9d8f",
        "Gradient Boosting": "#e9c46a",
        "Ensemble (avg of 3)": "#e63946",
    }
    for name, preds in preds_all.items():
        style = "-" if name == "Ensemble (avg of 3)" else "--"
        lw = 2.2 if name == "Ensemble (avg of 3)" else 1.4
        plt.plot(
            test["date"],
            preds,
            label=name,
            linestyle=style,
            linewidth=lw,
            color=colors[name],
        )
    plt.title("Multi-Model Forecast Comparison (Held-Out Test Period)")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/08_multi_model_forecast.png", dpi=150)
    plt.close()

    print("\n=== MULTI-MODEL FORECAST COMPARISON ===")
    print(results_df.to_string(index=False))
    return results_df, models, feature_cols, X_train, y_train, X_test, y_test


def climate_analysis(df):
    monthly = (
        df.groupby(["region", "month"])["temperature_celsius"].mean().reset_index()
    )
    pivot = monthly.pivot(index="month", columns="region", values="temperature_celsius")

    plt.figure(figsize=(11, 6))
    sns.heatmap(
        pivot.T, cmap="RdYlBu_r", annot=True, fmt=".1f", cbar_kws={"label": "°C"}
    )
    plt.title("Climate Normals: Avg Monthly Temperature by Continent")
    plt.xlabel("Month")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/09_climate_normals_heatmap.png", dpi=150)
    plt.close()
    print("\n[Climate Analysis] Monthly climate-normal heatmap saved.")
    return pivot


def environmental_impact(df):
    aqi_cols = [
        "air_quality_PM2.5",
        "air_quality_PM10",
        "air_quality_Carbon_Monoxide",
        "air_quality_Ozone",
        "air_quality_Nitrogen_dioxide",
        "air_quality_Sulphur_dioxide",
    ]
    weather_cols = [
        "temperature_celsius",
        "humidity",
        "wind_kph",
        "pressure_mb",
        "precip_mm",
    ]
    corr = df[aqi_cols + weather_cols].corr().loc[aqi_cols, weather_cols]

    plt.figure(figsize=(9, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="PRGn", center=0)
    plt.title("Environmental Impact: Air Quality vs. Weather Parameters")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/10_air_quality_correlation.png", dpi=150)
    plt.close()

    print("\n[Environmental Impact] Air quality vs weather correlation matrix saved.")
    print(corr.round(2).to_string())
    return corr


def feature_importance(models, feature_cols, X_train, y_train, X_test, y_test):
    rf = models["Random Forest"]
    tree_importance = pd.Series(
        rf.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)

    perm = permutation_importance(
        rf, X_test, y_test, n_repeats=20, random_state=RANDOM_STATE
    )
    perm_importance = pd.Series(perm.importances_mean, index=feature_cols).sort_values(
        ascending=False
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    tree_importance.plot(kind="barh", ax=axes[0], color="#2a9d8f")
    axes[0].invert_yaxis()
    axes[0].set_title("Random Forest: Built-in Feature Importance")

    perm_importance.plot(kind="barh", ax=axes[1], color="#e76f51")
    axes[1].invert_yaxis()
    axes[1].set_title("Permutation Importance (test set)")

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/11_feature_importance.png", dpi=150)
    plt.close()

    print("\n[Feature Importance] Top features (Random Forest built-in):")
    print(tree_importance.head(5).to_string())
    print("\n[Feature Importance] Top features (Permutation, test-set):")
    print(perm_importance.head(5).to_string())
    return tree_importance, perm_importance


def spatial_analysis(df):
    city_avg = (
        df.groupby(["location_name", "country", "region", "latitude", "longitude"])
        .agg(
            avg_temp=("temperature_celsius", "mean"),
            avg_pm25=("air_quality_PM2.5", "mean"),
            avg_precip=("precip_mm", "mean"),
        )
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    sc1 = axes[0].scatter(
        city_avg["longitude"],
        city_avg["latitude"],
        c=city_avg["avg_temp"],
        cmap="RdYlBu_r",
        s=90,
        edgecolor="k",
        linewidth=0.4,
    )
    axes[0].set_title("Global Spatial Distribution: Avg Temperature by City")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    plt.colorbar(sc1, ax=axes[0], label="°C")

    sc2 = axes[1].scatter(
        city_avg["longitude"],
        city_avg["latitude"],
        c=city_avg["avg_pm25"],
        cmap="YlOrRd",
        s=90,
        edgecolor="k",
        linewidth=0.4,
    )
    axes[1].set_title("Global Spatial Distribution: Avg PM2.5 by City")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    plt.colorbar(sc2, ax=axes[1], label="µg/m³")

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/12_spatial_analysis.png", dpi=150)
    plt.close()

    cont_summary = (
        df.groupby("region")
        .agg(
            avg_temp=("temperature_celsius", "mean"),
            avg_humidity=("humidity", "mean"),
            avg_precip=("precip_mm", "mean"),
            avg_pm25=("air_quality_PM2.5", "mean"),
        )
        .sort_values("avg_temp", ascending=False)
    )

    cont_summary.plot(
        kind="bar",
        subplots=True,
        layout=(2, 2),
        figsize=(11, 8),
        legend=False,
        color="#264653",
    )
    plt.suptitle("Geographical Patterns: Weather & Air Quality by Continent")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/13_geographical_patterns.png", dpi=150)
    plt.close()

    print("\n[Spatial Analysis] City-level geographic scatter plots saved.")
    print("\n[Geographical Patterns] Continent summary:")
    print(cont_summary.round(1).to_string())
    return city_avg, cont_summary


def run_advanced():
    df = load_clean()
    df = detect_anomalies(df)
    daily = build_daily_features(df)
    results_df, models, feature_cols, X_train, y_train, X_test, y_test = (
        multi_model_forecast(daily)
    )
    climate_analysis(df)
    environmental_impact(df)
    feature_importance(models, feature_cols, X_train, y_train, X_test, y_test)
    spatial_analysis(df)
    print(f"\nAll advanced-analysis figures saved to {FIGURES_DIR}")
    return results_df


if __name__ == "__main__":
    run_advanced()
