"""
eda.py
------
Basic Assessment requirement: "Perform basic EDA to uncover trends, correlations,
and patterns. Generate visualizations for temperature and precipitation."

Produces PNGs in outputs/figures/:
  01_temperature_distribution.png
  02_precipitation_distribution.png
  03_correlation_heatmap.png
  04_temperature_timeseries_by_continent.png
  05_temp_vs_humidity_scatter.png
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from utils import CLEAN_CSV_PATH, FIGURES_DIR

sns.set_theme(style="whitegrid")


def load_clean() -> pd.DataFrame:
    df = pd.read_csv(CLEAN_CSV_PATH)
    df["last_updated"] = pd.to_datetime(df["last_updated"])
    return df


def plot_temperature_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(df["temperature_celsius"], bins=40, kde=True, color="#e07a5f", ax=axes[0])
    axes[0].set_title("Distribution of Daily Temperature (°C)")
    axes[0].set_xlabel("Temperature (°C)")

    order = df.groupby("region")["temperature_celsius"].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x="region", y="temperature_celsius", order=order, ax=axes[1], palette="flare")
    axes[1].set_title("Temperature Spread by Continent")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/01_temperature_distribution.png", dpi=150)
    plt.close()


def plot_precipitation_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(df[df["precip_mm"] > 0]["precip_mm"], bins=40, kde=True, color="#457b9d", ax=axes[0])
    axes[0].set_title("Distribution of Precipitation on Rainy Days (mm)")
    axes[0].set_xlabel("Precipitation (mm)")

    rainy_share = (df.assign(is_rainy=df["precip_mm"] > 0)
                     .groupby("region")["is_rainy"].mean().sort_values(ascending=False) * 100)
    rainy_share.plot(kind="bar", ax=axes[1], color="#1d3557")
    axes[1].set_title("% of Days With Measurable Rain, by Continent")
    axes[1].set_ylabel("% rainy days")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/02_precipitation_distribution.png", dpi=150)
    plt.close()


def plot_correlation_heatmap(df):
    cols = ["temperature_celsius", "humidity", "precip_mm", "wind_kph", "pressure_mb",
             "cloud", "uv_index", "visibility_km", "air_quality_PM2.5", "air_quality_Ozone"]
    corr = df[cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Matrix: Weather & Air Quality Variables")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/03_correlation_heatmap.png", dpi=150)
    plt.close()
    return corr


def plot_timeseries_by_continent(df):
    daily = (df.set_index("last_updated")
               .groupby("region")["temperature_celsius"]
               .resample("W").mean()
               .reset_index())
    plt.figure(figsize=(12, 6))
    for region, grp in daily.groupby("region"):
        plt.plot(grp["last_updated"], grp["temperature_celsius"], label=region, linewidth=1.6)
    plt.title("Weekly Average Temperature Trend by Continent")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/04_temperature_timeseries_by_continent.png", dpi=150)
    plt.close()


def plot_temp_vs_humidity(df):
    plt.figure(figsize=(8, 6))
    sample = df.sample(min(3000, len(df)), random_state=42)
    sns.scatterplot(data=sample, x="temperature_celsius", y="humidity", hue="region",
                     alpha=0.5, s=18, palette="tab10")
    plt.title("Temperature vs. Humidity")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/05_temp_vs_humidity_scatter.png", dpi=150)
    plt.close()


def run_eda():
    df = load_clean()
    plot_temperature_distribution(df)
    plot_precipitation_distribution(df)
    corr = plot_correlation_heatmap(df)
    plot_timeseries_by_continent(df)
    plot_temp_vs_humidity(df)

    print("=== EDA SUMMARY ===")
    print(f"Global mean temperature: {df['temperature_celsius'].mean():.1f} °C")
    print(f"Global mean humidity: {df['humidity'].mean():.1f}%")
    print(f"% of days with measurable rain: {(df['precip_mm'] > 0).mean()*100:.1f}%")
    print("\nStrongest correlations with temperature:")
    print(corr["temperature_celsius"].drop("temperature_celsius").sort_values(key=abs, ascending=False).to_string())
    print(f"\n5 figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    run_eda()
