# Weather Trend Forecasting — Data Science Technical Assessment

**PM Accelerator | AI Engineer + Data Science Dual-Role Technical Assessment**
**Author:** Krish Patel


## Project Objective

Analyze the **Global Weather Repository** dataset (daily weather + air-quality
readings for cities worldwide, 40+ features) to uncover trends and build
forecasting models — completing **both** the Basic and Advanced tracks of the
assessment:

| Track | Covered |
|---|---|
| **Basic** | Data cleaning & preprocessing · EDA (temperature & precipitation) · Basic forecasting model + evaluation metrics |
| **Advanced** | Anomaly detection · Multi-model forecasting + ensemble · Climate analysis · Environmental impact (air quality) · Feature importance · Spatial & geographical pattern analysis |

---

##  A Note on the Dataset

The real dataset lives on Kaggle and requires a free account to download:
 https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository

This project ships with a **synthetic dataset** (`data/GlobalWeatherRepository.csv`)
that mirrors the real file's exact schema (same 40+ columns: temperature,
precipitation, wind, humidity, pressure, air quality indices, lat/long,
sunrise/sunset, etc.) across 33 global cities over a full year, with realistic
seasonal patterns, missing values, and outliers deliberately injected so the
cleaning/EDA/modeling steps have real work to do.

**To use the real Kaggle data instead:**
1. Download `GlobalWeatherRepository.csv` from the link above.
2. Replace the file at `data/GlobalWeatherRepository.csv` with it (same filename).
3. Run `python run_pipeline.py` — no code changes needed; the column names match.

---

##  Project Structure

```
weather-trend-forecasting/
├── README.md                      <- you are here
├── requirements.txt
├── run_pipeline.py                 <- run this to build everything, start to finish
├── data/
│   ├── generate_synthetic_data.py  <- builds the synthetic dataset (auto-run if no CSV present)
│   └── GlobalWeatherRepository.csv
├── src/
│   ├── utils.py                    <- shared paths/constants
│   ├── data_cleaning.py            <- missing values, outliers, normalization
│   ├── eda.py                      <- basic EDA + temperature/precip visualizations
│   ├── forecasting_basic.py        <- basic time-series forecasting model
│   └── advanced_analysis.py        <- anomaly detection, multi-model ensemble,
│                                       climate/environmental/spatial analysis
├── outputs/
│   ├── figures/                    <- all 13 generated charts (.png)
│   ├── cleaned_weather_data.csv
│   ├── basic_model_metrics.csv
│   └── advanced_model_comparison.csv
└── report/
    └── REPORT.md                   <- full written report: methodology + insights
```

---

##  How to Run This

### 1. Set up your environment
```bash
# (optional but recommended) create a virtual environment
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

### 2. Run the entire pipeline with one command
```bash
python run_pipeline.py
```
This will:
1. Generate the synthetic dataset (skipped automatically if a CSV already
   exists in `data/` — e.g. if you dropped in the real Kaggle file).
2. Clean the data (missing values, outliers, normalization).
3. Run EDA and save 5 visualizations.
4. Train a basic forecasting model and evaluate it.
5. Run all advanced analyses (anomaly detection, multi-model ensemble,
   climate/environmental/spatial analysis) and save 8 more visualizations.

Total runtime: under a minute on a laptop.

### 3. Explore the results
- **`outputs/figures/`** — all 13 charts, numbered in the order they're discussed in the report.
- **`outputs/*.csv`** — cleaned dataset + model comparison metrics.
- **`report/REPORT.md`** — the full write-up: methodology, results, and insights.

### Run steps individually (optional)
```bash
cd data && python generate_synthetic_data.py && cd ..
cd src
python data_cleaning.py
python eda.py
python forecasting_basic.py
python advanced_analysis.py
```

---

##  Methodology Summary

1. **Cleaning:** per-city median imputation for missing sensor readings;
   IQR-based winsorization for outliers (percentile-based capping for
   zero-inflated rainfall data specifically, since IQR over-flags skewed
   distributions); Min-Max normalization added as separate `_norm` columns
   so original units stay available for modeling.
2. **EDA:** temperature/precipitation distributions, correlation matrix,
   weekly trend lines by continent, temperature-vs-humidity relationships.
3. **Basic forecasting:** Linear Regression on a global daily-average series
   using trend + seasonal (sine/cosine of day-of-year) features, evaluated
   with MAE/RMSE/MAPE/R² against a naive persistence baseline.
4. **Advanced analysis:** Isolation Forest anomaly detection; a 3-model
   comparison (Linear Regression, Random Forest, Gradient Boosting) plus an
   averaged ensemble, using lag features (1/2/3/7-day) and seasonal terms;
   permutation + tree-based feature importance; climate normals by month and
   continent; air-quality-vs-weather correlation; spatial scatter maps by
   lat/long.

Full results, numbers, and discussion are in **`report/REPORT.md`**.

---

