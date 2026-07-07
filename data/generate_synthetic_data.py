"""
generate_synthetic_data.py
---------------------------
Generates a SYNTHETIC stand-in for Kaggle's "Global Weather Repository.csv"
(https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository).

WHY THIS EXISTS:
The real dataset requires a Kaggle account + API token to download, and this
environment has no internet access. This script builds a dataset with the
EXACT same column schema, realistic seasonal/latitude-driven physics, injected
missing values, and injected outliers -- so every downstream script
(cleaning, EDA, forecasting, advanced analysis) runs correctly.

>>> TO USE THE REAL DATA INSTEAD <<<
1. Download "GlobalWeatherRepository.csv" from the Kaggle link above.
2. Drop it into this `data/` folder (replacing the synthetic file, or keep
   both and edit CSV_PATH in src/utils.py to point at the real file).
3. Re-run `python run_pipeline.py`. No other code changes are required,
   because the column names match the real dataset.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

CITIES = [
    # name, country, continent, lat, lon, tz, base_annual_mean_C, seasonal_amplitude_C
    ("New York", "USA", "North America", 40.71, -74.01, "America/New_York", 12.5, 12),
    ("Los Angeles", "USA", "North America", 34.05, -118.24, "America/Los_Angeles", 17.0, 6),
    ("Chicago", "USA", "North America", 41.88, -87.63, "America/Chicago", 10.0, 15),
    ("Toronto", "Canada", "North America", 43.65, -79.38, "America/Toronto", 8.5, 15),
    ("Mexico City", "Mexico", "North America", 19.43, -99.13, "America/Mexico_City", 16.0, 3),
    ("London", "United Kingdom", "Europe", 51.51, -0.13, "Europe/London", 11.0, 8),
    ("Paris", "France", "Europe", 48.85, 2.35, "Europe/Paris", 12.0, 9),
    ("Berlin", "Germany", "Europe", 52.52, 13.40, "Europe/Berlin", 10.0, 10),
    ("Madrid", "Spain", "Europe", 40.42, -3.70, "Europe/Madrid", 15.0, 10),
    ("Rome", "Italy", "Europe", 41.90, 12.50, "Europe/Rome", 16.0, 9),
    ("Moscow", "Russia", "Europe", 55.75, 37.62, "Europe/Moscow", 5.5, 18),
    ("Cairo", "Egypt", "Africa", 30.04, 31.24, "Africa/Cairo", 22.0, 8),
    ("Lagos", "Nigeria", "Africa", 6.52, 3.38, "Africa/Lagos", 27.0, 2),
    ("Nairobi", "Kenya", "Africa", -1.29, 36.82, "Africa/Nairobi", 19.0, 2),
    ("Johannesburg", "South Africa", "Africa", -26.20, 28.05, "Africa/Johannesburg", 16.0, 6),
    ("Casablanca", "Morocco", "Africa", 33.57, -7.59, "Africa/Casablanca", 18.0, 6),
    ("Beijing", "China", "Asia", 39.90, 116.41, "Asia/Shanghai", 12.5, 16),
    ("Tokyo", "Japan", "Asia", 35.68, 139.69, "Asia/Tokyo", 16.0, 11),
    ("Mumbai", "India", "Asia", 19.08, 72.88, "Asia/Kolkata", 27.5, 4),
    ("New Delhi", "India", "Asia", 28.61, 77.21, "Asia/Kolkata", 25.0, 8),
    ("Bangkok", "Thailand", "Asia", 13.75, 100.50, "Asia/Bangkok", 28.5, 3),
    ("Jakarta", "Indonesia", "Asia", -6.21, 106.85, "Asia/Jakarta", 27.0, 1),
    ("Seoul", "South Korea", "Asia", 37.57, 126.98, "Asia/Seoul", 12.5, 14),
    ("Dubai", "United Arab Emirates", "Asia", 25.20, 55.27, "Asia/Dubai", 28.0, 9),
    ("Singapore", "Singapore", "Asia", 1.35, 103.82, "Asia/Singapore", 27.5, 1),
    ("Sydney", "Australia", "Oceania", -33.87, 151.21, "Australia/Sydney", 18.0, 5),
    ("Melbourne", "Australia", "Oceania", -37.81, 144.96, "Australia/Melbourne", 15.0, 6),
    ("Auckland", "New Zealand", "Oceania", -36.85, 174.76, "Pacific/Auckland", 15.5, 4),
    ("Sao Paulo", "Brazil", "South America", -23.55, -46.63, "America/Sao_Paulo", 19.5, 4),
    ("Buenos Aires", "Argentina", "South America", -34.60, -58.38, "America/Argentina/Buenos_Aires", 17.5, 7),
    ("Lima", "Peru", "South America", -12.05, -77.04, "America/Lima", 19.0, 4),
    ("Bogota", "Colombia", "South America", 4.71, -74.07, "America/Bogota", 14.5, 1),
    ("Santiago", "Chile", "South America", -33.45, -70.67, "America/Santiago", 14.5, 8),
]

N_DAYS = 365  
START_DATE = datetime(2024, 7, 1)

rows = []
for name, country, continent, lat, lon, tz, base_mean, amp in CITIES:
    
    phase_shift = 0 if lat >= 0 else np.pi

    for day_offset in range(N_DAYS):
        date = START_DATE + timedelta(days=day_offset)
        day_of_year = date.timetuple().tm_yday

        
        seasonal = amp * np.sin(2 * np.pi * (day_of_year - 80) / 365 + phase_shift)
        temp_c = base_mean + seasonal + np.random.normal(0, 1.8)

        temp_f = temp_c * 9 / 5 + 32
        feels_like_c = temp_c + np.random.normal(-1, 1.5)
        feels_like_f = feels_like_c * 9 / 5 + 32

        humidity = np.clip(65 - 0.8 * seasonal + np.random.normal(0, 10), 10, 100)
        cloud = np.clip(np.random.beta(2, 2) * 100, 0, 100)

        
        precip_prob = 0.15 + 0.35 * (humidity / 100) * (cloud / 100)
        precip_mm = np.random.exponential(4) if np.random.rand() < precip_prob else 0.0
        precip_in = precip_mm / 25.4

        wind_kph = np.clip(np.random.gamma(2, 6), 0, 90)
        wind_mph = wind_kph / 1.609
        wind_degree = np.random.randint(0, 360)
        wind_dir = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][wind_degree // 45]

        gust_kph = wind_kph * np.random.uniform(1.1, 1.6)
        gust_mph = gust_kph / 1.609

        pressure_mb = np.random.normal(1013, 6)
        pressure_in = pressure_mb * 0.02953

        vis_km = np.clip(np.random.normal(10, 3), 0.5, 20)
        vis_miles = vis_km / 1.609

        uv = np.clip(np.random.normal(6 + seasonal / 4, 2), 0, 12)

        
        aqi_base = 40 + (30 if continent in ("Asia", "Africa") else 0)
        pm2_5 = np.clip(aqi_base * np.random.uniform(0.5, 1.8) - wind_kph * 0.3, 2, 300)
        pm10 = pm2_5 * np.random.uniform(1.2, 1.8)
        co = np.clip(np.random.normal(300, 80), 50, 1200)
        o3 = np.clip(np.random.normal(60, 20), 5, 200)
        no2 = np.clip(np.random.normal(20, 10), 1, 150)
        so2 = np.clip(np.random.normal(8, 5), 0.1, 80)
        us_epa_index = int(np.clip(np.digitize(pm2_5, [12, 35, 55, 150, 250]) + 1, 1, 6))
        gb_defra_index = int(np.clip(np.digitize(pm2_5, [11, 23, 35, 41, 47, 53, 58, 64, 70]) + 1, 1, 10))

        condition_pool = ["Sunny", "Partly cloudy", "Cloudy", "Overcast", "Light rain",
                           "Moderate rain", "Heavy rain", "Mist", "Clear"]
        if precip_mm > 10:
            condition = "Heavy rain"
        elif precip_mm > 2:
            condition = "Moderate rain"
        elif precip_mm > 0:
            condition = "Light rain"
        else:
            condition = np.random.choice(["Sunny", "Partly cloudy", "Cloudy", "Clear", "Overcast"],
                                          p=[0.3, 0.3, 0.2, 0.1, 0.1])

        sunrise_hr = 6 + np.random.uniform(-1, 1)
        sunset_hr = 18 + np.random.uniform(-1, 1)

        last_updated = date + timedelta(hours=int(np.random.choice([6, 9, 12, 15, 18])))

        rows.append({
            "country": country,
            "location_name": name,
            "region": continent,
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "timezone": tz,
            "last_updated_epoch": int(last_updated.timestamp()),
            "last_updated": last_updated.strftime("%Y-%m-%d %H:%M"),
            "temperature_celsius": round(temp_c, 1),
            "temperature_fahrenheit": round(temp_f, 1),
            "condition_text": condition,
            "wind_mph": round(wind_mph, 1),
            "wind_kph": round(wind_kph, 1),
            "wind_degree": wind_degree,
            "wind_direction": wind_dir,
            "pressure_mb": round(pressure_mb, 1),
            "pressure_in": round(pressure_in, 2),
            "precip_mm": round(precip_mm, 2),
            "precip_in": round(precip_in, 3),
            "humidity": int(humidity),
            "cloud": int(cloud),
            "feels_like_celsius": round(feels_like_c, 1),
            "feels_like_fahrenheit": round(feels_like_f, 1),
            "visibility_km": round(vis_km, 1),
            "visibility_miles": round(vis_miles, 1),
            "uv_index": round(uv, 1),
            "gust_mph": round(gust_mph, 1),
            "gust_kph": round(gust_kph, 1),
            "air_quality_Carbon_Monoxide": round(co, 1),
            "air_quality_Ozone": round(o3, 1),
            "air_quality_Nitrogen_dioxide": round(no2, 1),
            "air_quality_Sulphur_dioxide": round(so2, 1),
            "air_quality_PM2.5": round(pm2_5, 1),
            "air_quality_PM10": round(pm10, 1),
            "air_quality_us-epa-index": us_epa_index,
            "air_quality_gb-defra-index": gb_defra_index,
            "sunrise": f"{int(sunrise_hr):02d}:{int((sunrise_hr%1)*60):02d} AM",
            "sunset": f"{int(sunset_hr-12):02d}:{int((sunset_hr%1)*60):02d} PM",
            "moon_phase": np.random.choice(["New Moon", "Waxing Crescent", "First Quarter",
                                             "Waxing Gibbous", "Full Moon", "Waning Gibbous",
                                             "Last Quarter", "Waning Crescent"]),
            "moon_illumination": np.random.randint(0, 100),
        })

df = pd.DataFrame(rows)

# ---- 2. Inject realistic data-quality issues (so cleaning step has real work) ----
rng = np.random.default_rng(7)


for col in ["temperature_celsius", "humidity", "precip_mm", "wind_kph",
            "air_quality_PM2.5", "pressure_mb", "visibility_km", "uv_index"]:
    mask = rng.random(len(df)) < 0.03
    df.loc[mask, col] = np.nan


dropout_idx = rng.choice(df.index, size=25, replace=False)
df.loc[dropout_idx, ["temperature_celsius", "temperature_fahrenheit", "humidity", "wind_kph"]] = np.nan


outlier_idx = rng.choice(df.index, size=15, replace=False)
df.loc[outlier_idx[:5], "temperature_celsius"] = rng.uniform(70, 90, 5)   
df.loc[outlier_idx[5:10], "wind_kph"] = rng.uniform(400, 600, 5)         
df.loc[outlier_idx[10:], "precip_mm"] = rng.uniform(500, 900, 5)         


dupes = df.sample(10, random_state=1)
df = pd.concat([df, dupes], ignore_index=True)

df = df.sort_values(["location_name", "last_updated_epoch"]).reset_index(drop=True)

out_path = "GlobalWeatherRepository.csv" if __name__ != "__main__" else "GlobalWeatherRepository.csv"
df.to_csv(out_path, index=False)
print(f"Synthetic dataset written: {out_path}")
print(f"Shape: {df.shape}")
print(f"Cities: {df['location_name'].nunique()}  |  Countries: {df['country'].nunique()}  |  Date range: {df['last_updated'].min()} to {df['last_updated'].max()}")
