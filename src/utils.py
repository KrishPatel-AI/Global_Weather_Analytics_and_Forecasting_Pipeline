"""
utils.py - shared paths, constants, and helper functions.
"""
import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
REPORT_DIR = os.path.join(ROOT_DIR, "report")

RAW_CSV_PATH = os.path.join(DATA_DIR, "GlobalWeatherRepository.csv")
CLEAN_CSV_PATH = os.path.join(OUTPUTS_DIR, "cleaned_weather_data.csv")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

NUMERIC_COLS_FOR_CLEANING = [
    "temperature_celsius", "temperature_fahrenheit", "feels_like_celsius", "feels_like_fahrenheit",
    "wind_mph", "wind_kph", "pressure_mb", "pressure_in", "precip_mm", "precip_in",
    "humidity", "cloud", "visibility_km", "visibility_miles", "uv_index",
    "gust_mph", "gust_kph", "air_quality_Carbon_Monoxide", "air_quality_Ozone",
    "air_quality_Nitrogen_dioxide", "air_quality_Sulphur_dioxide",
    "air_quality_PM2.5", "air_quality_PM10",
]

RANDOM_STATE = 42
