
# ==========================================
# SMART ENERGY PREDICTOR - CALAIS
# FEATURE ENGINEERING
# ==========================================

import os
import pandas as pd
import holidays

from pathlib import Path
from sqlalchemy import create_engine, URL
from dotenv import load_dotenv


# ==========================================
# 1. DATABASE CONNECTION
# ==========================================

load_dotenv()

db_url = URL.create(
    drivername="postgresql+psycopg2",
    username=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
    host=os.getenv("POSTGRES_HOST"),
    port=int(os.getenv("POSTGRES_PORT")),
    database=os.getenv("POSTGRES_DB"),
)

engine = create_engine(db_url)

print("\n====================================")
print("SMART ENERGY PREDICTOR - FEATURES")
print("====================================")


# ==========================================
# 2. LOAD DATA
# ==========================================

query = """
SELECT
    c.ts,
    c.energy_kwh,
    c.nb_points,
    w.temperature_c
FROM consumption_hourly c
JOIN weather_hourly w
    USING (ts, location_id)
ORDER BY c.ts;
"""

df = pd.read_sql(query, engine)

print("\nData loaded:", df.shape)


# ==========================================
# 3. CALCULATE TARGET
# ==========================================

df["consumption_per_meter"] = (
    df["energy_kwh"] / df["nb_points"]
)


# ==========================================
# 4. CONVERT UTC TO LOCAL TIME
# ==========================================

df["ts"] = pd.to_datetime(df["ts"], utc=True)

df["ts_local"] = df["ts"].dt.tz_convert(
    "Europe/Paris"
)

df = df.sort_values("ts").reset_index(drop=True)


# ==========================================
# 5. CALENDAR FEATURES
# ==========================================

df["hour"] = df["ts_local"].dt.hour

df["day_of_week"] = df["ts_local"].dt.dayofweek

df["day"] = df["ts_local"].dt.day

df["month"] = df["ts_local"].dt.month

df["day_of_year"] = df["ts_local"].dt.dayofyear

df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)


# ==========================================
# 6. FRENCH PUBLIC HOLIDAYS
# ==========================================

fr_holidays = holidays.country_holidays(
    "FR",
    years=[2021, 2022, 2023]
)

df["is_holiday"] = (
    df["ts_local"].dt.date
    .isin(fr_holidays)
    .astype(int)
)


# ==========================================
# 7. CYCLICAL TIME FEATURES
# ==========================================

import numpy as np

df["hour_sin"] = np.sin(
    2 * np.pi * df["hour"] / 24
)

df["hour_cos"] = np.cos(
    2 * np.pi * df["hour"] / 24
)

df["month_sin"] = np.sin(
    2 * np.pi * df["month"] / 12
)

df["month_cos"] = np.cos(
    2 * np.pi * df["month"] / 12
)


# ==========================================
# 8. LAG FEATURES
# ==========================================

target = "consumption_per_meter"

df["lag_1h"] = df[target].shift(1)

df["lag_24h"] = df[target].shift(24)

df["lag_168h"] = df[target].shift(168)


# ==========================================
# 9. ROLLING AVERAGE
# ==========================================

df["rolling_mean_24h"] = (
    df[target]
    .shift(1)
    .rolling(window=24)
    .mean()
)


# ==========================================
# 10. SELECT MODEL FEATURES
# ==========================================

features = [
    "temperature_c",
    "hour",
    "day_of_week",
    "day",
    "month",
    "day_of_year",
    "is_weekend",
    "is_holiday",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "lag_1h",
    "lag_24h",
    "lag_168h",
    "rolling_mean_24h"
]

target_column = "consumption_per_meter"

model_df = df[
    ["ts", "ts_local"] + features + [target_column]
].copy()


# ==========================================
# 11. REMOVE MISSING VALUES
# ==========================================

print("\nMissing values before cleaning:")
print(model_df.isnull().sum())

model_df = model_df.dropna().reset_index(drop=True)


# ==========================================
# 12. SAVE PROCESSED DATASET
# ==========================================

output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "features_2022.csv"

model_df.to_csv(output_file, index=False)


# ==========================================
# 13. FINAL REPORT
# ==========================================

print("\n====================================")
print("FEATURE ENGINEERING COMPLETED")
print("====================================")

print("Final dataset shape:", model_df.shape)

print("\nFeatures:")
print(features)

print("\nFirst rows:")
print(model_df.head())

print("\nMissing values after cleaning:")
print(model_df.isnull().sum())

print("\nSaved to:", output_file)

engine.dispose()
