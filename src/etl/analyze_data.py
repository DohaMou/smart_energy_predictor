
# ==========================================
# SMART ENERGY PREDICTOR - CALAIS
# Exploratory Data Analysis (EDA)
# ==========================================

import os
import pandas as pd
import plotly.express as px

from sqlalchemy import create_engine, URL
from dotenv import load_dotenv


# ==========================================
# 1. LOAD ENVIRONMENT VARIABLES
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
print("SMART ENERGY PREDICTOR - CALAIS")
print("====================================")


# ==========================================
# 2. LOAD DATA FROM POSTGRESQL
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

print("\n--- DATA LOADED SUCCESSFULLY ---")
print("Dataset shape:", df.shape)


# ==========================================
# 3. CALCULATE CONSUMPTION PER METER
# ==========================================

df["consumption_per_meter"] = (
    df["energy_kwh"] / df["nb_points"]
)

print("\n--- CONSUMPTION PER METER ---")

print(
    df["consumption_per_meter"]
    .describe()
    .round(3)
)


# ==========================================
# 4. DATASET OVERVIEW
# ==========================================

print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- STATISTICS ---")

print(
    df[
        [
            "energy_kwh",
            "nb_points",
            "temperature_c",
            "consumption_per_meter"
        ]
    ].describe().round(3)
)


# ==========================================
# 5. CONVERT UTC TO FRENCH LOCAL TIME
# ==========================================

df["ts_local"] = (
    pd.to_datetime(df["ts"], utc=True)
    .dt.tz_convert("Europe/Paris")
)

print("\n--- LOCAL TIME CONVERSION ---")
print(df[["ts", "ts_local"]].head())


# ==========================================
# 6. TEMPERATURE / CONSUMPTION CORRELATION
# ==========================================

correlation = df[
    ["temperature_c", "consumption_per_meter"]
].corr().iloc[0, 1]

print("\n--- TEMPERATURE / CONSUMPTION CORRELATION ---")
print(f"Correlation: {correlation:.3f}")


# ==========================================
# 7. SCATTER PLOT
# ==========================================

fig_scatter = px.scatter(
    df,
    x="temperature_c",
    y="consumption_per_meter",
    opacity=0.35,
    title="Temperature vs Energy Consumption per Meter",
    labels={
        "temperature_c": "Temperature (°C)",
        "consumption_per_meter": "Consumption (kWh/meter/hour)"
    },
    trendline="ols"
)

fig_scatter.update_layout(
    template="plotly_white",
    title_x=0.5
)

fig_scatter.show()


# ==========================================
# 8. DAILY AVERAGE CONSUMPTION
# ==========================================

daily = (
    df.set_index("ts_local")["consumption_per_meter"]
    .resample("D")
    .mean()
    .reset_index()
)

print("\n--- DAILY DATA ---")
print(daily.head())


# ==========================================
# 9. DAILY CONSUMPTION GRAPH
# ==========================================

fig_daily = px.line(
    daily,
    x="ts_local",
    y="consumption_per_meter",
    title="Daily Average Energy Consumption per Meter - 2022",
    labels={
        "ts_local": "Date",
        "consumption_per_meter": "Consumption (kWh/meter/hour)"
    }
)

fig_daily.update_layout(
    template="plotly_white",
    title_x=0.5
)

fig_daily.show()


# ==========================================
# 10. FINISH
# ==========================================

print("\n====================================")
print("EDA COMPLETED SUCCESSFULLY!")
print("====================================")

engine.dispose()
