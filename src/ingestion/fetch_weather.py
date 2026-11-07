import os
import sys

import pandas as pd
import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DB_URL = (
    f"postgresql+psycopg://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
    f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
)
LAT, LON = 50.9513, 1.8587  # Calais


def fetch_weather(start: str, end: str) -> pd.DataFrame:
    r = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": LAT,
            "longitude": LON,
            "start_date": start,
            "end_date": end,
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
            "timezone": "GMT",
        },
        timeout=60,
    )
    r.raise_for_status()
    h = r.json()["hourly"]
    return pd.DataFrame(
        {
            "ts": pd.to_datetime(h["time"]),
            "temperature_c": h["temperature_2m"],
            "humidity_pct": h["relative_humidity_2m"],
            "wind_speed_kmh": h["wind_speed_10m"],
        }
    )


def load(df: pd.DataFrame) -> None:
    engine = create_engine(DB_URL)
    with engine.begin() as conn:
        loc_id = conn.execute(
            text("SELECT id FROM dim_location WHERE name = 'Calais'")
        ).scalar_one()
        df["location_id"] = loc_id
        df.to_sql("weather_stage", conn, if_exists="replace", index=False)
        conn.execute(
            text(
                """
                INSERT INTO weather_hourly
                    (ts, location_id, temperature_c, humidity_pct, wind_speed_kmh)
                SELECT ts, location_id, temperature_c, humidity_pct, wind_speed_kmh
                FROM weather_stage
                ON CONFLICT (ts, location_id) DO NOTHING
                """
            )
        )
        conn.execute(text("DROP TABLE weather_stage"))


if __name__ == "__main__":
    start, end = sys.argv[1], sys.argv[2]
    data = fetch_weather(start, end)
    load(data)
    print(f"{len(data)} lignes meteo chargees pour Calais ({start} -> {end})")
