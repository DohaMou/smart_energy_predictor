import logging
import os
import sys

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("enedis_etl")

DB_URL = (
    f"postgresql+psycopg://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
    f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
)

COLS = [
    "horodate", "region", "code_region", "profil", "plage", "nb_points",
    "total_wh", "courbe1_wh", "rep1", "courbe2_wh", "rep2",
    "courbe12_wh", "rep12", "jour_max", "semaine_max",
]
REGION_CODE = "32"  # Hauts-de-France
PROFILE_LABEL = "RES"  # residentiel, tous profils RES additionnes


def extract(path):
    """Lit le CSV par morceaux et garde RES + P0 + region 32."""
    parts = []
    for chunk in pd.read_csv(
        path, sep=";", encoding="cp1252", header=0, names=COLS,
        dtype=str, chunksize=500_000,
    ):
        mask = (
            (chunk["code_region"].str.strip() == REGION_CODE)
            & chunk["profil"].str.startswith("RES", na=False)
            & chunk["plage"].str.startswith("P0", na=False)
        )
        parts.append(chunk.loc[mask, ["horodate", "profil", "nb_points", "total_wh"]])
    df = pd.concat(parts, ignore_index=True)
    log.info("%s : %d lignes RES/P0 extraites", os.path.basename(path), len(df))
    return df


def transform(df):
    """Nettoie, additionne les profils, passe de 30 min a 1 h (tout en UTC)."""
    df = df.copy()
    df["horodate"] = pd.to_datetime(df["horodate"])
    df["total_wh"] = pd.to_numeric(df["total_wh"], errors="coerce")
    df["nb_points"] = pd.to_numeric(df["nb_points"], errors="coerce")

    n_profiles = df["profil"].nunique()
    log.info("profils residentiels (%d) : %s", n_profiles, sorted(df["profil"].unique()))
    log.info("total_wh manquants : %d sur %d lignes", int(df["total_wh"].isna().sum()), len(df))

    half = (
        df.groupby("horodate")
        .agg(
            energy_wh=("total_wh", "sum"),
            nb_points=("nb_points", "sum"),
            n_ok=("total_wh", "count"),
        )
        .reset_index()
    )
    incomplete = half["n_ok"] < n_profiles
    log.info("demi-heures ecartees (profil manquant) : %d sur %d", int(incomplete.sum()), len(half))
    half = half.loc[~incomplete].copy()

    half["ts"] = half["horodate"].dt.floor("h")
    hourly = (
        half.groupby("ts")
        .agg(
            energy_wh=("energy_wh", "sum"),
            nb_points=("nb_points", "mean"),
            n_slots=("horodate", "count"),
        )
        .reset_index()
    )
    short = hourly["n_slots"] < 2
    log.info("heures ecartees (moins de 2 demi-heures) : %d", int(short.sum()))
    hourly = hourly.loc[~short].copy()

    hourly["energy_kwh"] = hourly["energy_wh"] / 1000.0
    hourly["profile"] = PROFILE_LABEL
    out = hourly[["ts", "profile", "energy_kwh", "nb_points"]]

    # controles qualite
    if out["ts"].duplicated().any():
        raise ValueError("doublons de timestamps apres agregation")
    if (out["energy_kwh"] < 0).any():
        raise ValueError("energie negative detectee")
    per_meter = out["energy_kwh"] / out["nb_points"]
    log.info("%d heures pretes : %s -> %s (UTC)", len(out), out["ts"].min(), out["ts"].max())
    log.info("kWh par compteur et par heure : moyenne %.3f, min %.3f, max %.3f",
             per_meter.mean(), per_meter.min(), per_meter.max())
    return out


def load(df):
    """Upsert dans consumption_hourly (relancable sans doublons)."""
    engine = create_engine(DB_URL)
    with engine.begin() as conn:
        loc_id = conn.execute(
            text("SELECT id FROM dim_location WHERE name = 'Calais'")
        ).scalar_one()
        stage = df.copy()
        stage["location_id"] = loc_id
        stage.to_sql("consumption_stage", conn, if_exists="replace", index=False)
        conn.execute(
            text(
                """
                INSERT INTO consumption_hourly
                    (ts, location_id, profile, energy_kwh, nb_points)
                SELECT ts, location_id, profile, energy_kwh, nb_points
                FROM consumption_stage
                ON CONFLICT (ts, location_id, profile)
                DO UPDATE SET energy_kwh = EXCLUDED.energy_kwh,
                              nb_points = EXCLUDED.nb_points
                """
            )
        )
        conn.execute(text("DROP TABLE consumption_stage"))
    log.info("%d lignes chargees dans consumption_hourly", len(df))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage : python src/etl/enedis_etl.py fichier1.csv [fichier2.csv ...]")
    clean = pd.concat([transform(extract(p)) for p in sys.argv[1:]], ignore_index=True)
    load(clean)

