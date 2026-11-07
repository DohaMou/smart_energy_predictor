import sys

import pandas as pd

COLS = [
    "horodate", "region", "code_region", "profil", "plage", "nb_points",
    "total_wh", "courbe1_wh", "rep1", "courbe2_wh", "rep2",
    "courbe12_wh", "rep12", "jour_max", "semaine_max",
]


def detect_sep(path):
    with open(path, encoding="cp1252") as f:
        first = f.readline()
    return ";" if first.count(";") > first.count(",") else ","


path = sys.argv[1]
sep = detect_sep(path)
print("separateur detecte :", repr(sep))

parts = []
codes = set()
for chunk in pd.read_csv(
    path, sep=sep, encoding="cp1252", header=0, names=COLS,
    dtype=str, chunksize=500_000,
):
    codes.update(chunk["code_region"].unique())
    parts.append(chunk[chunk["code_region"].str.strip() == "32"])

df = pd.concat(parts, ignore_index=True)
print("codes region presents :", sorted(codes))
print("lignes code 32 :", len(df))
print("nom de la region :", df["region"].unique())

df["horodate"] = pd.to_datetime(df["horodate"])
print("periode :", df["horodate"].min(), "->", df["horodate"].max())
print("pas de temps distincts :", df["horodate"].nunique())

print("\nprofils :")
print(df["profil"].value_counts().to_string())
print("\nplages de puissance :")
print(df["plage"].value_counts().to_string())

df["total_wh"] = pd.to_numeric(df["total_wh"], errors="coerce")
df["nb_points"] = pd.to_numeric(df["nb_points"], errors="coerce")
print("\ntotal_wh manquants :", int(df["total_wh"].isna().sum()))
print("nb_points manquants :", int(df["nb_points"].isna().sum()))

res = df[df["profil"].str.startswith("RES") & df["plage"].str.startswith("P0")]
print("\nlignes RES + P0 :", len(res))
per_ts = res.groupby("horodate")["total_wh"].sum()
by_hour = per_ts.groupby(per_ts.index.hour).mean().div(1e6).round(1)
print("\nenergie moyenne par heure de la colonne horodate (MWh par pas de 30 min) :")
print(by_hour.to_string())

df.to_csv("data/processed/hdf_t1.csv", index=False, encoding="utf-8")
print("\nextrait sauvegarde : data/processed/hdf_t1.csv")
