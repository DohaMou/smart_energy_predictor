
# ==========================================
# SMART ENERGY PREDICTOR - CALAIS
# MACHINE LEARNING MODEL TRAINING
# ==========================================

import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


# ==========================================
# 1. LOAD DATASET
# ==========================================

DATA_PATH = Path("data/processed/features_2022.csv")

df = pd.read_csv(DATA_PATH, parse_dates=["ts"])

df = df.sort_values("ts").reset_index(drop=True)

print("\n====================================")
print("SMART ENERGY PREDICTOR - ML TRAINING")
print("====================================")

print("Dataset shape:", df.shape)


# ==========================================
# 2. DEFINE FEATURES AND TARGET
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

target = "consumption_per_meter"

X = df[features]
y = df[target]


# ==========================================
# 3. CHRONOLOGICAL DATA SPLIT
# 70% TRAIN / 15% VALIDATION / 15% TEST
# ==========================================

n = len(df)

train_end = int(n * 0.70)
validation_end = int(n * 0.85)

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_val = X.iloc[train_end:validation_end]
y_val = y.iloc[train_end:validation_end]

X_test = X.iloc[validation_end:]
y_test = y.iloc[validation_end:]

print("\n--- DATA SPLIT ---")
print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))


# ==========================================
# 4. EVALUATION FUNCTION
# ==========================================

def evaluate_model(name, actual, predicted):

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    r2 = r2_score(actual, predicted)

    return {
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# ==========================================
# 5. BASELINE MODEL
# ==========================================

baseline_val = evaluate_model(
    "Baseline (lag 168h)",
    y_val,
    X_val["lag_168h"]
)

print("\n--- BASELINE VALIDATION ---")
print(baseline_val)


# ==========================================
# 6. RANDOM FOREST
# ==========================================

print("\nTraining Random Forest...")

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_val_predictions = rf_model.predict(X_val)

rf_results = evaluate_model(
    "Random Forest",
    y_val,
    rf_val_predictions
)

print("Random Forest validation:", rf_results)


# ==========================================
# 7. XGBOOST
# ==========================================

print("\nTraining XGBoost...")

xgb_model = XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)

xgb_val_predictions = xgb_model.predict(X_val)

xgb_results = evaluate_model(
    "XGBoost",
    y_val,
    xgb_val_predictions
)

print("XGBoost validation:", xgb_results)


# ==========================================
# 8. SELECT BEST MODEL ON VALIDATION
# ==========================================

results = pd.DataFrame([
    baseline_val,
    rf_results,
    xgb_results
])

print("\n--- VALIDATION COMPARISON ---")
print(results.to_string(index=False))

best_name = results.sort_values("MAE").iloc[0]["Model"]

print("\nBest validation model:", best_name)


# ==========================================
# 9. RETRAIN SELECTED MODEL
# ==========================================

X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

if best_name == "Random Forest":

    best_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

elif best_name == "XGBoost":

    best_model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

else:
    best_model = None


# ==========================================
# 10. FINAL TEST
# ==========================================

print("\n--- FINAL TEST RESULTS ---")

if best_model is not None:

    best_model.fit(X_train_val, y_train_val)

    test_predictions = best_model.predict(X_test)

    test_results = evaluate_model(
        best_name,
        y_test,
        test_predictions
    )

    print(test_results)

    baseline_test = evaluate_model(
        "Baseline (lag 168h)",
        y_test,
        X_test["lag_168h"]
    )

    print("\nBaseline test:", baseline_test)

    # ======================================
    # 11. SAVE MODEL
    # ======================================

    output_dir = Path("src/models/artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "best_model.joblib"

    joblib.dump(
        {
            "model": best_model,
            "features": features,
            "target": target,
            "model_name": best_name
        },
        model_path
    )

    print("\nModel saved to:", model_path)

else:
    print("Baseline is best. No ML model saved.")


# ==========================================
# 12. FINISH
# ==========================================

print("\n====================================")
print("MODEL TRAINING COMPLETED")
print("====================================")
