
import joblib
import pandas as pd
import plotly.express as px
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load dataset
df = pd.read_csv(
    "data/processed/features_2022.csv",
    parse_dates=["ts"]
)

# Define features
features = [
    "temperature_c", "hour", "day_of_week",
    "day", "month", "day_of_year",
    "is_weekend", "is_holiday",
    "hour_sin", "hour_cos",
    "month_sin", "month_cos",
    "lag_1h", "lag_24h", "lag_168h",
    "rolling_mean_24h"
]

target = "consumption_per_meter"

# Recreate chronological test split
test_start = int(len(df) * 0.85)

X_test = df[features].iloc[test_start:]
y_test = df[target].iloc[test_start:]

# Load trained model
artifact = joblib.load(
    "src/models/artifacts/best_model.joblib"
)

model = artifact["model"]

# Predict
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n--- FINAL MODEL EVALUATION ---")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")

# Comparison dataset
comparison = pd.DataFrame({
    "Date": df["ts"].iloc[test_start:].values,
    "Actual": y_test.values,
    "Predicted": predictions
})

# Save predictions
Path("data/processed").mkdir(exist_ok=True)

comparison.to_csv(
    "data/processed/predictions_test.csv",
    index=False
)

# Plot Actual vs Predicted
fig = px.line(
    comparison,
    x="Date",
    y=["Actual", "Predicted"],
    title="XGBoost - Actual vs Predicted Energy Consumption",
    labels={
        "value": "Consumption (kWh/meter/hour)",
        "Date": "Date",
        "variable": "Series"
    }
)

fig.update_layout(
    template="plotly_white",
    title_x=0.5
)

fig.show()

print("\nEvaluation completed successfully!")
