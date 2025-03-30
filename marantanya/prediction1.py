import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import os

cache_path = "../f1_cache"
# Check if the cache directory exists
if not os.path.exists(cache_path):
    os.makedirs(cache_path)

# Enable FastF1 caching
fastf1.Cache.enable_cache(cache_path)

# Load FastF1 2024 Australian GP race session
session_2024 = fastf1.get_session(2024, 3, "R")
session_2024.load()

# Extract lap times
laps_2024 = session_2024.laps[["Driver", "LapTime"]].copy()
laps_2024.dropna(subset=["LapTime"], inplace=True)
laps_2024["LapTime (s)"] = laps_2024["LapTime"].dt.total_seconds()

# 2025 Qualifying Data
qualifying_2025 = pd.DataFrame(
    {
        "Driver": [
            "Lando Norris",
            "Oscar Piastri",
            "Max Verstappen",
            "George Russell",
            "Yuki Tsunoda",
            "Alexander Albon",
            "Charles Leclerc",
            "Lewis Hamilton",
            "Pierre Gasly",
            "Carlos Sainz",
            "Fernando Alonso",
            "Lance Stroll",
        ],
        "QualifyingTime (s)": [
            75.096,
            75.180,
            75.481,
            75.546,
            75.670,
            75.737,
            75.755,
            75.973,
            75.980,
            76.062,
            76.4,
            76.5,
        ],
    }
)

# Map full names to FastF1 3-letter codes
driver_mapping = {
    "Lando Norris": "NOR",
    "Oscar Piastri": "PIA",
    "Max Verstappen": "VER",
    "George Russell": "RUS",
    "Yuki Tsunoda": "TSU",
    "Alexander Albon": "ALB",
    "Charles Leclerc": "LEC",
    "Lewis Hamilton": "HAM",
    "Pierre Gasly": "GAS",
    "Carlos Sainz": "SAI",
    "Lance Stroll": "STR",
    "Fernando Alonso": "ALO",
}

qualifying_2025["DriverCode"] = qualifying_2025["Driver"].map(driver_mapping)

# Merge 2025 Qualifying Data with 2024 Race Data
merged_data = qualifying_2025.merge(laps_2024, left_on="DriverCode", right_on="Driver")

# Use only "QualifyingTime (s)" as a feature
X = merged_data[["QualifyingTime (s)"]]
y = merged_data["LapTime (s)"]

if X.shape[0] == 0:
    raise ValueError("Dataset is empty after preprocessing. Check data sources!")

# Train Gradient Boosting Model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=39
)
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=39)
model.fit(X_train, y_train)

# Predict using 2025 qualifying times
predicted_lap_times = model.predict(qualifying_2025[["QualifyingTime (s)"]])
qualifying_2025["PredictedRaceTime (s)"] = predicted_lap_times

# Rank drivers by predicted race time
qualifying_2025 = qualifying_2025.sort_values(by="PredictedRaceTime (s)")

# Print final predictions
print("\n🏁 Predicted 2025 Chinese GP Winner 🏁\n")
print(qualifying_2025[["Driver", "PredictedRaceTime (s)"]])

# Evaluate Model
y_pred = model.predict(X_test)
print(f"\n🔍 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")

qualifying_2025["MAE"] = abs(
    qualifying_2025["PredictedRaceTime (s)"] - laps_2024["LapTime (s)"]
)
print("\n📊 MAE for each driver:")
print(qualifying_2025[["Driver", "MAE"]])

# Save the model
import joblib

models_dir = "models"
# Check if the models directory exists
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
# Save the model
model_filename = "f1_model_marantanya.pkl"
model_filename = os.path.join(models_dir, model_filename)
if os.path.exists(model_filename):
    os.remove(model_filename)
print(joblib.dump(model, model_filename))
print(f"\n📦 Model saved as {model_filename}")

# Save the predictions
predictions_dir = "predictions"
# Check if the predictions directory exists
if not os.path.exists(predictions_dir):
    os.makedirs(predictions_dir)
# Save predictions to CSV
predictions_filename = "f1_predictions_marantanya.csv"
predictions_dir = os.path.join(predictions_dir, predictions_filename)
if os.path.exists(predictions_dir):
    os.remove(predictions_dir)
qualifying_2025.to_csv(predictions_dir, index=False)
print(f"\n📊 Predictions saved as {predictions_filename}")

# show the predictions in plotly
import plotly.express as px

fig = px.bar(
    qualifying_2025,
    x="Driver",
    y="PredictedRaceTime (s)",
    color="DriverCode",
    title="2025 GP Predictions",
)

fig.show()
