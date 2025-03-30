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

# Load FastF1 2024 Chinese GP race session
session_2024 = fastf1.get_session(2024, "China", "R")
session_2024.load()

# Extract lap times
laps_2024 = session_2024.laps[["Driver", "LapTime"]].copy()
laps_2024.dropna(subset=["LapTime"], inplace=True)
laps_2024["LapTime (s)"] = laps_2024["LapTime"].dt.total_seconds()

# 2025 Qualifying Data Chinese GP
qualifying_2025 = pd.DataFrame(
    {
        "Driver": [
            "Oscar Piastri",
            "George Russell",
            "Lando Norris",
            "Max Verstappen",
            "Lewis Hamilton",
            "Charles Leclerc",
            "Isack Hadjar",
            "Andrea Kimi Antonelli",
            "Yuki Tsunoda",
            "Alexander Albon",
            "Esteban Ocon",
            "Nico Hülkenberg",
            "Fernando Alonso",
            "Lance Stroll",
            "Carlos Sainz Jr.",
            "Pierre Gasly",
            "Oliver Bearman",
            "Jack Doohan",
            "Gabriel Bortoleto",
            "Liam Lawson",
        ],
        "QualifyingTime (s)": [
            90.641,
            90.723,
            90.793,
            90.817,
            90.927,
            91.021,
            91.079,
            91.103,
            91.638,
            91.706,
            91.625,
            91.632,
            91.688,
            91.773,
            91.840,
            91.992,
            92.018,
            92.092,
            92.141,
            92.174,
        ],
    }
)

# Map full names to FastF1 3-letter codes
driver_mapping = {
    "Oscar Piastri": "PIA",
    "George Russell": "RUS",
    "Lando Norris": "NOR",
    "Max Verstappen": "VER",
    "Lewis Hamilton": "HAM",
    "Charles Leclerc": "LEC",
    "Isack Hadjar": "HAD",
    "Andrea Kimi Antonelli": "ANT",
    "Yuki Tsunoda": "TSU",
    "Alexander Albon": "ALB",
    "Esteban Ocon": "OCO",
    "Nico Hülkenberg": "HUL",
    "Fernando Alonso": "ALO",
    "Lance Stroll": "STR",
    "Carlos Sainz Jr.": "SAI",
    "Pierre Gasly": "GAS",
    "Oliver Bearman": "BEA",
    "Jack Doohan": "DOO",
    "Gabriel Bortoleto": "BOR",
    "Liam Lawson": "LAW",
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
print("\n🏁 Predicted 2025 Chinese GP Winner with no Change in ML Model🏁\n")
print(qualifying_2025[["Driver", "PredictedRaceTime (s)"]])

# Evaluate Model
y_pred = model.predict(X_test)
print(f"\n🔍 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")

qualifying_2025["MAE"] = abs(
    qualifying_2025["PredictedRaceTime (s)"] - laps_2024["LapTime (s)"]
)
print("\n📊 MAE for each driver:")
print(qualifying_2025[["Driver", "MAE"]])

# save the predictions to a CSV file
predictions_dir = "predictions"
# Check if the predictions directory exists
if not os.path.exists(predictions_dir):
    os.makedirs(predictions_dir)
# Save predictions to CSV
predictions_filename = "f1_predictions_ChineseGP_marantanya.csv"
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
    text_auto=True,
)

fig.show()
