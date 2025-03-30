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

# Load 2024 Chinese GP race session
session_2024 = fastf1.get_session(2024, "China", "R")
session_2024.load()

# Extract lap and sector times
laps_2024 = session_2024.laps[
    ["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]
].copy()
laps_2024.dropna(inplace=True)

# Convert times to seconds
for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

# Group by driver to get average sector times per driver
sector_times_2024 = (
    laps_2024.groupby("Driver")[
        ["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]
    ]
    .mean()
    .reset_index()
)

# 2025 Qualifying Data (Keeping Only 2024 Drivers)
qualifying_2025 = pd.DataFrame(
    {
        "Driver": [
            "Oscar Piastri",
            "George Russell",
            "Lando Norris",
            "Max Verstappen",
            "Lewis Hamilton",
            "Charles Leclerc",
            "Yuki Tsunoda",
            "Alexander Albon",
            "Esteban Ocon",
            "Nico Hülkenberg",
            "Fernando Alonso",
            "Lance Stroll",
            "Carlos Sainz Jr.",
            "Pierre Gasly",
        ],
        "QualifyingTime (s)": [
            90.641,
            90.723,
            90.793,
            90.817,
            90.927,
            91.021,
            91.638,
            91.706,
            91.625,
            91.632,
            91.688,
            91.773,
            91.840,
            91.992,
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
    "Yuki Tsunoda": "TSU",
    "Alexander Albon": "ALB",
    "Esteban Ocon": "OCO",
    "Nico Hülkenberg": "HUL",
    "Fernando Alonso": "ALO",
    "Lance Stroll": "STR",
    "Carlos Sainz Jr.": "SAI",
    "Pierre Gasly": "GAS",
}

qualifying_2025["DriverCode"] = qualifying_2025["Driver"].map(driver_mapping)

# Merge qualifying data with sector times
merged_data = qualifying_2025.merge(
    sector_times_2024, left_on="DriverCode", right_on="Driver", how="left"
)

# Define feature set (Qualifying + Sector Times)
X = merged_data[
    ["QualifyingTime (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]
].fillna(0)
y = merged_data.merge(
    laps_2024.groupby("Driver")["LapTime (s)"].mean(),
    left_on="DriverCode",
    right_index=True,
)["LapTime (s)"]


# Train Gradient Boosting Model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=38
)
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=38)
model.fit(X_train, y_train)

# Predict race times using 2025 qualifying and sector data
predicted_race_times = model.predict(X)
qualifying_2025["PredictedRaceTime (s)"] = predicted_race_times

# Rank drivers by predicted race time
qualifying_2025 = qualifying_2025.sort_values(by="PredictedRaceTime (s)")

# Print final predictions
print(
    "\n🏁 Predicted 2025 Chinese GP Winner (Just the Old Drivers, No New Drivers)🏁\n"
)
print(qualifying_2025[["Driver", "PredictedRaceTime (s)"]])

# Evaluate Model
y_pred = model.predict(X_test)
print(f"\n🔍 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")

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
    title="2025 GP Predictions (Old Drivers)",
    text_auto=True,
)

fig.show()
