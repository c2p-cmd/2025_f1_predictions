import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import os

cache_path = "f1_cache"
# Check if the cache directory exists
if not os.path.exists(cache_path):
    os.makedirs(cache_path)

# Enable FastF1 caching
fastf1.Cache.enable_cache(cache_path)

time_features = [
    "LapTime",
    "Sector1Time",
    "Sector2Time",
    "Sector3Time",
]

# Load FastF1 2024 Australian GP race session
session_2024 = fastf1.get_session(2024, "Australian", "R")
session_2024.load()

# Extract lap times
laps_2024 = session_2024.laps[["Driver"] + time_features].copy()
laps_2024.dropna(subset=["LapTime"], inplace=True)

# Extract weather data
weather_features = [
    "AirTemp",
    "TrackTemp",
    "Humidity",
    "Pressure",
    "WindSpeed",
    "WindDirection",
    "Rainfall",
]
weather_2024 = session_2024.weather_data[weather_features].copy()

# Convert times to seconds
for col in time_features:
    laps_2024[col] = laps_2024[col].dt.total_seconds()

# Group by driver to get average sector times per driver
sector_times_2024 = (
    laps_2024.groupby("Driver")[
        ["Sector1Time", "Sector2Time", "Sector3Time"]
    ]
    .mean()
    .reset_index()
)

# 2025 Qualifying Data Australian GP
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
        "QualifyingTime": [
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
merged_data = qualifying_2025.merge(
    sector_times_2024, left_on="DriverCode", right_on="Driver", how="left"
)

# calculate average weather conditions for 2024 and add them to the merged data
weather_2024_avg = weather_2024.mean()
merged_data = merged_data.assign(
    AirTemp=weather_2024_avg["AirTemp"],
    TrackTemp=weather_2024_avg["TrackTemp"],
    Humidity=weather_2024_avg["Humidity"],
    Pressure=weather_2024_avg["Pressure"],
    WindSpeed=weather_2024_avg["WindSpeed"],
    WindDirection=weather_2024_avg["WindDirection"],
    Rainfall=weather_2024_avg["Rainfall"],
).fillna(0)

X = merged_data[
    ["QualifyingTime", "Sector1Time", "Sector2Time", "Sector3Time"] + weather_features
].fillna(0)
y = laps_2024.groupby("Driver")["LapTime"].mean().reset_index()["LapTime"]
print(X.shape, y.shape)

if X.shape[0] == 0:
    raise ValueError("Dataset is empty after preprocessing. Check data sources!")

# Train Gradient Boosting Model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=39
)
model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, random_state=39)
model.fit(X_train, y_train)

# Predict using 2025 qualifying times
predicted_lap_times = model.predict(X)
qualifying_2025["PredictedRaceTime"] = predicted_lap_times

# Rank drivers by predicted race time
qualifying_2025 = qualifying_2025.sort_values(by="PredictedRaceTime", ascending=False)

# Print final predictions
print("\n🏁 Predicted 2025 Chinese GP Winner 🏁\n")
print(qualifying_2025[["Driver", "PredictedRaceTime"]])

# Plot final predictions using plotly
import plotly.express as px

fig = px.bar(
    qualifying_2025,
    y="Driver",
    x="PredictedRaceTime",
    color="PredictedRaceTime",
    title="🏁 Predicted 2025 Chinese GP Winner with New Drivers and Sector Times 🏁",
    color_continuous_scale=px.colors.sequential.Plasma,
    text_auto=True,
    orientation="h",
)

# Evaluate Model
y_pred = model.predict(X_test)
print(f"\n🔍 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")

# add MAE as text
qualifying_2025["MAE"] = abs(
    qualifying_2025["PredictedRaceTime"] - laps_2024["LapTime"]
)
fig.add_trace(
    px.scatter(
        qualifying_2025,
        x="PredictedRaceTime",
        y="Driver",
        text="MAE",
        color_discrete_sequence=["black"],
    ).data[0]
)
fig.update_traces(textposition="outside")
fig.show()
