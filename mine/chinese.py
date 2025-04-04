import fastf1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Enable FastF1 caching
fastf1.Cache.enable_cache("f1_cache")

time_features = [
    "LapTime",
    "Sector1Time",
    "Sector2Time",
    "Sector3Time",
]

# Load 2024 Chinese GP race session
session_2024 = fastf1.get_session(2024, "China", "R")
session_2024.load()

# Extract lap and sector times
laps_2024 = session_2024.laps[["Driver"] + time_features].copy()
laps_2024.dropna(inplace=True)

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
        "QualifyingTime": [
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

# Merge qualifying data with sector times
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

# Define feature set (Qualifying + Sector Times)
X = merged_data[
    ["QualifyingTime", "Sector1Time", "Sector2Time", "Sector3Time"] + weather_features
].fillna(0)
y = laps_2024.groupby("Driver")["LapTime"].mean().reset_index()["LapTime"]
print(X.shape, y.shape)

# Train Gradient Boosting Model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=38
)
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=38)
model.fit(X_train, y_train)

# Predict race times using 2025 qualifying and sector data
predicted_race_times = model.predict(X)
qualifying_2025["PredictedRaceTime"] = predicted_race_times

# Rank drivers by predicted race time
qualifying_2025 = qualifying_2025.sort_values(by="PredictedRaceTime", ascending=False)

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
fig.show()

# Evaluate Model
y_pred = model.predict(X_test)
print(f"\n🔍 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")
