import fastf1
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
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
session_2024 = fastf1.get_session(2024, "Japanese", "R")
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
    laps_2024.groupby("Driver")[["Sector1Time", "Sector2Time", "Sector3Time"]]
    .mean()
    .reset_index()
)

# 2025 Qualifying Data Japanese GP
qualifying_2025 = pd.DataFrame(
    [
        {
            "Driver": "Lance Stroll",
            "QualifyingTime": 89.271,
        },
        {
            "Driver": "Jack Doohan",
            "QualifyingTime": 88.877,
        },
        {
            "Driver": "Esteban Ocon",
            "QualifyingTime": 88.696,
        },
        {
            "Driver": "Gabriel Bortoleto",
            "QualifyingTime": 88.622,
        },
        {
            "Driver": "Nico Hülkenberg",
            "QualifyingTime": 88.570,
        },
        {
            "Driver": "Yuki Tsunoda",
            "QualifyingTime": 88.000,
        },
        {
            "Driver": "Liam Lawson",
            "QualifyingTime": 87.906,
        },
        {
            "Driver": "Fernando Alonso",
            "QualifyingTime": 87.987,
        },
        {
            "Driver": "Carlos Sainz Jr.",
            "QualifyingTime": 87.836,
        },
        {
            "Driver": "Pierre Gasly",
            "QualifyingTime": 87.822,
        },
        {
            "Driver": "Oliver Bearman",
            "QualifyingTime": 87.867,
        },
        {
            "Driver": "Alexander Albon",
            "QualifyingTime": 87.615,
        },
        {
            "Driver": "Lewis Hamilton",
            "QualifyingTime": 87.610,
        },
        {
            "Driver": "Isack Hadjar",
            "QualifyingTime": 87.569,
        },
        {
            "Driver": "Andrea Kimi Antonelli",
            "QualifyingTime": 87.555,
        },
        {
            "Driver": "George Russell",
            "QualifyingTime": 87.318,
        },
        {
            "Driver": "Charles Leclerc",
            "QualifyingTime": 87.299,
        },
        {
            "Driver": "Oscar Piastri",
            "QualifyingTime": 87.027,
        },
        {
            "Driver": "Lando Norris",
            "QualifyingTime": 86.995,
        },
        {
            "Driver": "Max Verstappen",
            "QualifyingTime": 86.983,
        },
    ]
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
)
merged_data.drop(columns=["Driver_y"], inplace=True)
for sector in ["Sector1Time", "Sector2Time", "Sector3Time"]:
    merged_data[sector] = merged_data[sector].fillna(merged_data[sector].median())

laps_2024 = laps_2024.groupby("Driver")["LapTime"].mean().reset_index()

# Qualifying time correlation
missing_drivers = [
    d for d in merged_data["DriverCode"] if d not in laps_2024["Driver"].values
]
if missing_drivers:
    # Calculate ratio between qualifying and race pace for existing drivers
    pace_ratio = []
    for driver in laps_2024["Driver"]:
        if driver in merged_data["DriverCode"].values:
            qual_time = merged_data.loc[
                merged_data["DriverCode"] == driver, "QualifyingTime"
            ].values[0]
            race_time = laps_2024.loc[laps_2024["Driver"] == driver, "LapTime"].values[
                0
            ]
            pace_ratio.append(race_time / qual_time)

    # Use average ratio to predict missing drivers' race pace
    avg_pace_ratio = sum(pace_ratio) / len(pace_ratio)

    for driver in missing_drivers:
        print(f"Missing driver: {driver}")
        print(
            merged_data.loc[
                merged_data["DriverCode"] == driver, "QualifyingTime"
            ].values
        )
        qual_time = merged_data.loc[
            merged_data["DriverCode"] == driver, "QualifyingTime"
        ].values[0]
        estimated_race_time = qual_time * avg_pace_ratio

        # Add this driver to laps_2024 with estimated time
        laps_2024 = pd.concat(
            [
                laps_2024,
                pd.DataFrame({"Driver": [driver], "LapTime": [estimated_race_time]}),
            ],
            ignore_index=True,
        )


# remove drivers not in merged_data from laps_2024
laps_2024 = laps_2024[
    laps_2024["Driver"].isin(merged_data["DriverCode"].values)
].reset_index(drop=True)

# Merge laps data with qualifying data
merged_data = merged_data.merge(
    laps_2024, left_on="DriverCode", right_on="Driver", how="left"
)

# Define feature set (Qualifying + Sector Times)
X = merged_data[
    ["QualifyingTime", "Sector1Time", "Sector2Time", "Sector3Time"] + weather_features
].fillna(0)
y = merged_data["LapTime"]
print(X.shape, y.shape)

# Train Gradient Boosting Model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=38
)
model = GridSearchCV(
    XGBRegressor(objective="reg:squarederror", random_state=38),
    param_grid={
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 5, 7],
    },
    scoring="neg_mean_absolute_error",
    cv=5,
    verbose=1,
    n_jobs=-1,
)
model.fit(X_train, y_train)

print()

# Predict race times using 2025 qualifying and sector data
predicted_race_times = model.predict(X)
qualifying_2025["PredictedRaceTime"] = predicted_race_times

# Rank drivers by predicted race time
qualifying_2025 = qualifying_2025.sort_values(by="PredictedRaceTime", ascending=False)

print(
    qualifying_2025[["Driver", "PredictedRaceTime"]]
    .sort_values(by="PredictedRaceTime", ascending=True)
    .reset_index(drop=True)
)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"\n🔍 Model Error (MAE): {mae:.2f} seconds")

# Plot final predictions using plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=2,
    cols=1,
    subplot_titles=[
        "🏁 Predicted 2025 Japanese GP Winner with New Drivers and Sector Times 🏁",
        "Feature Importances from Model",
    ],
)

fig.add_trace(
    go.Bar(
        x=qualifying_2025["PredictedRaceTime"],
        y=qualifying_2025["Driver"],
        orientation="h",
        marker_color=qualifying_2025["PredictedRaceTime"],
    ),
    row=1,
    col=1,
)

fig.update_layout(height=800)

if hasattr(model.best_estimator_, "feature_importances_"):
    fig.add_trace(
        go.Bar(
            x=model.best_estimator_.feature_importances_,
            y=X.columns,
            orientation="h",
            marker_color=model.best_estimator_.feature_importances_,
        ),
        row=2,
        col=1,
    )

fig.data[0].marker.colorscale = "Viridis"
fig.update_layout(
    title_text=f"🏁 Predicted 2025 Japanese GP Winner with New Drivers and Sector Times 🏁<br>MAE: {mae:.2f} seconds",
)

fig.show()

# fig = px.bar(
#     qualifying_2025,
#     y="Driver",
#     x="PredictedRaceTime",
#     color="PredictedRaceTime",
#     title="🏁 Predicted 2025 Chinese GP Winner with New Drivers and Sector Times 🏁",
#     color_continuous_scale=px.colors.sequential.Plasma,
#     text_auto=True,
#     orientation="h",
# )
# fig.show()
