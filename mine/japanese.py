import fastf1
import pandas as pd
from utils import load_gp_data, feature_engineering, fine_tune_model

# Enable FastF1 caching
fastf1.Cache.enable_cache("f1_cache")

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


# load the data
laps_data = load_gp_data(2024, "Japanese GP")

# feature engineering
data = feature_engineering(laps=laps_data, qualifying_data=qualifying_2025)
X = data["X"]
y = data["y"]

print(f"X shape: {X.shape}, columns: {X.columns}")
print(f"y shape: {y.shape}")

# fine-tune the model
fine_tune_model("./models/chinese_gp_model.joblib", X, y, "./models/japanese_gp_model.joblib")
