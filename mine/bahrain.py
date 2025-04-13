import fastf1
import pandas as pd
from utils import load_gp_data, feature_engineering, fine_tune_model

# Enable FastF1 caching
fastf1.Cache.enable_cache("f1_cache")

# 2025 Qualifying Data Bahrain GP
qualifying_2025 = pd.DataFrame(
    [
        {
            "Driver": "Oscar Piastri",
            "QualifyingTime": 89.841,
        },
        {
            "Driver": "George Russell",
            "QualifyingTime": 90.009,
        },
        {
            "Driver": "Lando Norris",
            "QualifyingTime": 90.267,
        },
        {
            "Driver": "Max Verstappen",
            "QualifyingTime": 90.423,
        },
        {
            "Driver": "Lewis Hamilton",
            "QualifyingTime": 90.772,
        },
        {
            "Driver": "Charles Leclerc",
            "QualifyingTime": 90.175,
        },
        {
            "Driver": "Isack Hadjar",
            "QualifyingTime": 91.271,
        },
        {
            "Driver": "Andrea Kimi Antonelli",
            "QualifyingTime": 90.213,
        },
        {
            "Driver": "Yuki Tsunoda",
            "QualifyingTime": 91.303,
        },
        {
            "Driver": "Alexander Albon",
            "QualifyingTime": 92.040,
        },
        {
            "Driver": "Esteban Ocon",
            "QualifyingTime": 91.594,
        },
        {
            "Driver": "Nico Hülkenberg",
            "QualifyingTime": 92.067,
        },
        {
            "Driver": "Fernando Alonso",
            "QualifyingTime": 91.886,
        },
        {
            "Driver": "Lance Stroll",
            "QualifyingTime": 91.334,
        },
        {
            "Driver": "Carlos Sainz Jr.",
            "QualifyingTime": 90.680,
        },
        {
            "Driver": "Pierre Gasly",
            "QualifyingTime": 90.216,
        },
        {
            "Driver": "Jack Doohan",
            "QualifyingTime": 91.245,
        },
        {
            "Driver": "Liam Lawson",
            "QualifyingTime": 92.165,
        },
        {
            "Driver": "Gabriel Bortoleto",
            "QualifyingTime": 92.186,
        },
        {
            "Driver": "Oliver Bearman",
            "QualifyingTime": 92.373,
        },
        {
            "Driver": "Lance Stroll",
            "QualifyingTime": 92.283,
        },
    ]
)

# Load the data
laps_data = load_gp_data(2024, "Bahrain GP")

# Feature engineering
data = feature_engineering(laps=laps_data, qualifying_data=qualifying_2025)
X = data["X"]
y = data["y"]
print(f"X shape: {X.shape}, columns: {X.columns}")
print(f"y shape: {y.shape}")

# Fine-tune the model
fine_tune_model(
    "Bahrain GP",
    "./models/chinese_gp_model.joblib",
    X,
    y,
    "./models/bahrain_gp_model.joblib",
)
