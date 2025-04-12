import os
from fastf1 import Cache
import pandas as pd
from utils import load_gp_data, feature_engineering, train_model

cache_path = "f1_cache"
# Check if the cache directory exists
if not os.path.exists(cache_path):
    os.makedirs(cache_path)

# Enable FastF1 caching
Cache.enable_cache(cache_path)

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
            "Carlos Sainz Jr.",
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

# load the data
laps_data = load_gp_data(2024, "Australian GP")

# feature engineering
data = feature_engineering(laps=laps_data, qualifying_data=qualifying_2025)
X = data["X"]
y = data["y"]
print(f"X shape: {X.shape}, columns: {X.columns}")
print(f"y shape: {y.shape}")

# train the model
train_model(X, y, "./models/australian_gp_model.joblib")
