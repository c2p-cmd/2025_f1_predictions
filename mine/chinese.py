import fastf1
import pandas as pd
from utils import load_gp_data, feature_engineering, fine_tune_model

# Enable FastF1 caching
fastf1.Cache.enable_cache("f1_cache")

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

# load the data
laps_data = load_gp_data(2024, "Chinese GP")

# feature engineering
data = feature_engineering(laps=laps_data, qualifying_data=qualifying_2025)
X = data["X"]
y = data["y"]
print(f"X shape: {X.shape}, columns: {X.columns}")
print(f"y shape: {y.shape}")

print(y)

# train the model
fine_tune_model("./models/australian_gp_model.joblib", X, y, "./models/chinese_gp_model.joblib")