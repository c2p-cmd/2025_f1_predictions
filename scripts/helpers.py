from pandas import DataFrame

# Driver mapping (full name to code)
driver_mapping = {
    "Max Verstappen": "VER",
    "Sergio Perez": "PER",
    "Lewis Hamilton": "HAM",
    "George Russell": "RUS",
    "Charles Leclerc": "LEC",
    "Carlos Sainz": "SAI",
    "Lando Norris": "NOR",
    "Oscar Piastri": "PIA",
    "Fernando Alonso": "ALO",
    "Lance Stroll": "STR",
    "Pierre Gasly": "GAS",
    "Esteban Ocon": "OCO",
    "Alexander Albon": "ALB",
    "Yuki Tsunoda": "TSU",
    "Valtteri Bottas": "BOT",
    "Guanyu Zhou": "ZHO",
    "Kevin Magnussen": "MAG",
    "Nico Hulkenberg": "HUL",
    "Daniel Ricciardo": "RIC",
    "Nyck de Vries": "DEV",
    "Logan Sargeant": "SAR",
    # Add more as needed
}

# Reverse mapping (code to full name)
driver_code_to_name = {v: k for k, v in driver_mapping.items()}


def load_qualifying_data(driver_names, qualifying_times) -> DataFrame:
    """Create a DataFrame with qualifying data"""
    return DataFrame({"DriverName": driver_names, "QualifyingTime": qualifying_times})
