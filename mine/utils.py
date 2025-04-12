import fastf1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
from joblib import dump, load

# Step 1: Load F1 Data from Fastf1
def load_gp_data(year, gp_name):
    session = fastf1.get_session(year, gp_name, 'Race')
    session.load()
    # Extract lap times and other features
    laps = session.laps
    laps = laps.reset_index()  # Reset index to make data easier to manipulate
    return laps

# Step 2: Feature Engineering
def feature_engineering(laps, qualifying_data):
    # Selecting relevant columns (e.g., lap times, sector times, car data, weather, etc.)
    features = laps[['Sector1Time', 'Sector2Time', 'Sector3Time', 'LapTime', 'Compound']]
    
    # Adding 2025 qualifying data as a feature
    features['QualiTime_2025'] = qualifying_data['QualiTime']
    
    # Converting lap times and sector times to numeric (total seconds)
    for col in ['Sector1Time', 'Sector2Time', 'Sector3Time', 'LapTime']:
        features[col] = features[col].dt.total_seconds()
    
    # Encoding categorical data (e.g., tire compound)
    features = pd.get_dummies(features, columns=['Compound'], drop_first=True)
    
    # Splitting features and target variable
    X = features.drop('LapTime', axis=1)
    y = features['LapTime']
    
    return X, y

# Step 3: Train and Save Model
def train_model(X, y, model_file):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features for neural networks (optional for XGBoost, but recommended for consistency)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train the model
    model = XGBRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
    
    # Save the model
    dump(model, model_file)
    print(f"Model saved to {model_file}")

# Step 4: Load Model for Fine-Tuning
def fine_tune_model(model_file, X_new, y_new):
    # Load existing model
    model = load(model_file)
    
    # Fine-tune the model
    model.fit(X_new, y_new)
    
    # Save updated model
    dump(model, model_file)
    print(f"Model fine-tuned and saved to {model_file}")

# Step 5: Main Script Workflow
if __name__ == "__main__":
    # Example: Load data for Australia GP (2024) and Qualifying data (2025)
    australian_gp_data = load_gp_data(2024, 'Australian GP')
    qualifying_data = pd.DataFrame({'QualiTime': [1.12, 1.13, 1.15]})  # Example qualifying data
    
    # Feature engineering
    X, y = feature_engineering(australian_gp_data, qualifying_data)
    
    # Train and save the model
    train_model(X, y, 'australian_gp_model.joblib')
    
    # Example: Fine-tune for another GP (e.g., Chinese GP)
    chinese_gp_data = load_gp_data(2024, 'Chinese GP')
    X_new, y_new = feature_engineering(chinese_gp_data, qualifying_data)
    fine_tune_model('australian_gp_model.joblib', X_new, y_new)
