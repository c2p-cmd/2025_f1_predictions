from typing import Union
import numpy as np
import fastf1
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd
from joblib import dump, load

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


# Step 1: Load F1 Data from Fastf1
def load_gp_data(year: int, gp_name: str) -> pd.DataFrame:
    """
    Load F1 data for a specific Grand Prix and year using FastF1.
    Args:
        year (int): Year of the Grand Prix.
        gp_name (str): Name of the Grand Prix.
    Returns:
        pd.DataFrame: DataFrame containing lap times and other features.
    """
    session = fastf1.get_session(year, gp_name, "Race")
    session.load()
    # Extract lap times and other features
    # 1.
    time_features = [
        "LapTime",
        "Sector1Time",
        "Sector2Time",
        "Sector3Time",
    ]
    # 2.
    tyre_features = [
        "TyreLife",
        "Compound",
    ]
    # 3.
    lap_features = [
        "PitInTime",
        "PitOutTime",
    ]
    # 4.
    driver_features = [
        "Driver",
        # "Team",
    ]
    # 5. weather features
    weather_features = [
        "AirTemp",
        "TrackTemp",
        "Humidity",
        "Pressure",
        "WindSpeed",
        "WindDirection",
        "Rainfall",
    ]
    laps = session.laps[
        driver_features + time_features + tyre_features + lap_features
    ].copy()
    laps.dropna(subset=["LapTime"], inplace=True)
    # Convert times to seconds
    for col in time_features:
        laps[col] = laps[col].dt.total_seconds()

    # Extract weather data
    weather_2024 = session.weather_data[weather_features].copy()
    weather_2024_avg = weather_2024.mean()
    # Merge weather data with laps data
    merged_data = laps.assign(
        AirTemp=weather_2024_avg["AirTemp"],
        TrackTemp=weather_2024_avg["TrackTemp"],
        Humidity=weather_2024_avg["Humidity"],
        Pressure=weather_2024_avg["Pressure"],
        WindSpeed=weather_2024_avg["WindSpeed"],
        WindDirection=weather_2024_avg["WindDirection"],
        Rainfall=weather_2024_avg["Rainfall"],
    ).fillna(0)
    return merged_data


# Step 2: Feature Engineering
def feature_engineering(
    laps: pd.DataFrame, qualifying_data: pd.DataFrame
) -> dict[str, Union[pd.DataFrame, pd.Series]]:
    """
    Perform feature engineering on the laps data and merge with qualifying data.
    Args:
        laps (pd.DataFrame): Laps data from FastF1.
        qualifying_data (pd.DataFrame): Qualifying data.
    Returns:
        dict: Dictionary containing features and target variable.
        - 'X': Feature set
        - 'y': Target variable (lap times)
    """
    # Map driver names to FastF1 3-letter codes
    qualifying_data["DriverCode"] = qualifying_data["Driver"].map(driver_mapping)

    sector_times = (
        laps[["Driver", "Sector1Time", "Sector2Time", "Sector3Time"]]
        .groupby("Driver")
        .mean()
        .reset_index()
    )

    lap_times = laps[["Driver", "LapTime"]].groupby("Driver").mean().reset_index()

    unmapped_drivers = qualifying_data["Driver"][qualifying_data["DriverCode"].isna()]
    print(f"Unmapped drivers: {unmapped_drivers}")

    assert (
        len(unmapped_drivers) == 0
    ), "Some drivers could not be mapped to FastF1 codes."

    # remove the drivers not in the driver mapping
    lap_times = lap_times[lap_times["Driver"].isin(driver_mapping.values())]

    # Merge qualifying data Driver in lap data and DriverCode in qualifying data
    merged_data = qualifying_data.merge(
        sector_times,
        left_on="DriverCode",
        right_on="Driver",
        how="left",
    )
    merged_data.drop(columns=["Driver_y"], inplace=True)
    merged_data.rename(columns={"Driver_x": "Driver"}, inplace=True)

    # merge laps data with merged data
    merged_data = merged_data.merge(
        lap_times,
        left_on="DriverCode",
        right_on="Driver",
        how="left",
    )
    merged_data.drop(columns=["Driver_y"], inplace=True)
    merged_data.rename(columns={"Driver_x": "Driver"}, inplace=True)

    # get average pit in and out times
    laps["PitInTime"] = laps["PitInTime"].apply(
        lambda x: x.total_seconds() if isinstance(x, pd.Timedelta) else x
    )
    laps["PitOutTime"] = laps["PitOutTime"].apply(
        lambda x: x.total_seconds() if isinstance(x, pd.Timedelta) else x
    )
    pit_in_time = laps.groupby("Driver")["PitInTime"].median().reset_index()
    pit_out_time = laps.groupby("Driver")["PitOutTime"].median().reset_index()
    merged_data = merged_data.merge(
        pit_in_time,
        left_on="DriverCode",
        right_on="Driver",
        how="left",
    )
    merged_data.rename(columns={"Driver_x": "Driver"}, inplace=True)
    merged_data.drop(columns=["Driver_y"], inplace=True)
    merged_data = merged_data.merge(
        pit_out_time,
        left_on="DriverCode",
        right_on="Driver",
        how="left",
    )
    merged_data.drop(columns=["Driver_y"], inplace=True)
    merged_data.rename(columns={"Driver_x": "Driver"}, inplace=True)

    # fill pit in and out times with 0
    merged_data["PitInTime"].fillna(0, inplace=True)
    merged_data["PitOutTime"].fillna(0, inplace=True)

    # 1.
    time_features = [
        "LapTime",
        "Sector1Time",
        "Sector2Time",
        "Sector3Time",
    ]
    # 2.
    tyre_features = [
        "TyreLife",
        "Compound",
    ]
    # 3.
    lap_features = [
        "PitInTime",
        "PitOutTime",
    ]
    # 4. weather features
    weather_features = [
        "AirTemp",
        "TrackTemp",
        "Humidity",
        "Pressure",
        "WindSpeed",
        "WindDirection",
        "Rainfall",
    ]

    # Converting lap times and sector times to numeric (total seconds)
    for col in [
        "Sector1Time",
        "Sector2Time",
        "Sector3Time",
        "LapTime",
        "PitInTime",
        "PitOutTime",
    ]:
        if isinstance(merged_data[col].iloc[0], pd.Timedelta):
            merged_data[col] = merged_data[col].dt.total_seconds()

    weather_data_avg = laps[weather_features].mean()
    merged_data = merged_data.assign(
        AirTemp=weather_data_avg["AirTemp"],
        TrackTemp=weather_data_avg["TrackTemp"],
        Humidity=weather_data_avg["Humidity"],
        Pressure=weather_data_avg["Pressure"],
        WindSpeed=weather_data_avg["WindSpeed"],
        WindDirection=weather_data_avg["WindDirection"],
        Rainfall=weather_data_avg["Rainfall"],
    )

    # add the lap_features to the merged data
    compunds_used = laps.groupby("Driver")["Compound"].value_counts().reset_index()
    # for each driver, get the most used compound
    compound_used = compunds_used.loc[
        compunds_used.groupby("Driver")["Compound"].idxmax()
    ]
    merged_data = merged_data.merge(
        compound_used[["Driver", "Compound"]],
        left_on="DriverCode",
        right_on="Driver",
        how="left",
    )
    merged_data.drop(columns=["Driver_y"], inplace=True)
    merged_data.rename(columns={"Driver_x": "Driver"}, inplace=True)

    merged_data["TyreLife"] = (
        laps.groupby("Driver")["TyreLife"].mean().reset_index()["TyreLife"]
    )

    # create new features
    merged_data["TotalSectorTime"] = (
        merged_data["Sector1Time"]
        + merged_data["Sector2Time"]
        + merged_data["Sector3Time"]
    )
    merged_data["Sector1Ratio"] = (
        merged_data["Sector1Time"] / merged_data["TotalSectorTime"]
    )
    merged_data["Sector2Ratio"] = (
        merged_data["Sector2Time"] / merged_data["TotalSectorTime"]
    )
    merged_data["Sector3Ratio"] = (
        merged_data["Sector3Time"] / merged_data["TotalSectorTime"]
    )

    # we need to ensure that "Laptime" which is the target variable has no missing values by using KNN to fill them
    # impute missing values
    imputer = KNNImputer(n_neighbors=3)
    merged_data[time_features] = imputer.fit_transform(merged_data[time_features])

    # Splitting features and target variable
    X = merged_data.drop(columns=["LapTime"])
    y = merged_data["LapTime"]

    assert X.shape[0] == len(y), "Mismatch between features and target variable."

    return {
        "X": X,
        "y": y,
    }


# Step 3: Train and Save Model
def train_model(X: pd.DataFrame, y: pd.Series, model_file: str):
    """
    Train a model using the provided features and target variable.
    Args:
        X (pd.DataFrame): Features for training.
        y (pd.Series): Target variable for training.
        model_file (str): Path to save the trained model.
    Returns:
        DataFrame: Feature importances.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", KNNImputer(n_neighbors=3)),
                        ("scaler", MinMaxScaler()),
                    ]
                ),
                X.select_dtypes(include=["int64", "float64"]).columns,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                ["Compound"],
            ),
        ],
    )
    # Create a pipeline with preprocessing and model
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", XGBRegressor(random_state=42)),
        ],
    )

    X_metadata = X[["DriverCode", "Driver"]].copy()  # Keep metadata for predictions
    X = X.drop(columns=["DriverCode", "Driver"])
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(X_train)

    # Train the model
    pipeline.fit(X_train, y_train)

    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")

    race_pred = pipeline.predict(X)

    print(race_pred)

    # show predictions with driver names
    predictions = pd.DataFrame(
        {
            "DriverCode": X_metadata["DriverCode"],
            "Driver": X_metadata["Driver"],
            "Predicted": race_pred,
        }
    )
    predictions = (
        predictions.groupby(["Driver"])[["Predicted", "DriverCode"]]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values(by="Predicted", ascending=True)
    )
    print(predictions)

    # draw table using plotly
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=[
            "🏁 Predicted 2025 Japanese GP Winner with New Drivers and Sector Times 🏁",
            "",
            "Feature Importances from Model",
        ],
        specs=[
            [{"type": "table", "rowspan": 2}],
            [{"type": "table"}],
            [{"type": "bar"}],
        ],
    )
    fig.add_trace(
        go.Table(
            header=dict(values=list(predictions.columns), align="left"),
            cells=dict(values=[predictions[col] for col in predictions.columns]),
        ),
        row=1,
        col=1,
    )

    # Cross-validation
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="neg_mean_absolute_error")
    print(f"Cross-validation MAE: {-cv_scores.mean()}")

    # Save the model
    dump(pipeline, model_file)
    print(f"Model saved to {model_file}")

    # return feature importances
    feature_importances = pipeline.named_steps["model"].feature_importances_
    feature_names = (
        pipeline.named_steps["preprocessor"]
        .transformers_[0][1]
        .get_feature_names_out()
        .tolist()
    )
    feature_names += (
        pipeline.named_steps["preprocessor"]
        .transformers_[1][1]
        .get_feature_names_out()
        .tolist()
    )
    feature_importances_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": feature_importances}
    ).sort_values(by="Importance", ascending=False)
    print("Feature Importances:")
    print(feature_importances_df)
    fig.add_trace(
        go.Bar(
            x=feature_importances_df["Importance"],
            y=feature_importances_df["Feature"],
            orientation="h",
            marker_color=feature_importances_df["Importance"],
        ),
        row=3,
        col=1,
    )
    fig.show()


# Step 4: Load Model for Fine-Tuning
def fine_tune_model(
    model_file: str, X_new: pd.DataFrame, y_new: pd.Series, new_model_file: str
):
    """
    Fine-tune an existing model with new data.
    Args:
        model_file (str): Path to the existing model.
        X_new (pd.DataFrame): New features for fine-tuning.
        y_new (pd.Series): New target variable for fine-tuning.
    """
    # Load existing model
    pipeline: Pipeline = load(model_file)

    # Fine-tune the model
    pipeline.fit(X_new, y_new)

    X_metadata = X_new[["DriverCode", "Driver"]].copy()  # Keep metadata for predictions

    # Evaluate the model
    y_pred = pipeline.predict(X_new)
    print(f"Fine-tuned Mean Absolute Error: {mean_absolute_error(y_new, y_pred)}")

    # show predictions with driver names
    predictions = pd.DataFrame(
        {
            "DriverCode": X_metadata["DriverCode"],
            "Driver": X_metadata["Driver"],
            "Predicted": y_pred,
        }
    )
    predictions = (
        predictions.groupby(["Driver"])[["Predicted", "DriverCode"]]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values(by="Predicted", ascending=True)
        .reset_index(drop=True)
    )
    print(predictions)

    # draw table using plotly
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=[
            "🏁 Fine-tuned Predictions with New Drivers and Sector Times 🏁",
            "",
            "Feature Importances from Model",
        ],
        specs=[
            [{"type": "table", "rowspan": 2}],
            [{"type": "table"}],
            [{"type": "bar"}],
        ],
    )
    fig.add_trace(
        go.Table(
            header=dict(values=list(predictions.columns), align="left"),
            cells=dict(values=[predictions[col] for col in predictions.columns]),
        ),
        row=1,
        col=1,
    )
    # Cross-validation
    cv_scores = cross_val_score(
        pipeline, X_new, y_new, cv=5, scoring="neg_mean_absolute_error"
    )
    print(f"Fine-tuned Cross-validation MAE: {-cv_scores.mean()}")
    # Feature importances
    feature_importances = pipeline.named_steps["model"].feature_importances_
    feature_names = (
        pipeline.named_steps["preprocessor"]
        .transformers_[0][1]
        .get_feature_names_out()
        .tolist()
    )
    feature_names += (
        pipeline.named_steps["preprocessor"]
        .transformers_[1][1]
        .get_feature_names_out()
        .tolist()
    )
    feature_importances_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": feature_importances}
    ).sort_values(by="Importance", ascending=False)
    print("Feature Importances:")
    print(feature_importances_df)
    fig.add_trace(
        go.Bar(
            x=feature_importances_df["Importance"],
            y=feature_importances_df["Feature"],
            orientation="h",
            marker_color=feature_importances_df["Importance"],
        ),
        row=3,
        col=1,
    )
    fig.show()

    # Save updated model
    dump(pipeline, new_model_file)
    print(f"Pipeline fine-tuned and saved to {new_model_file}")
