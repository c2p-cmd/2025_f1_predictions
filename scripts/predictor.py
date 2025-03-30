import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import fastf1
import plotly.express as px
import plotly.graph_objects as go


class F1Predictor:
    def __init__(self, cache_path="../f1_cache"):
        """Initialize the F1 prediction framework with a cache path."""
        # Setup cache
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        fastf1.Cache.enable_cache(cache_path)

        # Default model parameters
        self.model_params = {
            "n_estimators": 150,
            "learning_rate": 0.1,
            "max_depth": 3,
            "random_state": 42,
        }

        # Default feature columns
        self.feature_columns = [
            "Time",
            "Driver",
            "LapTime",
            "Sector1Time",
            "Sector2Time",
            "Sector3Time",
            "Compound",
            "TrackStatus",
        ]

        self.numeric_columns = [
            "QualifyingTime",
            "Sector1Time",
            "Sector2Time",
            "Sector3Time",
            "AirTemp",
            "Humidity",
            "WindSpeed",
        ]

        self.categorical_columns = ["Driver", "Compound", "TrackStatus", "Rainfall"]

        self.timing_columns = ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]

        self.model = None
        self.laps_combined = None
        self.feature_importances = None

    def load_race_data(self, years, race_name, session_type="R"):
        """
        Load data for specific race across multiple years.

        Args:
            years (list): List of years to load data for
            race_name (str): Name of the race (e.g., 'Australian GP')
            session_type (str): Session type ('R' for race, 'Q' for qualifying)

        Returns:
            dict: Dictionary of loaded sessions by year
        """
        sessions = {}
        laps_data = {}
        weather_data = {}

        for year in years:
            try:
                print(f"Loading {race_name} {year} {session_type}...")
                session = fastf1.get_session(year, race_name, session_type)
                session.load()
                sessions[year] = session

                # Extract laps data
                try:
                    laps = session.laps.copy()
                    
                    # Filter to only keep the columns we need
                    available_columns = [col for col in self.feature_columns if col in laps.columns]
                    missing_columns = [col for col in self.feature_columns if col not in laps.columns]
                    
                    if missing_columns:
                        print(f"Warning: Missing columns for {year}: {missing_columns}")
                    
                    laps = laps[available_columns]
                    laps["Year"] = year
                except Exception as e:
                    print(f"Error processing laps data for {year}: {e}")
                    continue

                # Extract weather data
                try:
                    weather = session.weather_data
                    avg_temp = weather["AirTemp"].mean() if "AirTemp" in weather.columns else None
                    avg_humidity = weather["Humidity"].mean() if "Humidity" in weather.columns else None
                    avg_wind_speed = weather["WindSpeed"].mean() if "WindSpeed" in weather.columns else None
                    avg_rain = weather["Rainfall"].mean() if "Rainfall" in weather.columns else 0
                    
                    # Add weather data to laps
                    laps["AirTemp"] = avg_temp
                    laps["Humidity"] = avg_humidity
                    laps["WindSpeed"] = avg_wind_speed
                    laps["Rainfall"] = avg_rain
                    
                    weather_data[year] = {
                        "AirTemp": avg_temp,
                        "Humidity": avg_humidity,
                        "WindSpeed": avg_wind_speed,
                        "Rainfall": avg_rain,
                    }
                except Exception as e:
                    print(f"Error processing weather data for {year}: {e}")
                    
                    # Set default weather values
                    laps["AirTemp"] = np.nan
                    laps["Humidity"] = np.nan
                    laps["WindSpeed"] = np.nan
                    laps["Rainfall"] = 0

                # Convert timing columns to seconds
                for col in self.timing_columns:
                    if col in laps.columns and pd.api.types.is_datetime64_dtype(laps[col]):
                        laps[col] = laps[col].dt.total_seconds()
                    elif col in laps.columns and isinstance(laps[col].iloc[0], str):
                        # Handle string format timing if present
                        try:
                            laps[col] = pd.to_timedelta(laps[col]).dt.total_seconds()
                        except:
                            print(f"Warning: Could not convert {col} to seconds for {year}")
                
                # Remove rows with NaN in timing columns
                laps = laps.dropna(subset=[col for col in self.timing_columns if col in laps.columns])
                
                if len(laps) > 0:
                    laps_data[year] = laps
                else:
                    print(f"Warning: No valid lap data for {year} after cleaning")
                
            except Exception as e:
                print(f"Error loading {year} data: {e}")

        # Combine all laps data
        if laps_data:
            try:
                self.laps_combined = pd.concat(list(laps_data.values()), ignore_index=True)
                print(f"Successfully combined data from {len(laps_data)} years with {len(self.laps_combined)} total laps")
            except Exception as e:
                print(f"Error combining laps data: {e}")
                self.laps_combined = pd.DataFrame()
        else:
            print("No valid laps data to combine")
            self.laps_combined = pd.DataFrame()

        return {
            "sessions": sessions,
            "laps_data": laps_data,
            "weather_data": weather_data,
        }

    def add_qualifying_data(self, qualifying_data, driver_mapping):
        """
        Add qualifying data to the combined laps data.

        Args:
            qualifying_data (pd.DataFrame): DataFrame with qualifying data
            driver_mapping (dict): Mapping from driver names to driver codes

        Returns:
            pd.DataFrame: Merged data with qualifying times
        """
        if self.laps_combined is None or self.laps_combined.empty:
            print("No lap data available. Please load race data first.")
            return None

        try:
            # Map driver names to codes if needed
            if "Driver" not in qualifying_data.columns and "DriverName" in qualifying_data.columns:
                qualifying_data["Driver"] = qualifying_data["DriverName"].map(driver_mapping)
                
            if "Driver" not in qualifying_data.columns:
                print("Error: No 'Driver' column in qualifying data")
                return None
                
            # Check if Driver column exists in laps_combined
            if "Driver" not in self.laps_combined.columns:
                print("Error: No 'Driver' column in lap data")
                return None
                
            # Print for debugging
            print(f"Qualifying data drivers: {qualifying_data['Driver'].unique()}")
            print(f"Lap data drivers: {self.laps_combined['Driver'].unique()}")
            
            # Merge data
            self.merged_data = self.laps_combined.merge(
                qualifying_data[["Driver", "QualifyingTime"]], on="Driver", how="left"
            )
            
            # Report merge results
            pre_count = len(self.laps_combined)
            post_count = len(self.merged_data)
            print(f"Pre-merge: {pre_count} rows, Post-merge: {post_count} rows")
            print(f"Rows with QualifyingTime: {self.merged_data['QualifyingTime'].notna().sum()}")

            # Clean up data - only drop if specified columns have NaN
            cols_to_check = ["QualifyingTime"] + [col for col in self.numeric_columns if col != "QualifyingTime" and col in self.merged_data.columns]
            self.merged_data = self.merged_data.dropna(subset=cols_to_check)
            print(f"After dropping NaNs: {len(self.merged_data)} rows")
            
            return self.merged_data
            
        except Exception as e:
            print(f"Error adding qualifying data: {e}")
            import traceback
            traceback.print_exc()
            return None

    def prepare_data_for_modeling(self):
        """Prepare data for modeling by encoding categorical features."""
        if not hasattr(self, 'merged_data') or self.merged_data is None or self.merged_data.empty:
            print("No merged data available. Please add qualifying data first.")
            return None, None
        
        # Make a copy to avoid modifying the original
        modeling_data = self.merged_data.copy()
        
        # Verify columns exist
        available_numeric = [col for col in self.numeric_columns if col in modeling_data.columns]
        available_categorical = [col for col in self.categorical_columns if col in modeling_data.columns]
        
        missing_numeric = [col for col in self.numeric_columns if col not in modeling_data.columns]
        missing_categorical = [col for col in self.categorical_columns if col not in modeling_data.columns]
        
        if missing_numeric or missing_categorical:
            print(f"Warning: Missing numeric columns: {missing_numeric}")
            print(f"Warning: Missing categorical columns: {missing_categorical}")
        
        # Encode categorical features
        for col in available_categorical:
            modeling_data[col] = pd.Categorical(modeling_data[col]).codes
        
        # Select features and target
        if "LapTime" not in modeling_data.columns:
            print("Error: No 'LapTime' column in data for target variable")
            return None, None
            
        X = modeling_data[available_numeric + available_categorical]
        y = modeling_data["LapTime"]
        
        print(f"Prepared data for modeling with {X.shape[1]} features and {X.shape[0]} examples")
        return X, y

    def train_model(self, X=None, y=None, test_size=0.2, custom_params=None):
        """
        Train the prediction model.

        Args:
            X (pd.DataFrame, optional): Features dataframe
            y (pd.Series, optional): Target variable
            test_size (float): Size of test split
            custom_params (dict, optional): Custom model parameters

        Returns:
            tuple: (model, evaluation metrics)
        """
        if X is None or y is None:
            X, y = self.prepare_data_for_modeling()
            
        if X is None or y is None:
            print("Failed to prepare data for modeling")
            return None, {"error": "Failed to prepare data"}

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            print(f"Training set: {X_train.shape[0]} examples")
            print(f"Test set: {X_test.shape[0]} examples")

            model_params = custom_params if custom_params else self.model_params
            model = GradientBoostingRegressor(**model_params)
            
            # Check for and handle any infinite or NaN values
            X_train = np.nan_to_num(X_train)
            y_train = np.nan_to_num(y_train)
            
            model.fit(X_train, y_train)

            # Evaluate model
            X_test = np.nan_to_num(X_test)
            y_test = np.nan_to_num(y_test)
            
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)

            # Save model and evaluation data
            self.model = model
            self.X_test = X_test
            self.y_test = y_test
            self.y_pred = y_pred
            self.mae = mae
            self.feature_names = list(X.columns)
            self.feature_importances = model.feature_importances_

            print(f"Model trained with Mean Absolute Error: {mae:.2f} seconds")
            
            # Return more detailed metrics
            return model, {
                "mae": mae, 
                "X_test": X_test, 
                "y_test": y_test, 
                "y_pred": y_pred,
                "feature_names": self.feature_names,
                "feature_importances": self.feature_importances
            }
            
        except Exception as e:
            print(f"Error training model: {e}")
            import traceback
            traceback.print_exc()
            return None, {"error": str(e)}

    def get_feature_importance(self):
        """Get feature importance from the trained model."""
        if self.model is None or not hasattr(self, 'feature_names') or not hasattr(self, 'feature_importances'):
            print("Model not trained yet or feature data is missing.")
            return None

        try:
            feature_importance_df = pd.DataFrame({
                "Feature": self.feature_names,
                "Importance": self.feature_importances
            }).sort_values(by="Importance", ascending=False)
            
            return feature_importance_df
        except Exception as e:
            print(f"Error getting feature importance: {e}")
            return None

    def plot_feature_importance(self):
        """Plot feature importance."""
        feature_importance_df = self.get_feature_importance()
        if feature_importance_df is None:
            return None

        try:
            fig = px.bar(
                feature_importance_df,
                x="Importance",
                y="Feature",
                title="Feature Importance",
                orientation="h",
                text_auto=True,
                color="Importance",
                color_continuous_scale=px.colors.sequential.Viridis,
            )
            fig.update_layout(
                xaxis_title="Importance",
                yaxis_title="Feature",
                title_x=0.5,
            )
            return fig
        except Exception as e:
            print(f"Error plotting feature importance: {e}")
            return None

    def plot_predictions_vs_actual(self):
        """Plot predicted vs actual lap times."""
        if self.model is None or not hasattr(self, 'y_test') or not hasattr(self, 'y_pred'):
            print("Model not trained yet or prediction data is missing.")
            return None

        try:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=self.y_test,
                    y=self.y_pred,
                    mode="markers",
                    name="Predicted vs Actual",
                    marker=dict(color="blue", size=5),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[min(self.y_test), max(self.y_test)],
                    y=[min(self.y_test), max(self.y_test)],
                    mode="lines",
                    name="Perfect Prediction",
                    line=dict(color="red", dash="dash"),
                )
            )
            fig.update_layout(
                title="Actual vs Predicted Lap Times",
                xaxis_title="Actual Lap Time (seconds)",
                yaxis_title="Predicted Lap Time (seconds)",
                showlegend=True,
            )
            return fig
        except Exception as e:
            print(f"Error plotting predictions vs actual: {e}")
            return None

    def make_predictions(self, new_data):
        """
        Make predictions on new data.

        Args:
            new_data (pd.DataFrame): New data for prediction

        Returns:
            pd.DataFrame: Data with predictions
        """
        if self.model is None:
            print("Model not trained yet. Please train the model first.")
            return None
            
        if not hasattr(self, 'feature_names'):
            print("Model feature names not saved. Retraining is needed.")
            return None

        try:
            # Make a copy to avoid modifying the original data
            prediction_data = new_data.copy()
            
            # Check for required columns
            missing_columns = [col for col in self.feature_names if col not in prediction_data.columns]
            if missing_columns:
                print(f"Missing columns in prediction data: {missing_columns}")
                print("Available columns:", prediction_data.columns.tolist())
                return None
            
            # Extract only the needed columns in the right order
            X_new = prediction_data[self.feature_names].copy()
            
            # Convert categorical columns - using same pattern as in training
            for col in self.categorical_columns:
                if col in X_new.columns:
                    if X_new[col].dtype == "object":
                        X_new[col] = pd.Categorical(X_new[col]).codes

            # Handle NaN values
            X_new = X_new.dropna()
            
            if X_new.empty:
                print("No valid rows for prediction after removing NaN values")
                return None
                
            # Convert any remaining NaNs to zeros for prediction
            X_new = np.nan_to_num(X_new)
            
            # Make predictions
            predictions = self.model.predict(X_new)
            prediction_data.loc[X_new.index, "PredictedLapTime"] = predictions
            
            print(f"Made predictions for {len(X_new)} examples")
            return prediction_data
            
        except Exception as e:
            print(f"Error making predictions: {e}")
            import traceback
            traceback.print_exc()
            return None

    def plot_driver_predictions(self, predictions_df):
        """
        Plot driver predictions.

        Args:
            predictions_df (pd.DataFrame): DataFrame with predictions

        Returns:
            plotly.graph_objects.Figure: Plot of predictions
        """
        if predictions_df is None or predictions_df.empty:
            print("No prediction data provided")
            return None
            
        if "PredictedLapTime" not in predictions_df.columns:
            print("No predicted lap times in data")
            return None
            
        try:
            # Check if actual lap times are available
            has_actual = "LapTime" in predictions_df.columns and predictions_df["LapTime"].notna().any()
            
            if has_actual:
                # Group by driver and calculate mean lap times
                driver_predictions = (
                    predictions_df.groupby("Driver")[["LapTime", "PredictedLapTime"]]
                    .mean()
                    .sort_values(by="PredictedLapTime", ascending=True)
                    .reset_index()
                )
                
                # Create grouped bar chart
                fig = px.bar(
                    driver_predictions,
                    x="Driver",
                    y=["LapTime", "PredictedLapTime"],
                    title="Predicted vs Actual Lap Times by Driver",
                    barmode="group",
                    text_auto=".2f",
                    color_discrete_map={"LapTime": "blue", "PredictedLapTime": "red"},
                )
            else:
                # Only use predicted lap times
                driver_predictions = (
                    predictions_df.groupby("Driver")[["PredictedLapTime"]]
                    .mean()
                    .sort_values(by="PredictedLapTime", ascending=True)
                    .reset_index()
                )
                
                # Create simple bar chart
                fig = px.bar(
                    driver_predictions,
                    x="Driver",
                    y="PredictedLapTime",
                    title="Predicted Lap Times by Driver",
                    text_auto=".2f",
                    color="PredictedLapTime",
                    color_continuous_scale=px.colors.sequential.Viridis_r,
                )
            
            # Update layout
            fig.update_layout(
                xaxis_title="Driver",
                yaxis_title="Lap Time (seconds)",
                title_x=0.5,
                legend_title_text="Lap Time Type" if has_actual else None,
            )
            
            # Add a line for the mean predicted lap time
            mean_predicted = driver_predictions["PredictedLapTime"].mean()
            fig.add_hline(
                y=mean_predicted,
                line_dash="dash",
                line_color="green",
                annotation_text=f"Mean: {mean_predicted:.2f}s",
            )
            
            return fig
            
        except Exception as e:
            print(f"Error plotting driver predictions: {e}")
            import traceback
            traceback.print_exc()
            return None