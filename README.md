# 🏎️ F1 Predictions 2025 - Machine Learning

Welcome to the **F1 Predictions 2025** repository! This project uses **machine learning, FastF1 API data, and historical F1 race results** to predict race outcomes for the 2025 Formula 1 season.

## 🚀 Project Overview
This repository contains a **Gradient Boosting Machine Learning model** that predicts race results based on past performance, qualifying times, and other structured F1 data. The model leverages:
- FastF1 API for historical race data
- 2024 race results as baseline training data
- 2025 qualifying session results for race predictions
- Progressive model fine-tuning across the season
- Advanced feature engineering including sector times, weather conditions and pit strategies

> **Note**: This project is a fork inspired by the original work found in the marantanya folder. My enhanced implementation is located in the mine folder.

## 📊 Data Sources
- **FastF1 API**: Comprehensive data including lap times, sector times, compound choices, and telemetry
- **2025 Qualifying Data**: Manually curated qualifying results
- **Weather Information**: Track and air temperature, humidity, wind speed and direction
- **Pit Strategy Data**: Pit in/out times and tire compound selection

## 🏁 How It Works
1. **Data Collection**: The script pulls relevant F1 data using the FastF1 API for each race.
2. **Preprocessing & Feature Engineering**: 
   - Converts lap times to seconds
   - Normalizes driver names using 3-letter codes
   - Calculates sector time ratios
   - Handles driver changes with mapping to team performance
3. **Progressive Model Training**: 
   - Initial model trained with Australian GP data
   - Fine-tuned with each subsequent race (Chinese GP, Japanese GP, etc.)
4. **Prediction**: The model predicts race times and ranks drivers accordingly.
5. **Evaluation & Visualization**: Performance measured using MAE with interactive Plotly visualizations.

## Dependencies
- `fastf1`: For accessing official F1 timing data
- `numpy` & `pandas`: For data manipulation
- `scikit-learn` & `xgboost`: For machine learning pipelines
- `plotly`: For interactive visualizations

## 📁 File Structure
- mine: Core project files with my enhanced implementation
  - utils.py: Core utility functions for data processing and modeling
  - chinese.py, japanese.py, etc.: Race-specific prediction scripts
  - feature_exploration.ipynb: Jupyter notebook for exploratory analysis
- marantanya: Original implementation that inspired this work
- models: Saved model files with progressive improvements
- f1_cache: FastF1 cache directory

## 🔧 Usage
Run prediction for a specific Grand Prix:
```bash
python3 mine/japanese.py
```

The output includes:
- Table of predicted race times by driver
- Feature importance visualization
- Model performance metrics

## 📈 Model Performance
- **Mean Absolute Error (MAE)**: Primary evaluation metric
- **Cross-validation**: 5-fold CV for robust performance estimation
- **Feature Importance Analysis**: Identifies key predictors of race performance

## 📌 Future Improvements
- Incorporate driver-specific historical performance trends
- Add detailed pit stop strategy modeling
- Develop a more sophisticated weather impact model
- Explore deep learning approaches for improved prediction accuracy

## 📜 License
This project is licensed under the MIT License.

🏎️ **Start predicting F1 races like a data scientist!** 🚀