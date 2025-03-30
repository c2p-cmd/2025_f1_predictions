import pandas as pd
from predictor import F1Predictor
from helpers import driver_mapping, load_qualifying_data


def predict_australian_gp() -> None:
    # Initialize predictor
    predictor = F1Predictor(cache_path="f1_cache")

    # Load race data for Australian GP
    race_data = predictor.load_race_data(
        years=[2023, 2024], race_name="Australia", session_type="R"
    )

    # Add qualifying data for 2025
    qualifying_2025 = load_qualifying_data(
        driver_names=[
            "Lando Norris",
            "Oscar Piastri",
            "Max Verstappen",
            "George Russell",
            "Yuki Tsunoda",
            "Alexander Albon",
            "Charles Leclerc",
            "Lewis Hamilton",
            "Pierre Gasly",
            "Carlos Sainz",
            "Fernando Alonso",
            "Lance Stroll",
        ],
        qualifying_times=[
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
    )

    # Add qualifying data to laps data
    merged_data = predictor.add_qualifying_data(qualifying_2025, driver_mapping)

    # Train model
    X, y = predictor.prepare_data_for_modeling()
    model, metrics = predictor.train_model(X, y)
    print("Model training complete.")
    print(f"Model metrics: {metrics}")

    # Show feature importance
    fig_importance = predictor.plot_feature_importance()
    fig_importance.show()

    # Show prediction accuracy
    fig_accuracy = predictor.plot_predictions_vs_actual()
    fig_accuracy.show()

    # Make predictions for year
    try:
        australian_gp_2025 = predictor.load_race_data(
            years=[2025],
            race_name="Australia",
            session_type="Q",
        )

        laps_2025 = australian_gp_2025["laps_data"].get(2025)
        if laps_2025 is not None:
            predictions = predictor.make_predictions(laps_2025)

            # Plot driver predictions
            fig_predictions = predictor.plot_driver_predictions(predictions)
            fig_predictions.show()
    except Exception as e:
        print(f"Could not make predictions for 2025: {e}")


if __name__ == "__main__":
    # argument parser
    import argparse

    parser = argparse.ArgumentParser(description="Predict GP results")
    options = [
        "Australian GP",
        "Chinese GP",
        "Japanese GP",
    ]
    # add options to parser
    parser.add_argument(
        "--gp",
        type=str,
        choices=options,
        default="Australian GP",
        help="Choose the GP to predict",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2025,
        help="Year to predict",
    )
    parser.add_argument(
        "--history_years",
        type=int,
        nargs="+",
        default=[2023, 2024],
        help="Years to use for training",
    )
    args = parser.parse_args()
    # run the prediction
    if args.gp and args.year:
        if args.gp not in options:
            print(f"Invalid GP option. Available options: {options}")
            exit(1)
        if args.year < 2025:
            print("Year must be 2025 or later.")
            exit(1)
        if args.history_years and len(args.history_years) < 2:
            print("At least two years of history are required.")
            exit(1)
        # Call the prediction function
        print(
            f"Predicting {args.gp} for {args.year} using history years {args.history_years}"
        )
        print(f"Predicting {args.gp} for {args.year}")
        if args.gp == "Australian GP":
            predict_australian_gp()
        elif args.gp == "Chinese GP":
            print("Chinese GP prediction not implemented yet.")
        elif args.gp == "Japanese GP":
            print("Japanese GP prediction not implemented yet.")
    else:
        print("Please provide a GP and year to predict.")
