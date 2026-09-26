# src/inference/predictor.py

import pandas as pd
import mlflow.sklearn

from src.features.feature_engineering import (
    engineer_features
)

from src.features.feature_selection import (
    select_features
)


REGISTERED_MODEL_NAME = "LoanDefaultModel"


def load_model():
    """
    Load the latest registered model
    from MLflow Model Registry.
    """

    model_uri = (
        f"models:/{REGISTERED_MODEL_NAME}/latest"
    )

    model = mlflow.sklearn.load_model(
        model_uri
    )

    print(
        "Model loaded successfully."
    )

    return model


def prepare_input(
    input_data: pd.DataFrame
):
    """
    Apply the same feature engineering
    and feature selection used during training.
    """

    # Feature engineering
    X_engineered = engineer_features(
        input_data
    )

    # Feature selection
    X_selected = select_features(
        X_engineered
    )

    return X_selected


def predict(
    input_data: pd.DataFrame
):
    """
    Generate default probability
    and final prediction.
    """

    # Load trained model
    model = load_model()

    # Prepare input
    X_selected = prepare_input(
        input_data
    )

    # Generate probability
    probabilities = (
        model.predict_proba(
            X_selected
        )[:, 1]
    )

    # Convert probability to class
    predictions = (
        probabilities >= 0.5
    ).astype(int)

    result = pd.DataFrame({

        "default_probability":
            probabilities,

        "prediction":
            predictions

    })

    return result