# src/inference/predictor.py

from pathlib import Path

import pandas as pd
import mlflow.sklearn
from mlflow import MlflowClient

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

    client = MlflowClient()
    registered_model = client.get_registered_model(
        REGISTERED_MODEL_NAME
    )

    if not registered_model.latest_versions:
        raise FileNotFoundError(
            f"No versions found for registered model "
            f"'{REGISTERED_MODEL_NAME}'."
        )

    latest_version = max(
        registered_model.latest_versions,
        key=lambda version: int(version.version)
    )
    model_id = latest_version.source.rsplit("/", 1)[-1]
    artifact_paths = list(
        Path("mlruns").glob(
            f"*/models/{model_id}/artifacts"
        )
    )

    if not artifact_paths:
        raise FileNotFoundError(
            f"Artifacts for registered model version "
            f"{latest_version.version} ({model_id}) were not found "
            "under the local mlruns directory."
        )

    model = mlflow.sklearn.load_model(
        str(artifact_paths[0])
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