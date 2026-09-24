import pandas as pd
import mlflow
import mlflow.sklearn

from src.features.feature_engineering import engineer_features
from src.models.train import train_model
from src.models.evaluate import evaluate_model
from src.models.quality_gate import validate_model_quality


PROCESSED_DATA_PATH = "data/processed"

MLFLOW_EXPERIMENT_NAME = "loan_default_prediction"

REGISTERED_MODEL_NAME = "LoanDefaultModel"


def load_training_data():
    """
    Load training and validation datasets.
    """

    # ---------------------------------------------------------
    # Training data
    # ---------------------------------------------------------

    X_train = pd.read_csv(
        f"{PROCESSED_DATA_PATH}/X_train.csv"
    )

    y_train = pd.read_csv(
        f"{PROCESSED_DATA_PATH}/y_train.csv"
    ).squeeze()

    # ---------------------------------------------------------
    # Validation data
    # ---------------------------------------------------------

    X_val = pd.read_csv(
        f"{PROCESSED_DATA_PATH}/X_val.csv"
    )

    y_val = pd.read_csv(
        f"{PROCESSED_DATA_PATH}/y_val.csv"
    ).squeeze()

    return (
        X_train,
        y_train,
        X_val,
        y_val
    )


def run_training_pipeline():
    """
    Execute the complete training pipeline.

    Flow:
        Load data
        ↓
        Feature engineering
        ↓
        Model training
        ↓
        Validation prediction
        ↓
        Model evaluation
        ↓
        MLflow tracking
        ↓
        Model quality gate
        ↓
        MLflow model registry
    """

    print("=" * 60)
    print("STARTING TRAINING PIPELINE")
    print("=" * 60)

    # =========================================================
    # 1. Configure MLflow
    # =========================================================

    print("\n[1/8] Configuring MLflow...")

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT_NAME
    )

    with mlflow.start_run():

        print(
            "MLflow run started."
        )

        # =====================================================
        # 2. Load data
        # =====================================================

        print(
            "\n[2/8] Loading training and validation data..."
        )

        (
            X_train,
            y_train,
            X_val,
            y_val
        ) = load_training_data()

        print(
            f"X_train shape: {X_train.shape}"
        )

        print(
            f"y_train shape: {y_train.shape}"
        )

        print(
            f"X_val shape:   {X_val.shape}"
        )

        print(
            f"y_val shape:   {y_val.shape}"
        )

        # =====================================================
        # 3. Feature engineering
        # =====================================================

        print(
            "\n[3/8] Applying feature engineering..."
        )

        X_train_engineered = engineer_features(
            X_train
        )

        X_val_engineered = engineer_features(
            X_val
        )

        print(
            f"X_train after feature engineering: "
            f"{X_train_engineered.shape}"
        )

        print(
            f"X_val after feature engineering: "
            f"{X_val_engineered.shape}"
        )

        # =====================================================
        # 4. Train model
        # =====================================================

        print(
            "\n[4/8] Training Logistic Regression model..."
        )

        model_pipeline = train_model(
            X_train=X_train_engineered,
            y_train=y_train
        )

        print(
            "Model training completed."
        )

        # =====================================================
        # MLflow: Log parameters
        # =====================================================

        mlflow.log_params({
            "model_type": "LogisticRegression",
            "training_rows": X_train.shape[0],
            "validation_rows": X_val.shape[0],
            "training_features": X_train_engineered.shape[1],
            "validation_features": X_val_engineered.shape[1]
        })

        # =====================================================
        # 5. Generate validation predictions
        # =====================================================

        print(
            "\n[5/8] Generating validation predictions..."
        )

        validation_probabilities = (
            model_pipeline.predict_proba(
                X_val_engineered
            )[:, 1]
        )

        validation_predictions = (
            model_pipeline.predict(
                X_val_engineered
            )
        )

        print(
            "Validation probabilities shape:",
            validation_probabilities.shape
        )

        print(
            "Validation predictions shape:",
            validation_predictions.shape
        )

        # =====================================================
        # 6. Evaluate model
        # =====================================================

        print(
            "\n[6/8] Evaluating model..."
        )

        (
            metrics,
            confusion_matrix_result,
            classification_report_result
        ) = evaluate_model(
            y_true=y_val,
            y_probability=validation_probabilities,
            y_prediction=validation_predictions
        )

        print(
            "\n" + "=" * 60
        )

        print(
            "MODEL EVALUATION RESULTS"
        )

        print(
            "=" * 60
        )

        for metric_name, metric_value in metrics.items():

            print(
                f"{metric_name.upper():<12}: "
                f"{metric_value:.4f}"
            )

        print(
            "\nConfusion Matrix:"
        )

        print(
            confusion_matrix_result
        )

        print(
            "\nClassification Report:"
        )

        print(
            classification_report_result
        )

        # =====================================================
        # MLflow: Log metrics
        # =====================================================

        for metric_name, metric_value in metrics.items():

            mlflow.log_metric(
                metric_name,
                metric_value
            )

        # =====================================================
        # 7. Model Quality Gate
        # =====================================================

        print(
            "\n[7/8] Running model quality gate..."
        )

        model_passed = validate_model_quality(
            metrics
        )

        # =====================================================
        # Log quality gate result
        # =====================================================

        mlflow.log_param(
            "quality_gate",
            "PASSED" if model_passed else "FAILED"
        )

        # =====================================================
        # Stop if model fails quality gate
        # =====================================================

        if not model_passed:

            print(
                "\nTraining pipeline stopped."
            )

            print(
                "Model did not meet the required quality thresholds."
            )

            return (
                model_pipeline,
                metrics,
                confusion_matrix_result,
                classification_report_result,
                y_val
            )

        # =====================================================
        # 8. Register model
        # =====================================================

        print(
            "\n[8/8] Registering model in MLflow..."
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=model_pipeline,
            name="loan_default_model",
            skops_trusted_types=["numpy.dtype"],
            registered_model_name=REGISTERED_MODEL_NAME
        )

        print(
            "\nModel registered successfully."
        )

        print(
            f"Registered model: "
            f"{REGISTERED_MODEL_NAME}"
        )

        print(
            f"Model URI: "
            f"{model_info.model_uri}"
        )

        print(
            "\nTraining pipeline completed successfully."
        )

        return (
            model_pipeline,
            metrics,
            confusion_matrix_result,
            classification_report_result,
            y_val
        )


# =============================================================
# Run pipeline
# =============================================================

if __name__ == "__main__":

    run_training_pipeline()