import pandas as pd
import mlflow
import mlflow.sklearn

from src.features.feature_engineering import engineer_features
from src.features.feature_selection import select_features
from src.models.train import train_model
from src.models.evaluate import evaluate_model
from src.models.quality_gate import validate_model_quality


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROCESSED_DATA_PATH = "data/processed"

MLFLOW_EXPERIMENT_NAME = "loan_default_prediction"

REGISTERED_MODEL_NAME = "LoanDefaultModel"


# ---------------------------------------------------------
# Load training and validation data
# ---------------------------------------------------------

def load_training_data():
    """
    Load training and validation datasets.
    """

    X_train = pd.read_csv(
        f"{PROCESSED_DATA_PATH}/X_train.csv"
    )

    y_train = pd.read_csv(
        f"{PROCESSED_DATA_PATH}/y_train.csv"
    ).squeeze()

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


# ---------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------

def run_training_pipeline():

    print("=" * 60)
    print("STARTING TRAINING PIPELINE")
    print("=" * 60)

    # -----------------------------------------------------
    # Step 1: Configure MLflow
    # -----------------------------------------------------

    print("\n[1/9] Configuring MLflow...")

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT_NAME
    )

    with mlflow.start_run():

        print("MLflow run started.")

        # -------------------------------------------------
        # Step 2: Load data
        # -------------------------------------------------

        print(
            "\n[2/9] Loading training and validation data..."
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

        # -------------------------------------------------
        # Step 3: Feature Engineering
        # -------------------------------------------------

        print(
            "\n[3/9] Applying feature engineering..."
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

        # -------------------------------------------------
        # Step 4: Feature Selection
        # -------------------------------------------------

        print(
            "\n[4/9] Selecting model features..."
        )

        X_train_selected = select_features(
            X_train_engineered
        )

        X_val_selected = select_features(
            X_val_engineered
        )

        print(
            f"X_train after feature selection: "
            f"{X_train_selected.shape}"
        )

        print(
            f"X_val after feature selection: "
            f"{X_val_selected.shape}"
        )

        print(
            "\nSelected features:"
        )

        print(
            X_train_selected.columns.tolist()
        )

        # -------------------------------------------------
        # Step 5: Train Model
        # -------------------------------------------------

        print(
            "\n[5/9] Training Logistic Regression model..."
        )

        model_pipeline = train_model(
            X_train=X_train_selected,
            y_train=y_train
        )

        print(
            "Model training completed."
        )

        # -------------------------------------------------
        # MLflow Parameters
        # -------------------------------------------------

        mlflow.log_params({

            "model_type": "LogisticRegression",

            "training_rows": X_train.shape[0],

            "validation_rows": X_val.shape[0],

            "features_before_selection":
                X_train_engineered.shape[1],

            "features_after_selection":
                X_train_selected.shape[1],

            "selected_features":
                ", ".join(
                    X_train_selected.columns.tolist()
                )
        })

        # -------------------------------------------------
        # Step 6: Validation Predictions
        # -------------------------------------------------

        print(
            "\n[6/9] Generating validation predictions..."
        )

        validation_probabilities = (
            model_pipeline.predict_proba(
                X_val_selected
            )[:, 1]
        )

        validation_predictions = (
            model_pipeline.predict(
                X_val_selected
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

        # -------------------------------------------------
        # Step 7: Model Evaluation
        # -------------------------------------------------

        print(
            "\n[7/9] Evaluating model..."
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

        # -------------------------------------------------
        # Log Metrics to MLflow
        # -------------------------------------------------

        for metric_name, metric_value in metrics.items():

            mlflow.log_metric(
                metric_name,
                metric_value
            )

        # -------------------------------------------------
        # Step 8: Quality Gate
        # -------------------------------------------------

        print(
            "\n[8/9] Running model quality gate..."
        )

        model_passed = validate_model_quality(
            metrics
        )

        mlflow.log_param(
            "quality_gate",
            "PASSED" if model_passed else "FAILED"
        )

        # -------------------------------------------------
        # Stop if model fails
        # -------------------------------------------------

        if not model_passed:

            print(
                "\nTraining pipeline stopped."
            )

            print(
                "Model did not meet the required "
                "quality thresholds."
            )

            return (
                model_pipeline,
                metrics,
                confusion_matrix_result,
                classification_report_result,
                y_val
            )

        # -------------------------------------------------
        # Step 9: Register Model
        # -------------------------------------------------

        print(
            "\n[9/9] Registering model in MLflow..."
        )

        model_info = mlflow.sklearn.log_model(

            sk_model=model_pipeline,

            name="loan_default_model",

            registered_model_name=
                REGISTERED_MODEL_NAME,

            skops_trusted_types=["numpy.dtype"]
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


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    run_training_pipeline()