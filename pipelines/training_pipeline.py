# src/training_pipeline.py

import pandas as pd

import mlflow
import mlflow.sklearn

from src.features.feature_engineering import (
    engineer_features
)

from src.features.feature_selection import (
    select_features
)

from src.models.train import (
    train_model
)

from src.models.evaluate import (
    evaluate_model
)

from src.models.quality_gate import (
    validate_model_quality
)


# ============================================================
# CONFIGURATION
# ============================================================

PROCESSED_DATA_PATH = "data/processed"

MLFLOW_EXPERIMENT_NAME = (
    "loan_default_prediction"
)

REGISTERED_MODEL_NAME = (
    "LoanDefaultModel"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_training_data():
    """
    Load training, validation and test datasets.
    """

    # --------------------------------------------------------
    # Training data
    # --------------------------------------------------------

    X_train = pd.read_csv(
        f"{PROCESSED_DATA_PATH}/X_train.csv"
    )

    y_train = pd.read_csv(
        f"{PROCESSED_DATA_PATH}/y_train.csv"
    ).squeeze()

    # --------------------------------------------------------
    # Validation data
    # --------------------------------------------------------

    X_val = pd.read_csv(
        f"{PROCESSED_DATA_PATH}/X_val.csv"
    )

    y_val = pd.read_csv(
        f"{PROCESSED_DATA_PATH}/y_val.csv"
    ).squeeze()

    # --------------------------------------------------------
    # Test data
    # --------------------------------------------------------

    X_test = pd.read_csv(
        f"{PROCESSED_DATA_PATH}/X_test.csv"
    )

    y_test = pd.read_csv(
        f"{PROCESSED_DATA_PATH}/y_test.csv"
    ).squeeze()

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    )


# ============================================================
# TRAINING PIPELINE
# ============================================================

def run_training_pipeline():

    print("=" * 60)
    print("STARTING TRAINING PIPELINE")
    print("=" * 60)


    # ========================================================
    # STEP 1 — CONFIGURE MLFLOW
    # ========================================================

    print(
        "\n[1/11] Configuring MLflow..."
    )

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT_NAME
    )


    with mlflow.start_run():

        print(
            "MLflow run started."
        )


        # ====================================================
        # STEP 2 — LOAD DATA
        # ====================================================

        print(
            "\n[2/11] Loading train, validation and test data..."
        )

        (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test
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

        print(
            f"X_test shape:  {X_test.shape}"
        )

        print(
            f"y_test shape:  {y_test.shape}"
        )


        # ====================================================
        # STEP 3 — FEATURE ENGINEERING
        # ====================================================

        print(
            "\n[3/11] Applying feature engineering..."
        )

        X_train_engineered = (
            engineer_features(
                X_train
            )
        )

        X_val_engineered = (
            engineer_features(
                X_val
            )
        )

        X_test_engineered = (
            engineer_features(
                X_test
            )
        )


        print(
            "X_train after feature engineering:",
            X_train_engineered.shape
        )

        print(
            "X_val after feature engineering:",
            X_val_engineered.shape
        )

        print(
            "X_test after feature engineering:",
            X_test_engineered.shape
        )


        # ====================================================
        # STEP 4 — FEATURE SELECTION
        # ====================================================

        print(
            "\n[4/11] Selecting model features..."
        )

        X_train_selected = (
            select_features(
                X_train_engineered
            )
        )

        X_val_selected = (
            select_features(
                X_val_engineered
            )
        )

        X_test_selected = (
            select_features(
                X_test_engineered
            )
        )


        print(
            "X_train after feature selection:",
            X_train_selected.shape
        )

        print(
            "X_val after feature selection:",
            X_val_selected.shape
        )

        print(
            "X_test after feature selection:",
            X_test_selected.shape
        )


        print(
            "\nSelected features:"
        )

        print(
            X_train_selected.columns.tolist()
        )


        # ====================================================
        # STEP 5 — TRAIN MODEL
        # ====================================================

        print(
            "\n[5/11] Training Logistic Regression model..."
        )

        model_pipeline = train_model(
            X_train=X_train_selected,
            y_train=y_train
        )


        print(
            "Model training completed."
        )


        # ====================================================
        # MLFLOW — LOG MODEL PARAMETERS
        # ====================================================

        mlflow.log_params({

            "model_type":
                "LogisticRegression",

            "training_rows":
                X_train.shape[0],

            "validation_rows":
                X_val.shape[0],

            "test_rows":
                X_test.shape[0],

            "features_before_selection":
                X_train_engineered.shape[1],

            "features_after_selection":
                X_train_selected.shape[1],

            "selected_features":
                ",".join(
                    X_train_selected.columns.tolist()
                )
        })


        # ====================================================
        # STEP 6 — TRAINING DATA INFERENCE
        # ====================================================

        print(
            "\n[6/11] Generating training predictions..."
        )

        train_probabilities = (
            model_pipeline.predict_proba(
                X_train_selected
            )[:, 1]
        )

        train_predictions = (
            model_pipeline.predict(
                X_train_selected
            )
        )


        # ====================================================
        # TRAINING DATA EVALUATION
        # ====================================================

        print(
            "\nEvaluating training performance..."
        )

        (
            train_metrics,
            train_confusion_matrix,
            train_classification_report
        ) = evaluate_model(

            y_true=y_train,

            y_probability=
                train_probabilities,

            y_prediction=
                train_predictions
        )


        print(
            "\n" + "=" * 60
        )

        print(
            "TRAINING DATA PERFORMANCE"
        )

        print(
            "=" * 60
        )


        for (
            metric_name,
            metric_value
        ) in train_metrics.items():

            print(
                f"{metric_name.upper():<12}: "
                f"{metric_value:.4f}"
            )


        print(
            "\nTraining Confusion Matrix:"
        )

        print(
            train_confusion_matrix
        )


        print(
            "\nTraining Classification Report:"
        )

        print(
            train_classification_report
        )


        # ----------------------------------------------------
        # Log training metrics
        # ----------------------------------------------------

        for (
            metric_name,
            metric_value
        ) in train_metrics.items():

            mlflow.log_metric(
                f"train_{metric_name}",
                metric_value
            )


        # ====================================================
        # STEP 7 — VALIDATION DATA INFERENCE
        # ====================================================

        print(
            "\n[7/11] Generating validation predictions..."
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


        # ====================================================
        # VALIDATION DATA EVALUATION
        # ====================================================

        print(
            "\nEvaluating validation performance..."
        )

        (
            validation_metrics,
            validation_confusion_matrix,
            validation_classification_report
        ) = evaluate_model(

            y_true=y_val,

            y_probability=
                validation_probabilities,

            y_prediction=
                validation_predictions
        )


        print(
            "\n" + "=" * 60
        )

        print(
            "VALIDATION DATA PERFORMANCE"
        )

        print(
            "=" * 60
        )


        for (
            metric_name,
            metric_value
        ) in validation_metrics.items():

            print(
                f"{metric_name.upper():<12}: "
                f"{metric_value:.4f}"
            )


        print(
            "\nValidation Confusion Matrix:"
        )

        print(
            validation_confusion_matrix
        )


        print(
            "\nValidation Classification Report:"
        )

        print(
            validation_classification_report
        )


        # ----------------------------------------------------
        # Log validation metrics
        # ----------------------------------------------------

        for (
            metric_name,
            metric_value
        ) in validation_metrics.items():

            mlflow.log_metric(
                f"validation_{metric_name}",
                metric_value
            )


        # ====================================================
        # STEP 8 — QUALITY GATE
        # ====================================================

        print(
            "\n[8/11] Running model quality gate..."
        )

        # IMPORTANT:
        #
        # Quality gate uses VALIDATION metrics.
        #
        # Training metrics are used for diagnosing
        # overfitting.
        #
        # Test metrics are used only for final
        # unbiased evaluation.

        model_passed = (
            validate_model_quality(
                validation_metrics
            )
        )


        mlflow.log_param(
            "quality_gate",
            "PASSED"
            if model_passed
            else "FAILED"
        )


        # ====================================================
        # STOP IF QUALITY GATE FAILS
        # ====================================================

        if not model_passed:

            print(
                "\nTraining pipeline stopped."
            )

            print(
                "Model did not meet the required "
                "validation quality thresholds."
            )

            return (
                model_pipeline,
                train_metrics,
                validation_metrics,
                None,
                train_confusion_matrix,
                validation_confusion_matrix,
                None,
                train_classification_report,
                validation_classification_report,
                None,
                y_test
            )


        # ====================================================
        # STEP 9 — TEST DATA INFERENCE
        # ====================================================

        print(
            "\n[9/11] Generating test predictions..."
        )

        test_probabilities = (
            model_pipeline.predict_proba(
                X_test_selected
            )[:, 1]
        )

        test_predictions = (
            model_pipeline.predict(
                X_test_selected
            )
        )


        print(
            "Test probabilities shape:",
            test_probabilities.shape
        )

        print(
            "Test predictions shape:",
            test_predictions.shape
        )


        # ====================================================
        # STEP 10 — TEST DATA EVALUATION
        # ====================================================

        print(
            "\n[10/11] Evaluating final test performance..."
        )

        (
            test_metrics,
            test_confusion_matrix,
            test_classification_report
        ) = evaluate_model(

            y_true=y_test,

            y_probability=
                test_probabilities,

            y_prediction=
                test_predictions
        )


        print(
            "\n" + "=" * 60
        )

        print(
            "FINAL TEST DATA PERFORMANCE"
        )

        print(
            "=" * 60
        )


        for (
            metric_name,
            metric_value
        ) in test_metrics.items():

            print(
                f"{metric_name.upper():<12}: "
                f"{metric_value:.4f}"
            )


        print(
            "\nTest Confusion Matrix:"
        )

        print(
            test_confusion_matrix
        )


        print(
            "\nTest Classification Report:"
        )

        print(
            test_classification_report
        )


        # ----------------------------------------------------
        # Log test metrics to MLflow
        # ----------------------------------------------------

        for (
            metric_name,
            metric_value
        ) in test_metrics.items():

            mlflow.log_metric(
                f"test_{metric_name}",
                metric_value
            )


        # ====================================================
        # STEP 11 — REGISTER MODEL
        # ====================================================

        print(
            "\n[11/11] Registering model in MLflow..."
        )


        model_info = (
            mlflow.sklearn.log_model(

                sk_model=
                    model_pipeline,

                name=
                    "loan_default_model",

                registered_model_name=
                    REGISTERED_MODEL_NAME,

                skops_trusted_types=["numpy.dtype"]
            )
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
            train_metrics,
            validation_metrics,
            test_metrics,
            train_confusion_matrix,
            validation_confusion_matrix,
            test_confusion_matrix,
            train_classification_report,
            validation_classification_report,
            test_classification_report,
            y_test
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_training_pipeline()