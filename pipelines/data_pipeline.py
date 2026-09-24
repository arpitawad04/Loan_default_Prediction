from pathlib import Path
import json

from src.data.ingestion import load_data
from src.data.validation import validate_data
from src.data.spllitting import split_data


RAW_DATA_PATH = "data/raw/credit_score_data.csv"
PROCESSED_DATA_PATH = Path("data/processed")

TARGET_COLUMN = "DEFAULT"
ID_COLUMN = "CUST_ID"


def run_data_pipeline():

    print("=" * 60)
    print("STARTING DATA PIPELINE")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Create processed-data directory
    # ---------------------------------------------------------

    PROCESSED_DATA_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # 2. Ingestion
    # ---------------------------------------------------------

    print("\n[1/3] Loading data...")

    df = load_data(RAW_DATA_PATH)

    print(f"Dataset shape: {df.shape}")

    # ---------------------------------------------------------
    # 3. Validation
    # ---------------------------------------------------------

    print("\n[2/3] Validating data...")

    validate_data(df)

    # ---------------------------------------------------------
    # 4. Train / Validation / Test split
    # ---------------------------------------------------------

    print("\n[3/3] Splitting data...")

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = split_data(
        df=df,
        target_column=TARGET_COLUMN
    )

    # ---------------------------------------------------------
    # 5. Save datasets
    # ---------------------------------------------------------

    X_train.to_csv(
        PROCESSED_DATA_PATH / "X_train.csv",
        index=False
    )

    X_val.to_csv(
        PROCESSED_DATA_PATH / "X_val.csv",
        index=False
    )

    X_test.to_csv(
        PROCESSED_DATA_PATH / "X_test.csv",
        index=False
    )

    y_train.to_csv(
        PROCESSED_DATA_PATH / "y_train.csv",
        index=False
    )

    y_val.to_csv(
        PROCESSED_DATA_PATH / "y_val.csv",
        index=False
    )

    y_test.to_csv(
        PROCESSED_DATA_PATH / "y_test.csv",
        index=False
    )

    # ---------------------------------------------------------
    # 6. Save metadata
    # ---------------------------------------------------------

    metadata = {
        "total_rows": len(df),
        "train_rows": len(X_train),
        "validation_rows": len(X_val),
        "test_rows": len(X_test),
        "number_of_features": X_train.shape[1],
        "target_column": TARGET_COLUMN,
        "id_column": ID_COLUMN,
        "random_state": 42
    }

    with open(
        PROCESSED_DATA_PATH / "data_metadata.json",
        "w"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=4
        )

    print("\nData split completed.")

    print(f"Train      : {X_train.shape}")
    print(f"Validation : {X_val.shape}")
    print(f"Test       : {X_test.shape}")

    print("\nData pipeline completed successfully.")


if __name__ == "__main__":
    run_data_pipeline()
