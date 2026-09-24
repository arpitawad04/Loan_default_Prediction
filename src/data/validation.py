import pandas as pd


TARGET_COLUMN = "DEFAULT"
ID_COLUMN = "CUST_ID"

def validate_data(df:pd.DataFrame) ->None:
    """
    Validate raw dataset before further processing.
    """
    # 1.Required columns 
    required_columns = [
        TARGET_COLUMN,
        ID_COLUMN
    ]

    missing_columns=[col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )


    # 2. Duplicate rows
    if df.duplicated().any():
        duplicated_count=df.duplicated().sum()

        raise ValueError(
            f"Dataset contains {duplicated_count} duplicate rows."
        )

    # 3. Duplicate customer IDs
    if df[ID_COLUMN].duplicated().any():
        duplicate_count = df[ID_COLUMN].duplicated().sum()

        raise ValueError(
            f"Dataset contains {duplicate_count} duplicate customer IDs."
        )

    # 4. Missing values
    missing_values = df.isnull().sum()

    missing_columns = missing_values[
        missing_values > 0
    ]

    if not missing_columns.empty:
        raise ValueError(
            f"Missing values found:\n{missing_columns}"
        )

    # 5. Target validation
    unique_target_values = set(
        df[TARGET_COLUMN].unique()
    )

    if not unique_target_values.issubset({0, 1}):
        raise ValueError(
            f"Target must contain only 0 and 1. "
            f"Found: {unique_target_values}"
        )

    # 6. Target cannot be empty
    if df[TARGET_COLUMN].empty:
        raise ValueError("Target column is empty.")

    print("Data validation passed successfully.")