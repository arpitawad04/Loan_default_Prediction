import pandas as pd


ID_COLUMN = "CUST_ID"


# Features identified during EDA as highly correlated
# We will remove only the redundant 6-month versions
# for the first baseline model.

DROP_FEATURES = [
    "T_HOUSING_6",
    "T_EDUCATION_6",
    "T_UTILITIES_6",
    "T_TAX_6",
    "T_GAMBLING_6",
    "T_GROCERIES_6",
    "T_ENTERTAINMENT_6",
    "T_EXPENDITURE_6",
    "T_TRAVEL_6",
]


def engineer_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Apply feature engineering rules.

    Parameters
    ----------
    df : pd.DataFrame
        Input feature dataframe.

    Returns
    -------
    pd.DataFrame
        Feature-engineered dataframe.
    """

    df = df.copy()

    # ---------------------------------------------------------
    # 1. Remove customer ID from model features
    # ---------------------------------------------------------

    df = df.drop(
        columns=[ID_COLUMN],
        errors="ignore"
    )

    # ---------------------------------------------------------
    # 2. Remove redundant highly correlated features
    # ---------------------------------------------------------

    existing_drop_features = [
        col
        for col in DROP_FEATURES
        if col in df.columns
    ]

    df = df.drop(
        columns=existing_drop_features,
        errors="ignore"
    )

    return df