# ---------------------------------------------------------
# Features selected for model training
# ---------------------------------------------------------
# Add only the variables that you want to use in the model.
# Do NOT include the ID column.

SELECTED_FEATURES = ['CREDIT_SCORE','R_UTILITIES_DEBT','R_DEBT_SAVINGS','R_GROCERIES_DEBT',
                     'R_CLOTHING_DEBT','R_TAX_DEBT','R_EDUCATION_DEBT','CAT_DEPENDENTS']


def select_features(X):
    """
    Select predefined features from the dataset.

    Parameters
    ----------
    X : pandas.DataFrame
        Input dataset after feature engineering.

    Returns
    -------
    pandas.DataFrame
        Dataset containing only selected features.
    """

    # Check whether all selected features exist
    missing_features = [
        feature
        for feature in SELECTED_FEATURES
        if feature not in X.columns
    ]

    if missing_features:
        raise ValueError(
            f"The following selected features are missing: "
            f"{missing_features}"
        )

    # Select only the required model variables
    X_selected = X[SELECTED_FEATURES].copy()

    return X_selected