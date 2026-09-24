import pandas as pd

from sklearn.model_selection import train_test_split


def split_data(
    df: pd.DataFrame,
    target_column: str = "DEFAULT",
    test_size: float = 0.15,
    validation_size: float = 0.15,
    random_state: int = 42
):
    """
    Split dataset into train, validation and test sets.

    Final distribution:
        Train      = 70%
        Validation = 15%
        Test       = 15%
    """

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # First split:
    # 70% train
    # 30% temporary
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=test_size + validation_size,
        stratify=y,
        random_state=random_state
    )

    # Split temporary data equally:
    # 15% validation
    # 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=random_state
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )