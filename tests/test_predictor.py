import pandas as pd

from src.inference.predictor import predict


def test_predictor():

    X_test = pd.read_csv(
        "data/processed/X_test.csv"
    )

    sample_data = (
        X_test
        .head(2)
        .copy()
    )

    result = predict(
        sample_data
    )

    assert len(result) == 2

    assert "default_probability" in result.columns

    assert "prediction" in result.columns

    assert result[
        "default_probability"
    ].between(
        0,
        1
    ).all()

    assert result[
        "prediction"
    ].isin(
        [0, 1]
    ).all()