import pandas as pd

from src.inference.predictor import (
    predict
)


# Load validation data
X_test = pd.read_csv(
    "data/processed/X_test.csv"
)


# Take a few customers for testing
sample_data = X_test.head(5)


# Generate predictions
result = predict(
    sample_data
)


print("\nPrediction Results:")
print(result)