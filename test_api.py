# test_api.py

import pandas as pd
import requests


BASE_URL = (
    "http://127.0.0.1:8000"
)

TEST_DATA_PATH = (
    "data/processed/X_test.csv"
)


# =========================================================
# Load Test Data
# =========================================================

X_test = pd.read_csv(
    TEST_DATA_PATH
)

print(
    "X_test shape:",
    X_test.shape
)


# =========================================================
# 1. TEST SINGLE CUSTOMER API
# =========================================================

print("\n" + "=" * 60)
print("TESTING SINGLE CUSTOMER API")
print("=" * 60)


single_customer = (
    X_test
    .head(1)
    .iloc[0]
    .where(
        pd.notna(
            X_test
            .head(1)
            .iloc[0]
        ),
        None
    )
    .to_dict()
)


response = requests.post(
    f"{BASE_URL}/predict",
    json=single_customer
)


print(
    "Status code:",
    response.status_code
)

print(
    "Response:"
)

print(
    response.json()
)


# =========================================================
# 2. TEST BATCH API
# =========================================================

print("\n" + "=" * 60)
print("TESTING BATCH PREDICTION API")
print("=" * 60)


NUMBER_OF_ROWS = 5

batch_data = (
    X_test
    .head(NUMBER_OF_ROWS)
    .copy()
)


# Convert DataFrame → list of dictionaries

batch_customers = (
    batch_data
    .where(
        pd.notna(batch_data),
        None
    )
    .to_dict(
        orient="records"
    )
)


response = requests.post(
    f"{BASE_URL}/predict/batch",
    json=batch_customers
)


print(
    "Status code:",
    response.status_code
)

print(
    "Response:"
)

print(
    response.json()
)