# api/main.py

from fastapi import FastAPI, HTTPException

import pandas as pd

from src.inference.predictor import predict


app = FastAPI(
    title="Loan Default Prediction API",
    description=(
        "API for predicting customer default probability "
        "using the registered MLflow model."
    ),
    version="1.0.0"
)


# ---------------------------------------------------------
# Health Check
# ---------------------------------------------------------

@app.get("/health")
def health_check():

    return {
        "status": "healthy"
    }


# ---------------------------------------------------------
# Single Customer Prediction
# ---------------------------------------------------------

@app.post("/predict")
def prediction(input_data: dict):

    try:

        # Convert single JSON customer into DataFrame
        input_df = pd.DataFrame(
            [input_data]
        )

        # Generate prediction
        result = predict(
            input_df
        )

        prediction_result = result.iloc[0]

        return {
            "default_probability": float(
                prediction_result[
                    "default_probability"
                ]
            ),
            "prediction": int(
                prediction_result[
                    "prediction"
                ]
            )
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ---------------------------------------------------------
# Batch Customer Prediction
# ---------------------------------------------------------

@app.post("/predict/batch")
def batch_prediction(input_data: list[dict]):

    try:

        # Convert multiple JSON records into DataFrame
        input_df = pd.DataFrame(
            input_data
        )

        # Generate predictions for all customers
        result = predict(
            input_df
        )

        # Convert DataFrame results to JSON records
        predictions = []

        for index, row in result.iterrows():

            predictions.append({
                "row_index": index,
                "default_probability": float(
                    row[
                        "default_probability"
                    ]
                ),
                "prediction": int(
                    row[
                        "prediction"
                    ]
                )
            })

        return {
            "number_of_customers": len(
                predictions
            ),
            "predictions": predictions
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )