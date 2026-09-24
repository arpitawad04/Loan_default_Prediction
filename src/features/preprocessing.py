import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

ID_COLUMN = "CUST_ID"

def create_preprocessing_pipeline(X_train:pd.DataFrame):
    """
    Create preprocessing pipeline using training data schema.
    """

    # ---------------------------------------------------------
    # 1. 
    # ---------------------------------------------------------

    # X_train=X_train.drop(columns=[ID_COLUMN],errors="ignore")

    # ---------------------------------------------------------
    # 2. Identify numerical and categorical columns
    # ---------------------------------------------------------

    numerical_features=X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features=X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    print("Numerical features:", len(numerical_features))
    print("Categorical features:", len(categorical_features))

    print("Categorical columns:", categorical_features)

    # ---------------------------------------------------------
    # 3. Numerical preprocessing
    # ---------------------------------------------------------

    numerical_pipeline=Pipeline(steps=[('Imputer', SimpleImputer(strategy='median')),('scaler', StandardScaler())])

    # ---------------------------------------------------------
    # 4. Categorical preprocessing
    # ---------------------------------------------------------

    categorical_pipeline=Pipeline(steps=[('Imputer', SimpleImputer(strategy='most_frequent')),('encoder', OneHotEncoder(handle_unknown='ignore'))])

    # ---------------------------------------------------------
    # 5. Combine both pipelines
    # ---------------------------------------------------------

    preprocessor =ColumnTransformer(transformers=[('numerical',numerical_pipeline,numerical_features),('categorical',categorical_pipeline,categorical_features)])

    return preprocessor