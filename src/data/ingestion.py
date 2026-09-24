from pathlib import Path
import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load raw credit data from CSV.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if path.suffix.lower() != ".csv":
        raise ValueError("Only CSV files are supported.")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Loaded dataset is empty.")

    return df