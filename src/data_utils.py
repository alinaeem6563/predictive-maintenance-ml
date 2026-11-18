"""Data loading, leakage-safe splitting and preprocessing."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C


def load_modeling_data() -> pd.DataFrame:
    if not C.MODEL_DATA.exists():
        raise FileNotFoundError(
            f"{C.MODEL_DATA} not found. Run: python -m src.download_data"
        )
    df = pd.read_csv(C.MODEL_DATA)
    expected = C.FEATURES + [C.TARGET]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df[expected]


def split_data(df: pd.DataFrame):
    """Stratified 70/15/15 split. Deterministic (fixed random_state)."""
    X, y = df[C.FEATURES], df[C.TARGET]
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=C.VAL_SIZE + C.TEST_SIZE,
        stratify=y, random_state=C.RANDOM_STATE,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=C.TEST_SIZE / (C.VAL_SIZE + C.TEST_SIZE),
        stratify=y_tmp, random_state=C.RANDOM_STATE,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), C.CATEGORICAL),
            ("num", StandardScaler(), C.NUMERIC),
        ]
    )
