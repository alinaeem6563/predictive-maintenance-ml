"""Central configuration: paths, column names, split settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

RAW_FILE = RAW_DIR / "ai4i2020.csv"
MODEL_DATA = PROC_DIR / "ai4i_modeling_data.csv"
MODEL_PATH = MODELS_DIR / "predictive_maintenance_model.joblib"

CATEGORICAL = ["Type"]
NUMERIC = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "Machine failure"

# Never used as predictors (target leakage / identifiers)
LEAKAGE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
ID_COLUMNS = ["UDI", "Product ID"]

RANDOM_STATE = 42
TRAIN_SIZE, VAL_SIZE, TEST_SIZE = 0.70, 0.15, 0.15
