"""Download UCI AI4I 2020 (ID 601) and create the restricted modelling table.

If the download fails, save the official CSV manually as data/raw/ai4i2020.csv
and run this script again - it will use that file.
"""
import re
import pandas as pd

from . import config as C

CANONICAL = C.FEATURES + [C.TARGET] + C.LEAKAGE_COLUMNS + C.ID_COLUMNS


def _norm(name: str) -> str:
    """Compare names ignoring units in brackets and case."""
    return re.sub(r"\s*\[.*?\]", "", name).strip().lower()


def _canonicalise(df: pd.DataFrame) -> pd.DataFrame:
    lookup = {_norm(c): c for c in CANONICAL}
    return df.rename(columns={c: lookup.get(_norm(c), c) for c in df.columns})


def _fetch() -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=601)
    parts = [ds.data.ids, ds.data.features, ds.data.targets]
    df = pd.concat([p for p in parts if p is not None], axis=1)
    return df


def main() -> None:
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    C.PROC_DIR.mkdir(parents=True, exist_ok=True)

    if C.RAW_FILE.exists():
        print(f"Using existing {C.RAW_FILE}")
        df = pd.read_csv(C.RAW_FILE)
    else:
        print("Downloading from UCI (id=601)...")
        df = _fetch()

    df = _canonicalise(df)
    missing = [c for c in C.FEATURES + [C.TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Expected columns missing: {missing}\nFound: {df.columns.tolist()}")

    df.to_csv(C.RAW_FILE, index=False)
    model_df = df[C.FEATURES + [C.TARGET]].copy()
    model_df.to_csv(C.MODEL_DATA, index=False)

    print("Raw table:", df.shape, "->", C.RAW_FILE)
    print("Modelling table:", model_df.shape, "->", C.MODEL_DATA)
    print(model_df[C.TARGET].value_counts())


if __name__ == "__main__":
    main()
