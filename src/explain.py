"""Permutation feature importance on the validation set (scoring: Average Precision)."""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

from . import config as C
from .data_utils import load_modeling_data, split_data


def main() -> None:
    bundle = joblib.load(C.MODEL_PATH)
    _, X_val, _, _, y_val, _ = split_data(load_modeling_data())
    r = permutation_importance(bundle["pipeline"], X_val, y_val,
                               scoring="average_precision", n_repeats=20,
                               random_state=C.RANDOM_STATE, n_jobs=-1)
    imp = (pd.DataFrame({"feature": X_val.columns,
                         "importance_mean": r.importances_mean,
                         "importance_std": r.importances_std})
           .sort_values("importance_mean", ascending=False))
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    imp.to_csv(C.REPORTS_DIR / "permutation_feature_importance.csv", index=False)
    print(imp.round(4).to_string(index=False))

    d = imp.sort_values("importance_mean")
    plt.figure(figsize=(7, 4))
    plt.barh(d["feature"], d["importance_mean"], xerr=d["importance_std"])
    plt.xlabel("Decrease in Average Precision when permuted")
    plt.title("Permutation feature importance (validation set)")
    plt.savefig(C.FIG_DIR / "permutation_feature_importance.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
