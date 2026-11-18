"""Final one-time evaluation on the untouched test set."""
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay,
                             RocCurveDisplay, classification_report,
                             confusion_matrix)

from . import config as C
from .data_utils import load_modeling_data, split_data
from .train import metrics_at


def main() -> None:
    bundle = joblib.load(C.MODEL_PATH)   # only load files you created yourself
    pipe, thr = bundle["pipeline"], bundle["threshold"]
    _, _, X_test, _, _, y_test = split_data(load_modeling_data())

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= thr).astype(int)
    m = metrics_at(y_test, proba, thr)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    out = {"model": bundle["model_name"], "threshold": thr,
           "test_prevalence": float(y_test.mean()), **m,
           "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}

    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    (C.REPORTS_DIR / "test_metrics.json").write_text(json.dumps(out, indent=2))
    (C.REPORTS_DIR / "classification_report.txt").write_text(
        classification_report(y_test, pred, target_names=["No failure", "Failure"], zero_division=0))

    ConfusionMatrixDisplay.from_predictions(y_test, pred, cmap="Blues")
    plt.title(f"Confusion matrix (threshold={thr:.2f})")
    plt.savefig(C.FIG_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight"); plt.close()
    PrecisionRecallDisplay.from_predictions(y_test, proba)
    plt.title("Precision-Recall curve (test)")
    plt.savefig(C.FIG_DIR / "precision_recall_curve.png", dpi=150, bbox_inches="tight"); plt.close()
    RocCurveDisplay.from_predictions(y_test, proba)
    plt.title("ROC curve (test)")
    plt.savefig(C.FIG_DIR / "roc_curve.png", dpi=150, bbox_inches="tight"); plt.close()

    print(json.dumps(out, indent=2))
    print(classification_report(y_test, pred, target_names=["No failure", "Failure"], zero_division=0))


if __name__ == "__main__":
    main()
