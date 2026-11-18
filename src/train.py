"""Train baseline + 3 classifiers, compare on validation, choose threshold."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from . import config as C
from .data_utils import build_preprocessor, load_modeling_data, split_data


def get_models():
    rs = C.RANDOM_STATE
    return {
        "Dummy (prior)": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=rs),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", max_depth=6, min_samples_leaf=10, random_state=rs),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample",
            min_samples_leaf=3, n_jobs=-1, random_state=rs),
    }


def metrics_at(y_true, proba, threshold=0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "average_precision": average_precision_score(y_true, proba),
    }


def best_f1_threshold(y_true, proba) -> float:
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[int(np.argmax(f1))])


def main() -> None:
    df = load_modeling_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    print(f"Train {len(X_train)} | Val {len(X_val)} | Test {len(X_test)}")
    print("Failure rate  train/val/test:",
          round(y_train.mean(), 4), round(y_val.mean(), 4), round(y_test.mean(), 4))

    rows, fitted = [], {}
    for name, est in get_models().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", est)])
        pipe.fit(X_train, y_train)               # training data only
        proba = pipe.predict_proba(X_val)[:, 1]
        rows.append({"model": name, **metrics_at(y_val, proba, 0.5)})
        fitted[name] = pipe

    results = pd.DataFrame(rows).set_index("model").round(4)
    print("\nValidation results (threshold 0.50):\n", results)

    candidates = results.drop(index="Dummy (prior)")
    best_name = candidates["average_precision"].idxmax()
    best_pipe = fitted[best_name]
    val_proba = best_pipe.predict_proba(X_val)[:, 1]
    threshold = best_f1_threshold(y_val, val_proba)
    tuned = metrics_at(y_val, val_proba, threshold)

    print(f"\nSelected model: {best_name} (best validation Average Precision)")
    print(f"Selected threshold (max validation F1): {threshold:.4f}")
    print("Validation metrics at that threshold:", {k: round(v, 4) for k, v in tuned.items()})

    C.MODELS_DIR.mkdir(exist_ok=True)
    C.REPORTS_DIR.mkdir(exist_ok=True)
    joblib.dump({"pipeline": best_pipe, "threshold": threshold,
                 "model_name": best_name, "features": C.FEATURES}, C.MODEL_PATH)
    results.to_csv(C.REPORTS_DIR / "validation_model_comparison.csv")
    with open(C.REPORTS_DIR / "selected_model.json", "w") as f:
        json.dump({"model": best_name, "threshold": threshold,
                   "validation_at_threshold": tuned}, f, indent=2)
    print("Saved:", C.MODEL_PATH)


if __name__ == "__main__":
    main()
