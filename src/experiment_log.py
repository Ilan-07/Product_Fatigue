"""
experiment_log.py — Lightweight CSV-based experiment tracking.

Appends one row per training run to outputs/experiment_log.csv so you can
compare metrics across code changes without any external dependencies.

Usage
-----
From main.py (called automatically after each modality/model is evaluated):

    from src.experiment_log import log_run
    log_run(modality="reviews", model="xgboost", ...)

To view run history from the terminal:

    python src/experiment_log.py
"""

import csv
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

LOG_PATH = os.path.join("outputs", "experiment_log.csv")

_COLUMNS = [
    "timestamp",
    "modality",
    "model",
    "n_features",
    "n_train",
    "n_test",
    "cv_f1",
    "test_f1",
    "cv_test_gap",
    "optimal_threshold",
    "balanced_accuracy",
    "macro_recall",
    "prediction_distribution_drift_l1",
    "scenario_score",
    "raw_brier_score",
    "raw_ece",
    "calibrated_brier_score",
    "calibrated_ece",
    "best_params",
]


def log_run(
    modality: str,
    model: str,
    n_features: int,
    n_train: int,
    n_test: int,
    cv_f1: float,
    test_f1: float,
    optimal_threshold: Optional[float] = None,
    balanced_accuracy: Optional[float] = None,
    macro_recall: Optional[float] = None,
    prediction_distribution_drift_l1: Optional[float] = None,
    scenario_score: Optional[float] = None,
    raw_brier_score: Optional[float] = None,
    raw_ece: Optional[float] = None,
    calibrated_brier_score: Optional[float] = None,
    calibrated_ece: Optional[float] = None,
    best_params: Optional[Dict[str, Any]] = None,
    log_path: str = LOG_PATH,
) -> None:
    """
    Append one row to the experiment log.

    Parameters
    ----------
    modality           : "reviews" | "sales" | "usage"
    model              : "xgboost" | "random_forest" | "logistic_regression"
    n_features         : number of features after preprocessing
    n_train / n_test   : training and test set sizes
    cv_f1              : best cross-validation F1 macro from GridSearchCV
    test_f1            : held-out test F1 macro
    optimal_threshold  : decision threshold found by find_optimal_threshold()
    best_params        : dict of best hyperparameters from GridSearchCV
    log_path           : path to the CSV log file
    """
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    write_header = not os.path.exists(log_path)

    if not write_header:
        with open(log_path, "r", newline="") as fh:
            reader = csv.DictReader(fh)
            existing_columns = reader.fieldnames or []
            existing_rows = list(reader)
        if existing_columns != _COLUMNS:
            with open(log_path, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=_COLUMNS)
                writer.writeheader()
                for existing in existing_rows:
                    writer.writerow({col: existing.get(col, "") for col in _COLUMNS})

    row = {
        "timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "modality":          modality,
        "model":             model,
        "n_features":        n_features,
        "n_train":           n_train,
        "n_test":            n_test,
        "cv_f1":             round(float(cv_f1), 6),
        "test_f1":           round(float(test_f1), 6),
        "cv_test_gap":       round(float(cv_f1) - float(test_f1), 6),
        "optimal_threshold": round(float(optimal_threshold), 4) if optimal_threshold is not None else "",
        "balanced_accuracy": round(float(balanced_accuracy), 6) if balanced_accuracy is not None else "",
        "macro_recall": round(float(macro_recall), 6) if macro_recall is not None else "",
        "prediction_distribution_drift_l1": (
            round(float(prediction_distribution_drift_l1), 6)
            if prediction_distribution_drift_l1 is not None else ""
        ),
        "scenario_score": round(float(scenario_score), 6) if scenario_score is not None else "",
        "raw_brier_score": round(float(raw_brier_score), 6) if raw_brier_score is not None else "",
        "raw_ece": round(float(raw_ece), 6) if raw_ece is not None else "",
        "calibrated_brier_score": (
            round(float(calibrated_brier_score), 6)
            if calibrated_brier_score is not None else ""
        ),
        "calibrated_ece": round(float(calibrated_ece), 6) if calibrated_ece is not None else "",
        "best_params":       json.dumps(best_params or {}),
    }

    with open(log_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def compare_runs(log_path: str = LOG_PATH, n: int = 30) -> None:
    """
    Print the last *n* experiment runs as a formatted table.

    Call from the terminal:  python src/experiment_log.py
    """
    if not os.path.exists(log_path):
        print(f"No experiment log found at {log_path}. Run main.py first.")
        return

    with open(log_path, "r", newline="") as fh:
        rows: List[Dict[str, str]] = list(csv.DictReader(fh))

    if not rows:
        print("Experiment log is empty.")
        return

    recent = rows[-n:]

    # Group by (modality, model) and highlight best test_f1
    best: Dict[str, float] = {}
    for r in recent:
        key = f"{r['modality']}/{r['model']}"
        val = float(r["test_f1"]) if r["test_f1"] else 0.0
        best[key] = max(best.get(key, 0.0), val)

    header = (
        f"  {'Timestamp':<20} {'Modality':<9} {'Model':<24} "
        f"{'Feat':>5} {'Train':>7} {'Test':>7} "
        f"{'CV F1':>7} {'Test F1':>8} {'Gap':>7} {'Thr':>5}"
    )
    sep = "=" * (len(header) - 2)
    print(f"\n  {sep}")
    print(f"  Experiment Log — last {len(recent)} run(s)  [{log_path}]")
    print(f"  {sep}")
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    for r in recent:
        key = f"{r['modality']}/{r['model']}"
        is_best = float(r["test_f1"]) == best[key] if r["test_f1"] else False
        marker = "★" if is_best else " "
        gap = float(r["cv_test_gap"]) if r["cv_test_gap"] else 0.0
        thr = r["optimal_threshold"] if r["optimal_threshold"] else " —"
        print(
            f"{marker} {r['timestamp']:<20} {r['modality']:<9} {r['model']:<24} "
            f"{r['n_features']:>5} {r['n_train']:>7} {r['n_test']:>7} "
            f"{float(r['cv_f1']):>7.4f} {float(r['test_f1']):>8.4f} "
            f"{gap:>+7.4f} {thr:>5}"
        )

    print(f"  {sep}\n  ★ = best test F1 per modality/model\n")


if __name__ == "__main__":
    log_path = sys.argv[1] if len(sys.argv) > 1 else LOG_PATH
    compare_runs(log_path=log_path)
