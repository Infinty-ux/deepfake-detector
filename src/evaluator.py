from typing import List, Dict
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    average_precision_score,
)


def compute_metrics(
    y_true: List[int],
    y_prob: List[float],
    threshold: float = 0.5,
) -> Dict:
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "ap": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def full_evaluation_report(
    y_true: List[int],
    y_prob: List[float],
    thresholds: List[float] = None,
) -> Dict:
    thresholds = thresholds or [0.3, 0.4, 0.5, 0.6, 0.7]
    y_true_np = np.array(y_true)
    y_prob_np = np.array(y_prob)

    report = {
        "auc": float(roc_auc_score(y_true_np, y_prob_np)),
        "average_precision": float(average_precision_score(y_true_np, y_prob_np)),
    }

    threshold_reports = {}
    for t in thresholds:
        y_pred = (y_prob_np >= t).astype(int)
        cm = confusion_matrix(y_true_np, y_pred)
        threshold_reports[str(t)] = {
            "f1": float(f1_score(y_true_np, y_pred, zero_division=0)),
            "accuracy": float(accuracy_score(y_true_np, y_pred)),
            "precision": float(precision_score(y_true_np, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true_np, y_pred, zero_division=0)),
            "confusion_matrix": cm.tolist(),
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        }

    report["threshold_analysis"] = threshold_reports
    return report


def find_optimal_threshold(y_true: List[int], y_prob: List[float]) -> float:
    y_true_np = np.array(y_true)
    y_prob_np = np.array(y_prob)
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.1, 0.9, 81):
        f1 = f1_score(y_true_np, (y_prob_np >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t
