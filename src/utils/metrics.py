"""
Evaluation metrics for AI audio detection.

Reports metrics at multiple confidence thresholds for the publication
and for selecting the browser extension's operating point.
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> tuple[float, float]:
    """
    Compute Equal Error Rate (EER) -- the primary metric for
    anti-spoofing / deepfake detection (ASVspoof standard).
    """
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr

    # Find the threshold where FPR == FNR
    try:
        eer_threshold = brentq(lambda x: interp1d(fpr, thresholds)(x) - interp1d(fnr, thresholds)(x), 0, 1)
        eer = interp1d(fpr, fnr)(eer_threshold)
    except ValueError:
        # Fallback: find closest point
        idx = np.nanargmin(np.abs(fpr - fnr))
        eer = (fpr[idx] + fnr[idx]) / 2
        eer_threshold = thresholds[idx]

    return float(eer), float(eer_threshold)


def compute_metrics(
    regression_scores: np.ndarray,
    class_logits: np.ndarray,
    ai_ratios: np.ndarray,
    class_labels: np.ndarray,
    thresholds: list[float] = [0.5, 0.6, 0.7, 0.8, 0.9],
) -> dict:
    """
    Compute full evaluation metrics for dual-head model.

    Args:
        regression_scores: (N,) predicted AI probability
        class_logits: (N, 3) predicted class logits
        ai_ratios: (N,) ground truth AI ratio
        class_labels: (N,) ground truth class (0=real, 1=mixed, 2=AI)
        thresholds: confidence thresholds for binary detection metrics

    Returns:
        Dictionary of all metrics.
    """
    results = {}

    # --- Regression metrics ---
    mse = float(np.mean((regression_scores - ai_ratios) ** 2))
    mae = float(np.mean(np.abs(regression_scores - ai_ratios)))
    results["regression/mse"] = mse
    results["regression/mae"] = mae

    # --- Binary detection (real vs any-AI) ---
    # Convert to binary: real (0) vs contains-AI (1)
    binary_true = (ai_ratios >= 0.2).astype(int)

    if len(np.unique(binary_true)) > 1:
        results["binary/auc_roc"] = float(roc_auc_score(binary_true, regression_scores))
        results["binary/avg_precision"] = float(average_precision_score(binary_true, regression_scores))

        eer, eer_thresh = compute_eer(binary_true, regression_scores)
        results["binary/eer"] = eer
        results["binary/eer_threshold"] = eer_thresh

        # Metrics at each threshold (for browser extension operating point selection)
        for t in thresholds:
            preds = (regression_scores >= t).astype(int)
            tp = np.sum((preds == 1) & (binary_true == 1))
            fp = np.sum((preds == 1) & (binary_true == 0))
            fn = np.sum((preds == 0) & (binary_true == 1))
            tn = np.sum((preds == 0) & (binary_true == 0))

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            fpr = fp / (fp + tn + 1e-8)

            results[f"binary/precision@{t}"] = float(precision)
            results[f"binary/recall@{t}"] = float(recall)
            results[f"binary/f1@{t}"] = float(f1)
            results[f"binary/fpr@{t}"] = float(fpr)

    # --- 3-class classification metrics ---
    class_preds = np.argmax(class_logits, axis=1)
    class_names = ["real", "mixed", "AI"]

    correct = (class_preds == class_labels).sum()
    results["classification/accuracy"] = float(correct / len(class_labels))

    for i, name in enumerate(class_names):
        mask = class_labels == i
        if mask.sum() > 0:
            acc = (class_preds[mask] == i).mean()
            results[f"classification/accuracy_{name}"] = float(acc)

    return results


def format_metrics_table(metrics: dict) -> str:
    """Format metrics as a printable table."""
    lines = []
    sections = {}
    for key, val in sorted(metrics.items()):
        section = key.split("/")[0]
        if section not in sections:
            sections[section] = []
        sections[section].append((key, val))

    for section, entries in sections.items():
        lines.append(f"\n{'=' * 50}")
        lines.append(f"  {section.upper()}")
        lines.append(f"{'=' * 50}")
        for key, val in entries:
            metric_name = key.split("/", 1)[1]
            lines.append(f"  {metric_name:<30} {val:.4f}")

    return "\n".join(lines)
