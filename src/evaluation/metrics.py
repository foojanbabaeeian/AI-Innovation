"""
Evaluation metrics for audio deepfake detection.

Implements:
- Equal Error Rate (EER): threshold where FAR == FRR
- Minimum Detection Cost Function (min-DCF): ASVspoof official metric

These are computed from raw scores (model confidence for the "spoof" class)
and ground-truth labels. Higher scores should indicate higher likelihood of
being AI-generated/spoofed.

For final published results, use the official ASVspoof scoring scripts:
https://github.com/asvspoof-challenge/asvspoof2021
Our implementations here are for real-time monitoring during training.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> tuple:
    """Compute Equal Error Rate.

    EER is the operating point where False Acceptance Rate equals
    False Rejection Rate. Lower is better.

    Args:
        labels: Ground truth binary labels. 0 = bonafide/real, 1 = spoof/AI.
        scores: Model confidence scores for the spoof/AI class. Higher = more
                likely to be AI-generated.

    Returns:
        Tuple of (eer, threshold):
        - eer: Equal Error Rate as a float in [0, 1].
        - threshold: Score threshold at the EER operating point.
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    # Find the point where FPR == FNR via interpolation
    try:
        eer = brentq(
            lambda x: interp1d(fpr, fpr)(x) - interp1d(fpr, fnr)(x),
            0.0,
            1.0,
        )
        # Find the threshold closest to EER
        eer_idx = np.nanargmin(np.abs(fpr - eer))
        threshold = thresholds[eer_idx]
    except ValueError:
        # Fallback: find crossing point directly
        abs_diff = np.abs(fpr - fnr)
        eer_idx = np.nanargmin(abs_diff)
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
        threshold = thresholds[eer_idx]

    return float(eer), float(threshold)


def compute_min_dcf(
    labels: np.ndarray,
    scores: np.ndarray,
    p_target: float = 0.05,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
) -> tuple:
    """Compute minimum Detection Cost Function (min-DCF).

    This is the primary metric for ASVspoof challenges. It computes the
    minimum normalized DCF over all possible thresholds.

    DCF = c_miss * p_miss * p_target + c_fa * p_fa * (1 - p_target)

    Args:
        labels: Ground truth binary labels. 0 = bonafide, 1 = spoof.
        scores: Model confidence scores for the spoof class.
        p_target: Prior probability of a target (spoof) trial.
        c_miss: Cost of missing a spoof (false negative).
        c_fa: Cost of false alarm (false positive).

    Returns:
        Tuple of (min_dcf, threshold):
        - min_dcf: Minimum normalized DCF value.
        - threshold: Optimal threshold for min-DCF.
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    # Compute DCF at each threshold
    dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)

    # Normalize by the best cost without the system (always accept or always reject)
    c_default = min(c_miss * p_target, c_fa * (1 - p_target))
    dcf_norm = dcf / c_default

    # Find minimum
    min_idx = np.argmin(dcf_norm)
    min_dcf = dcf_norm[min_idx]
    threshold = thresholds[min_idx] if min_idx < len(thresholds) else 0.0

    return float(min_dcf), float(threshold)


def compute_accuracy(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> float:
    """Compute binary classification accuracy at a given threshold.

    Args:
        labels: Ground truth binary labels.
        scores: Model confidence scores for the positive (spoof) class.
        threshold: Decision threshold.

    Returns:
        Accuracy as a float in [0, 1].
    """
    labels = np.asarray(labels, dtype=int)
    predictions = (np.asarray(scores) >= threshold).astype(int)
    return float(np.mean(predictions == labels))


def compute_all_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    p_target: float = 0.05,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
) -> dict:
    """Compute all evaluation metrics at once.

    Args:
        labels: Ground truth binary labels.
        scores: Model confidence scores for the positive (spoof) class.
        p_target: Prior probability for min-DCF.
        c_miss: Miss cost for min-DCF.
        c_fa: False alarm cost for min-DCF.

    Returns:
        Dictionary with keys: 'eer', 'eer_threshold', 'min_dcf',
        'min_dcf_threshold', 'accuracy'.
    """
    eer, eer_thresh = compute_eer(labels, scores)
    min_dcf, dcf_thresh = compute_min_dcf(labels, scores, p_target, c_miss, c_fa)
    accuracy = compute_accuracy(labels, scores, threshold=eer_thresh)

    return {
        "eer": eer,
        "eer_threshold": eer_thresh,
        "min_dcf": min_dcf,
        "min_dcf_threshold": dcf_thresh,
        "accuracy": accuracy,
    }
