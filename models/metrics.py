"""
FedMedVision — Clinical Classification & Calibration Metrics
Vectorized multi-label clinical evaluation:
- Per-class AUROC, Macro-AUROC, Macro-AUPRC
- Optimal threshold F1 score
- Expected Calibration Error (ECE) for clinical reliability
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    """
    Compute comprehensive clinical classification metrics for multi-label chest X-ray predictions.

    Args:
        y_true: Binary ground-truth matrix of shape (N, num_classes)
        y_probs: Predicted probability matrix of shape (N, num_classes) in [0, 1]
        class_names: List of string names for each pathology

    Returns:
        Dictionary containing per-class AUC, Macro-AUROC, Macro-AUPRC, and optimal macro F1.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_probs = np.asarray(y_probs, dtype=float)

    num_classes = len(class_names)
    assert y_true.shape[1] == num_classes, f"Class dimension mismatch: {y_true.shape[1]} vs {num_classes}"
    assert y_probs.shape[1] == num_classes, f"Probability dimension mismatch: {y_probs.shape[1]} vs {num_classes}"

    metrics: dict[str, Any] = {
        "per_class_auroc": {},
        "per_class_auprc": {},
        "macro_auroc": 0.0,
        "macro_auprc": 0.0,
        "macro_f1": 0.0,
    }

    auroc_list = []
    auprc_list = []
    f1_list = []

    for c in range(num_classes):
        c_name = class_names[c]
        y_t = y_true[:, c]
        y_p = y_probs[:, c]

        # Handle edge-case where a class has only 0s or only 1s in split
        unique_vals = np.unique(y_t)
        if len(unique_vals) < 2:
            auc = 0.5
            auprc = float(np.mean(y_t)) if len(y_t) > 0 else 0.0
            best_f1 = 0.0
        else:
            try:
                auc = float(roc_auc_score(y_t, y_p))
            except Exception:
                auc = 0.5

            try:
                auprc = float(average_precision_score(y_t, y_p))
            except Exception:
                auprc = 0.0

            # Find best threshold on PR curve for optimal F1
            try:
                precision, recall, thresholds = precision_recall_curve(y_t, y_p)
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
                best_f1 = float(np.max(f1_scores))
            except Exception:
                best_f1 = float(f1_score(y_t, (y_p >= 0.5).astype(int), zero_division=0))

        metrics["per_class_auroc"][c_name] = round(auc, 4)
        metrics["per_class_auprc"][c_name] = round(auprc, 4)
        auroc_list.append(auc)
        auprc_list.append(auprc)
        f1_list.append(best_f1)

    metrics["macro_auroc"] = round(float(np.mean(auroc_list)), 4)
    metrics["macro_auprc"] = round(float(np.mean(auprc_list)), 4)
    metrics["macro_f1"] = round(float(np.mean(f1_list)), 4)

    return metrics


def compute_ece(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Compute multi-label Expected Calibration Error (ECE) across all disease predictions.
    Partitions predictions into `n_bins` equally spaced confidence intervals and
    computes the weighted absolute difference between observed accuracy and average confidence.

    Args:
        y_true: Binary ground-truth matrix of shape (N, num_classes)
        y_probs: Predicted probability matrix of shape (N, num_classes) in [0, 1]
        n_bins: Number of confidence bins (standard is 15)

    Returns:
        float: Expected Calibration Error value in [0, 1]
    """
    y_true_flat = np.asarray(y_true, dtype=float).flatten()
    y_probs_flat = np.asarray(y_probs, dtype=float).flatten()

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(y_probs_flat)

    if total_samples == 0:
        return 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Samples falling into the current bin
        if i == n_bins - 1:
            in_bin = (y_probs_flat >= bin_lower) & (y_probs_flat <= bin_upper)
        else:
            in_bin = (y_probs_flat >= bin_lower) & (y_probs_flat < bin_upper)

        bin_count = np.sum(in_bin)
        if bin_count > 0:
            avg_confidence = float(np.mean(y_probs_flat[in_bin]))
            avg_accuracy = float(np.mean(y_true_flat[in_bin]))
            bin_weight = bin_count / total_samples
            ece += bin_weight * abs(avg_accuracy - avg_confidence)

    return round(float(ece), 4)


def fit_temperature_scaler(
    logits: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
    max_iter: int = 50,
    lr: float = 0.01,
) -> tuple[float, np.ndarray]:
    """
    Fit post-hoc scalar temperature T > 0 using L-BFGS optimization on binary cross entropy
    loss across all multi-label targets:
      min_T  BCEWithLogitsLoss(logits / T, targets)

    Args:
        logits: Raw uncalibrated network logits of shape (N, num_classes)
        targets: Binary ground-truth matrix of shape (N, num_classes)
        max_iter: Maximum L-BFGS iterations
        lr: Optimization learning rate

    Returns:
        optimal_temperature: Optimized scalar float T > 0
        calibrated_probs: Scaled probabilities torch.sigmoid(logits / T) as np.ndarray
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim

    if isinstance(logits, np.ndarray):
        logits_t = torch.from_numpy(logits).float()
    else:
        logits_t = logits.detach().cpu().float()

    if isinstance(targets, np.ndarray):
        targets_t = torch.from_numpy(targets).float()
    else:
        targets_t = targets.detach().cpu().float()

    # Temperature parameter initialized to 1.0 (log_temp = 0.0)
    log_temperature = nn.Parameter(torch.zeros(1, requires_grad=True))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.LBFGS([log_temperature], lr=lr, max_iter=max_iter)

    def eval_loss():
        optimizer.zero_grad()
        T = torch.exp(log_temperature)
        loss = criterion(logits_t / T, targets_t)
        loss.backward()
        return loss

    try:
        optimizer.step(eval_loss)
    except Exception:
        pass

    optimal_T = float(torch.exp(log_temperature).item())
    # Bounded between 0.05 and 10.0 for clinical stability
    optimal_T = max(0.05, min(10.0, optimal_T))

    with torch.no_grad():
        calibrated_probs = torch.sigmoid(logits_t / optimal_T).numpy()

    return round(optimal_T, 4), calibrated_probs


