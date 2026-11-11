"""Primary metric: TPR at a fixed FPR, threshold set on held-out REAL validation images."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def threshold_at_fpr(real_val_scores: np.ndarray, fpr: float = 0.01) -> float:
    """Score above which a real image is flagged with probability <= fpr."""
    return float(np.quantile(real_val_scores, 1.0 - fpr, method="higher"))


def tpr_at_threshold(fake_scores: np.ndarray, thr: float) -> float:
    return float(np.mean(fake_scores > thr))


def fpr_at_threshold(real_scores: np.ndarray, thr: float) -> float:
    return float(np.mean(real_scores > thr))


def auc(real_scores: np.ndarray, fake_scores: np.ndarray) -> float:
    y = np.r_[np.zeros(len(real_scores)), np.ones(len(fake_scores))]
    return float(roc_auc_score(y, np.r_[real_scores, fake_scores]))
