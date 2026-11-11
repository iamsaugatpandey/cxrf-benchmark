"""Shortcut audit (Pipeline step 2): can format-only features separate two image sets?

Features are computed on the *preprocessed* image the detector sees (uint8, square):
  * 32-bin intensity histogram (normalised)
  * mean, std, skewness, kurtosis, 1/5/50/95/99th percentiles, fraction at 0 and at 255
  * the 8 lowest bins of the radial log power spectrum (global frequency content only)
  * marker proxy: fraction of near-white pixels (>= 245) in each 12%-sized corner box,
    plus the max over corners (burned-in text/markers are near-white)
A logistic regression on these features with group-aware 5-fold CV gives an out-of-fold AUC.
AUC near 0.5 = no format shortcut. The marker block can be dropped to see its contribution.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cxrf.detectors import fft_features

HIST_BINS = 32
FFT_LOW = 8
CORNER = 0.12


def _corner_fracs(im: np.ndarray) -> np.ndarray:
    h, w = im.shape
    ch, cw = int(h * CORNER), int(w * CORNER)
    boxes = [im[:ch, :cw], im[:ch, -cw:], im[-ch:, :cw], im[-ch:, -cw:]]
    f = np.array([(b >= 245).mean() for b in boxes], np.float32)
    return np.r_[f, f.max()]


def features(imgs: np.ndarray) -> dict[str, np.ndarray]:
    """imgs: (N, H, W) uint8. Returns named feature blocks."""
    x = imgs.reshape(len(imgs), -1).astype(np.float32)
    hist = np.stack([np.bincount((r // (256 // HIST_BINS)).astype(int), minlength=HIST_BINS)
                     for r in x]).astype(np.float32) / x.shape[1]
    stats = np.c_[x.mean(1), x.std(1), skew(x, axis=1), kurtosis(x, axis=1),
                  np.percentile(x, [1, 5, 50, 95, 99], axis=1).T,
                  (x == 0).mean(1), (x == 255).mean(1)]
    fft = fft_features(imgs, 64)[:, :FFT_LOW]
    marker = np.stack([_corner_fracs(im) for im in imgs])
    return {"hist": hist, "stats": stats, "fft_low": fft, "marker": marker}


def audit_auc(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits=5, n_boot=1000,
              seed=0) -> dict:
    """Out-of-fold AUC (GroupKFold) with a group-level bootstrap 95% CI."""
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits).split(X, y, groups):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y, oof)
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(groups, return_inverse=True)
    idx_by_g = [np.flatnonzero(inv == g) for g in range(len(ug))]
    boots = []
    for _ in range(n_boot):
        pick = np.concatenate([idx_by_g[g] for g in rng.integers(0, len(ug), len(ug))])
        if len(np.unique(y[pick])) == 2:
            boots.append(roc_auc_score(y[pick], oof[pick]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"auc": float(auc), "ci_lo": float(lo), "ci_hi": float(hi), "n0": int((y == 0).sum()),
            "n1": int((y == 1).sum())}
