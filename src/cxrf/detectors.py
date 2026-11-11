"""Detectors on frozen embeddings (Pipeline steps 5-6).

* LBRDetector: 2-layer MLP trained with latent blending (SimLBR-style).
  Plan hyperparameters: alpha ~ U(0.5, 0.8), Adam lr 1e-4, weight decay 1e-2,
  batch 500, 10 epochs, 5 seeds.
  Blending (verified 2026-11-08 against Dhakal et al., SimLBR, arXiv 2602.20412, Sec. 4.2
  and Sec. 12): each fake encountered in a batch is replaced by a blend with a random real,
      z_blend = alpha * z_real + (1 - alpha) * z_fake,  alpha ~ U(0.5, 0.8),  label = fake (1),
  head = 2-layer MLP with ReLU, BCE loss, Adam 1e-4, wd 1e-2, batch 500, <= 10 epochs.
  Defaults reproduce the paper; blend_label / blend_mode stay as arguments for ablations.
* Baselines: the same MLP without LBR (use_lbr=False), one-class SVM on reals, a kNN
  probe and FFT-spectrum logistic regression (pixel-level).

Convention throughout: label 1 = fake/tampered, label 0 = real; scores higher = more fake.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


class _MLP(nn.Module):
    def __init__(self, d: int, h: int = 512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class LBRDetector:
    def __init__(self, use_lbr=True, alpha=(0.5, 0.8), lr=1e-4, wd=1e-2, batch=500,
                 epochs=10, hidden=512, seed=0, blend_label=1, blend_mode="replace",
                 device="cpu", balance=False):
        self.__dict__.update(locals())
        del self.__dict__["self"]

    def fit(self, X: np.ndarray, y: np.ndarray):
        g = torch.Generator().manual_seed(self.seed)
        torch.manual_seed(self.seed)
        X = torch.as_tensor(X, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.float32)
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-6
        X = (X - self.mu_) / self.sd_
        reals = X[y == 0]
        self.model_ = _MLP(X.shape[1], self.hidden).to(self.device)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.wd)
        # balance=True: weight fakes by n_real/n_fake so every training real can be kept
        pw = torch.tensor(float((y == 0).sum() / max((y == 1).sum(), 1))) if self.balance else None
        lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
        n = len(X)
        for _ in range(self.epochs):
            for idx in torch.randperm(n, generator=g).split(self.batch):
                xb, yb = X[idx], y[idx]
                if self.use_lbr:
                    fake = yb == 1
                    k = int(fake.sum())
                    if k:
                        a = torch.empty(k, 1).uniform_(*self.alpha, generator=g)
                        r = reals[torch.randint(len(reals), (k,), generator=g)]
                        zb = a * r + (1 - a) * xb[fake]
                        lb = torch.full((k,), float(self.blend_label))
                        if self.blend_mode == "replace":
                            xb, yb = xb.clone(), yb.clone()
                            xb[fake], yb[fake] = zb, lb
                        else:  # "add"
                            xb, yb = torch.cat([xb, zb]), torch.cat([yb, lb])
                opt.zero_grad()
                loss = lossf(self.model_(xb.to(self.device)), yb.to(self.device))
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def score(self, X: np.ndarray) -> np.ndarray:
        X = (torch.as_tensor(X, dtype=torch.float32) - self.mu_) / self.sd_
        return torch.sigmoid(self.model_(X.to(self.device))).cpu().numpy()


class OCSVMDetector:
    """Trained on reals only; score = negative decision function (higher = more fake)."""

    def __init__(self, nu=0.05, gamma="scale"):
        self.m = make_pipeline(StandardScaler(), OneClassSVM(nu=nu, gamma=gamma))

    def fit(self, X, y):
        self.m.fit(X[y == 0])
        return self

    def score(self, X):
        return -self.m.decision_function(X)


class KNNDetector:
    """Score = mean cosine distance to k nearest reals minus to k nearest fakes."""

    def __init__(self, k=10):
        self.k = k

    def fit(self, X, y):
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        self.nr = NearestNeighbors(n_neighbors=self.k, metric="cosine").fit(Xn[y == 0])
        self.nf = NearestNeighbors(n_neighbors=self.k, metric="cosine").fit(Xn[y == 1])
        return self

    def score(self, X):
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        dr = self.nr.kneighbors(Xn)[0].mean(1)
        df = self.nf.kneighbors(Xn)[0].mean(1)
        return dr - df


def fft_features(imgs: np.ndarray, n_bins: int = 64) -> np.ndarray:
    """Azimuthally averaged log power spectrum of (N, H, W) uint8 images."""
    N, H, W = imgs.shape
    yy, xx = np.indices((H, W))
    r = np.hypot(yy - H / 2, xx - W / 2)
    bins = np.minimum((r / r.max() * n_bins).astype(int), n_bins - 1)
    counts = np.bincount(bins.ravel(), minlength=n_bins)
    out = np.empty((N, n_bins), np.float32)
    for i, im in enumerate(imgs.astype(np.float32) / 255.0):
        p = np.abs(np.fft.fftshift(np.fft.fft2(im - im.mean()))) ** 2
        out[i] = np.log1p(np.bincount(bins.ravel(), p.ravel(), n_bins) / counts)
    return out


class FFTDetector:
    """Pixel-level baseline: logistic regression on the radial spectrum."""

    def __init__(self, n_bins=64):
        self.n_bins = n_bins
        self.m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))

    def fit(self, imgs, y):
        self.m.fit(fft_features(imgs, self.n_bins), y)
        return self

    def score(self, imgs):
        return self.m.predict_proba(fft_features(imgs, self.n_bins))[:, 1]
