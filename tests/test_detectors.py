import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cxrf.detectors import FFTDetector, KNNDetector, LBRDetector, OCSVMDetector  # noqa: E402
from cxrf.metrics import auc, threshold_at_fpr, tpr_at_threshold  # noqa: E402


def _toy(n=600, d=32, shift=1.5, seed=0):
    rng = np.random.default_rng(seed)
    real = rng.standard_normal((n, d))
    fake = rng.standard_normal((n, d))
    fake[:, 0] += shift
    return np.r_[real, fake].astype(np.float32), np.r_[np.zeros(n), np.ones(n)].astype(int)


def test_embedding_detectors_separate_toy_data():
    X, y = _toy()
    Xt, yt = _toy(seed=1)
    for det in (LBRDetector(epochs=30, lr=1e-3), LBRDetector(use_lbr=False, epochs=30, lr=1e-3),
                KNNDetector()):
        s = det.fit(X, y).score(Xt)
        assert auc(s[yt == 0], s[yt == 1]) > 0.7, type(det).__name__


def test_ocsvm_flags_out_of_distribution():
    # one-class: only sees reals, so needs fakes that are novel in many directions
    rng = np.random.default_rng(0)
    real = rng.standard_normal((600, 32))
    fake = rng.standard_normal((600, 32)) * 1.8
    X, y = np.r_[real, fake], np.r_[np.zeros(600), np.ones(600)].astype(int)
    s = OCSVMDetector().fit(X, y).score(X)
    assert auc(s[y == 0], s[y == 1]) > 0.8


def test_lbr_seed_is_deterministic():
    X, y = _toy(n=200)
    a = LBRDetector(epochs=2, seed=3).fit(X, y).score(X)
    b = LBRDetector(epochs=2, seed=3).fit(X, y).score(X)
    assert np.allclose(a, b)


def test_fft_detector_finds_spectral_artifact():
    rng = np.random.default_rng(0)
    real = rng.integers(60, 200, (100, 64, 64)).astype(np.uint8)
    fake = real.copy()
    fake[:, ::8, :] = np.clip(fake[:, ::8, :].astype(int) + 40, 0, 255)  # periodic grid
    imgs, y = np.r_[real, fake], np.r_[np.zeros(100), np.ones(100)].astype(int)
    s = FFTDetector().fit(imgs, y).score(imgs)
    assert auc(s[y == 0], s[y == 1]) > 0.9


def test_threshold_controls_fpr():
    rng = np.random.default_rng(0)
    real_val = rng.standard_normal(10000)
    thr = threshold_at_fpr(real_val, 0.01)
    assert abs(np.mean(rng.standard_normal(10000) > thr) - 0.01) < 0.005
    assert tpr_at_threshold(np.full(10, thr + 1), thr) == 1.0
