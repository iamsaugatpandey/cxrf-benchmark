"""Phase 6, confirmatory: frozen-encoder detectors over the preregistered factorial grid.
Preregistration: https://osf.io/5vwkj (registered 2026-11-09), Design plan + Measured variables.

Grid
  encoder   : dinov3, rad-dino, biomedclip            (embeddings from embed_all.py)
  detector  : mlp, mlp_lbr (5 seeds each), knn, ocsvm (deterministic; seed 0)
  hospital  : nih_only | loho_nih, loho_openi, loho_shenzhen, loho_chexpert
              (loho_X = pooled training reals of {NIH, Open-I, Shenzhen, CheXpert} minus X,
               plus NODE21 training images as auxiliary reals; Montgomery is never trained on)
  generator : g1 (SynthCheX), g2 (MONAI), g12 (both)  -- training-split fakes only.
              G4/G5 and all image-conditioned fakes are never trained on.
Threshold   : 99th percentile of scores on VALIDATION reals of the named training hospitals
              (NODE21 auxiliary reals excluded from the threshold set), frozen per fit.
Imbalance   : all training reals are kept; the MLP loss weights fakes by n_real / n_fake.

Saved (one file per encoder x detector x hospital x generator, all seeds inside):
  results/scores/embed/<encoder>/<detector>/<hospital>__<generator>.parquet
      columns: seed, image_id, split (val|test), score, thr, flag
      rows: every included val and test image of every source (scores are kept for the
            val split too, for sensitivity analysis 6: threshold from pooled reals)
  results/scores/embed/runs.csv   one row per fit: config, n_train_real, n_train_fake,
                                       n_thr, thr, seconds, data-freeze sha
Resumable: existing parquet files are skipped.
Pixel baselines: --encoders pix_fft,pix_format --detectors logit (features from compute_pixel_features.py).
Usage: python scripts/run_embed_grid.py [--smoke]   (--smoke: one small fit, val only, no file writes)
"""
import argparse
import csv
import hashlib
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cxrf.detectors import KNNDetector, LBRDetector, OCSVMDetector
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cxrf.metrics import threshold_at_fpr

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
OUT = ROOT / "results/scores/embed"
FREEZE = "acbb483e7568d1bd1286c1a46fc7d34bee21793e8967aeff1408da6e1342c589"

ENCODERS = ["dinov3", "rad-dino", "biomedclip"]
DETECTORS = {"mlp": 5, "mlp_lbr": 5, "knn": 1, "ocsvm": 1, "logit": 1}  # -> number of seeds
# "logit" = the pixel baselines: run with --encoders pix_fft,pix_format --detectors logit
HOSPITALS = {"nih_only": ["nih"]}
CORE = ["nih", "openi", "shenzhen", "chexpert"]
for h in CORE:
    HOSPITALS[f"loho_{h}"] = [x for x in CORE if x != h]
GENERATORS = {"g1": ["synthchex"], "g2": ["monai_ldm"], "g12": ["synthchex", "monai_ldm"]}

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--encoders", default=",".join(ENCODERS))
ap.add_argument("--detectors", default="mlp,mlp_lbr,knn,ocsvm")
a = ap.parse_args()

sha = hashlib.sha256((D / "meta/images.csv").read_bytes()).hexdigest()
assert sha == FREEZE, f"images.csv changed since the data freeze ({sha[:12]})"
meta = pd.read_csv(D / "meta/images.csv", keep_default_na=False, dtype=str)
meta = meta[meta.exclude == ""].set_index("image_id")


def load(enc):
    ids = (D / "embeddings" / enc / "ids.txt").read_text().split()
    E = np.load(D / "embeddings" / enc / "emb.npy")
    pos = {k: i for i, k in enumerate(ids)}
    keep = [i for i in meta.index if i in pos]
    if not enc.endswith("__512"):  # S4 (512 px) excludes Open-I by design; every other set must be complete
        assert len(keep) == len(meta), f"{enc}: {len(meta) - len(keep)} included images lack embeddings"
    return E[[pos[i] for i in keep]].astype(np.float32), meta.loc[keep]


def train_set(m, hosp, gen):
    real_src = HOSPITALS[hosp] + ([] if hosp == "nih_only" else ["node21"])
    tr = m.split == "train"
    r = tr & m.source.isin(real_src) & (m.cls == "real")
    f = tr & m.source.isin(GENERATORS[gen])
    thr = (m.split == "val") & m.source.isin(HOSPITALS[hosp]) & (m.cls == "real")
    return r.values, f.values, thr.values


def make(det, seed):
    if det == "mlp":
        return LBRDetector(use_lbr=False, seed=seed, balance=True)
    if det == "mlp_lbr":
        return LBRDetector(use_lbr=True, seed=seed, balance=True)
    if det == "knn":
        return KNNDetector(k=10)
    if det == "logit":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced"))
    return OCSVMDetector(nu=0.05)


OUT.mkdir(parents=True, exist_ok=True)
runs_f = OUT / "runs.csv"
for enc in a.encoders.split(","):
    X, m = load(enc)
    evalmask = m.split.isin(["val", "test"]).values
    if a.smoke:  # code check on train/val only: nothing from the test split is scored
        evalmask = (m.split == "val").values
    ev_ids, ev_split = m.index[evalmask].values, m.split[evalmask].values
    ocsvm_cache = {}
    for det, hosp, gen in itertools.product(a.detectors.split(","), HOSPITALS, GENERATORS):
        dst = OUT / enc / det / f"{hosp}__{gen}.parquet"
        if dst.exists() and not a.smoke:
            continue
        r, f, thr_mask = train_set(m, hosp, gen)
        Xtr = np.r_[X[r], X[f]]
        ytr = np.r_[np.zeros(r.sum(), int), np.ones(f.sum(), int)]
        parts = []
        for seed in range(DETECTORS[det]):
            t = time.time()
            if det == "ocsvm" and hosp in ocsvm_cache:  # reals-only model: same fit for every generator
                s = ocsvm_cache[hosp]
            else:
                if a.smoke:  # tiny fit, just exercises the code path
                    rng = np.random.default_rng(0)
                    idx = np.r_[rng.choice(np.flatnonzero(ytr == 0), 300, replace=False),
                                rng.choice(np.flatnonzero(ytr == 1), 300, replace=False)]
                    model = make(det, seed).fit(Xtr[idx], ytr[idx])
                else:
                    model = make(det, seed).fit(Xtr, ytr)
                s = (model.predict_proba(X[evalmask])[:, 1] if det == "logit"
                     else model.score(X[evalmask])).astype(np.float32)
                if det == "ocsvm":
                    ocsvm_cache[hosp] = s
            thr_scores = s[thr_mask[evalmask]]
            thr = threshold_at_fpr(thr_scores, 0.01)
            parts.append(pd.DataFrame({"seed": seed, "image_id": ev_ids, "split": ev_split,
                                       "score": s, "thr": thr, "flag": (s > thr).astype(np.int8)}))
            row = {"encoder": enc, "detector": det, "hospital": hosp, "generator": gen, "seed": seed,
                   "n_train_real": int(r.sum()), "n_train_fake": int(f.sum()), "n_thr": int(thr_mask.sum()),
                   "thr": thr, "seconds": round(time.time() - t, 1), "freeze_sha": sha[:12]}
            print(row, flush=True)
            if a.smoke:
                vr = s[thr_mask[evalmask]]
                assert np.isfinite(s).all() and abs(np.mean(vr > thr) - 0.01) < 0.01
                break
            new = not runs_f.exists()
            with open(runs_f, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(row))
                if new:
                    w.writeheader()
                w.writerow(row)
        if a.smoke:
            break
        dst.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(parts).to_parquet(dst, index=False)
    if a.smoke:
        print(f"SMOKE OK ({enc}): code path ran on train/val only")
