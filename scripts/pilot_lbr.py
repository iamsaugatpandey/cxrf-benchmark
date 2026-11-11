"""Phase 2 pilot, step 5: first LBR test on the laptop (train-split data only).

Train : 1,700 NIH train PA reals + 1,700 SynthCheX (G1) train fakes, DINOv3 @ 256 px
Thresh: 500 other NIH train PA reals -> threshold at 1% FPR
Test  : 200 Open-I train frontal reals (unseen hospital) + 20 API fakes (G4/G5 pilot v2,
        unseen generators) + the remaining 368 SynthCheX train fakes (seen generator)
MLP with vs without LBR, 5 seeds. Also the format-feature logistic baseline on the same split.
Usage: python scripts/pilot_lbr.py
"""
import csv
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from cxrf.audit import features
from cxrf.detectors import LBRDetector
from cxrf.embed import Encoder
from cxrf.env import load_env
from cxrf.metrics import auc, fpr_at_threshold, threshold_at_fpr, tpr_at_threshold
from cxrf.preprocess import preprocess, save_png

load_env()
ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
OUT = D / "pilot"
rows = [r for r in csv.DictReader(open(D / "meta/images.csv")) if not r["exclude"] and r["split"] == "train"]
random.seed(1)
nih = random.sample([r for r in rows if r["source"] == "nih" and r["view"] == "PA"], 2200)
sx = [r for r in rows if r["source"] == "synthchex"]
random.shuffle(sx)
oi = random.sample([r for r in rows if r["source"] == "openi"], 200)

api = []
for gen in ("G4", "G5"):
    for f in sorted((D / "raw/api" / gen / "pilot_v2").glob("pilot*.*")):
        dst = OUT / "api_256" / f"{gen}_{f.stem}.png"
        if not dst.exists():
            save_png(preprocess(f, size=256), dst)
        api.append(dst)

sets = {"nih_tr": [D / r["out256"] for r in nih[:1700]], "nih_cal": [D / r["out256"] for r in nih[1700:]],
        "sx_tr": [D / r["out256"] for r in sx[:1700]], "sx_te": [D / r["out256"] for r in sx[1700:]],
        "openi": [D / r["out256"] for r in oi], "api": api}
cache = OUT / "emb_dinov3.npz"
if cache.exists():
    Z = dict(np.load(cache))
else:
    enc = Encoder("dinov3")
    Z = {k: enc.embed(v) for k, v in sets.items()}
    np.savez(cache, **Z)
print({k: v.shape for k, v in Z.items()})

Xtr = np.r_[Z["nih_tr"], Z["sx_tr"]].astype(np.float32)
ytr = np.r_[np.zeros(1700), np.ones(1700)].astype(int)
res = {}
for use_lbr in (False, True):
    rs = []
    for seed in range(5):
        det = LBRDetector(use_lbr=use_lbr, seed=seed).fit(Xtr, ytr)
        s = {k: det.score(Z[k].astype(np.float32)) for k in ("nih_cal", "openi", "api", "sx_te")}
        thr = threshold_at_fpr(s["nih_cal"], 0.01)
        rs.append({"tpr_api": tpr_at_threshold(s["api"], thr),
                   "tpr_api_G4": tpr_at_threshold(s["api"][:10], thr),
                   "tpr_api_G5": tpr_at_threshold(s["api"][10:], thr),
                   "tpr_synthchex": tpr_at_threshold(s["sx_te"], thr),
                   "fpr_openi": fpr_at_threshold(s["openi"], thr),
                   "auc_openi_vs_api": auc(s["openi"], s["api"]),
                   "auc_openi_vs_synthchex": auc(s["openi"], s["sx_te"])})
    res["LBR" if use_lbr else "noLBR"] = {k: (float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs])))
                                         for k in rs[0]}

# format-feature baseline (logistic on histogram/stats/FFT/marker), same split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def F(paths):
    f = features(np.stack([np.asarray(Image.open(p)) for p in paths]))
    return np.c_[f["hist"], f["stats"], f["fft_low"], f["marker"]]


m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(
    np.r_[F(sets["nih_tr"]), F(sets["sx_tr"])], ytr)
s = {k: m.predict_proba(F(sets[k]))[:, 1] for k in ("nih_cal", "openi", "api", "sx_te")}
thr = threshold_at_fpr(s["nih_cal"], 0.01)
res["format_baseline"] = {"tpr_api": tpr_at_threshold(s["api"], thr), "tpr_synthchex": tpr_at_threshold(s["sx_te"], thr),
                          "fpr_openi": fpr_at_threshold(s["openi"], thr),
                          "auc_openi_vs_api": auc(s["openi"], s["api"]),
                          "auc_openi_vs_synthchex": auc(s["openi"], s["sx_te"])}
for k, v in res.items():
    print(k, {m: (f"{x[0]:.3f}±{x[1]:.3f}" if isinstance(x, tuple) else f"{x:.3f}") for m, x in v.items()})
(ROOT / "results").mkdir(exist_ok=True)
json.dump(res, open(ROOT / "results/pilot_lbr.json", "w"), indent=1)
