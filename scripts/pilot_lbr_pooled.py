"""Pilot variant: pooled multi-site reals in training (Open-I still held out).
Train reals: 1,700 = NIH PA 1,000 + NODE21 400 + Shenzhen 250 + Montgomery 50 (train split).
Calibration reals: 500 pooled the same way (disjoint). Everything else as pilot_lbr.py.
Usage: python scripts/pilot_lbr_pooled.py"""
import csv, json, random
from pathlib import Path
import numpy as np
from cxrf.detectors import LBRDetector
from cxrf.embed import Encoder
from cxrf.env import load_env
from cxrf.metrics import auc, fpr_at_threshold, threshold_at_fpr, tpr_at_threshold

load_env()
ROOT = Path(__file__).resolve().parents[1]; D = ROOT / "data"
Z = dict(np.load(D / "pilot/emb_dinov3.npz"))
rows = [r for r in csv.DictReader(open(D / "meta/images.csv")) if not r["exclude"] and r["split"] == "train"]
random.seed(2)
quota = {"nih": (1000, 300), "node21": (400, 120), "shenzhen": (250, 60), "montgomery": (50, 20)}
tr, cal = [], []
for s, (a, b) in quota.items():
    pool = [r for r in rows if r["source"] == s and r["view"] != "AP"]
    pick = random.sample(pool, a + b)
    tr += pick[:a]; cal += pick[a:]
cache = D / "pilot/emb_dinov3_pooled.npz"
if cache.exists():
    P = dict(np.load(cache))
else:
    enc = Encoder("dinov3")
    P = {"real_tr": enc.embed([D / r["out256"] for r in tr]), "real_cal": enc.embed([D / r["out256"] for r in cal])}
    np.savez(cache, **P)
Xtr = np.r_[P["real_tr"], Z["sx_tr"]].astype(np.float32)
ytr = np.r_[np.zeros(len(P["real_tr"])), np.ones(len(Z["sx_tr"]))].astype(int)
res = {}
for use_lbr in (False, True):
    rs = []
    for seed in range(5):
        det = LBRDetector(use_lbr=use_lbr, seed=seed).fit(Xtr, ytr)
        s = {k: det.score(v.astype(np.float32)) for k, v in
             {"cal": P["real_cal"], "openi": Z["openi"], "api": Z["api"], "sx_te": Z["sx_te"]}.items()}
        thr = threshold_at_fpr(s["cal"], 0.01)
        rs.append({"tpr_api": tpr_at_threshold(s["api"], thr), "tpr_synthchex": tpr_at_threshold(s["sx_te"], thr),
                   "fpr_openi": fpr_at_threshold(s["openi"], thr), "auc_openi_vs_api": auc(s["openi"], s["api"]),
                   "auc_openi_vs_synthchex": auc(s["openi"], s["sx_te"])})
    res["LBR" if use_lbr else "noLBR"] = {k: (float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs]))) for k in rs[0]}
for k, v in res.items():
    print("pooled", k, {m: f"{x[0]:.3f}±{x[1]:.3f}" for m, x in v.items()})
json.dump(res, open(ROOT / "results/pilot_lbr_pooled.json", "w"), indent=1)
