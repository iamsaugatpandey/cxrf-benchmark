"""Pilot: pooled-reals LBR vs no-LBR for each encoder (same images as pilot_lbr_pooled.py).
Usage: python scripts/pilot_encoders.py dinov3 rad-dino biomedclip"""
import csv, json, random, sys
from pathlib import Path
import numpy as np
from cxrf.detectors import LBRDetector
from cxrf.embed import Encoder
from cxrf.env import load_env
from cxrf.metrics import auc, fpr_at_threshold, threshold_at_fpr, tpr_at_threshold

load_env()
ROOT = Path(__file__).resolve().parents[1]; D = ROOT / "data"
rows = [r for r in csv.DictReader(open(D / "meta/images.csv")) if not r["exclude"] and r["split"] == "train"]
# identical selections to pilot_lbr.py / pilot_lbr_pooled.py
random.seed(1)
nih = random.sample([r for r in rows if r["source"] == "nih" and r["view"] == "PA"], 2200)
sx = [r for r in rows if r["source"] == "synthchex"]; random.shuffle(sx)
oi = random.sample([r for r in rows if r["source"] == "openi"], 200)
random.seed(2)
quota = {"nih": (1000, 300), "node21": (400, 120), "shenzhen": (250, 60), "montgomery": (50, 20)}
tr, cal = [], []
for s, (a, b) in quota.items():
    pick = random.sample([r for r in rows if r["source"] == s and r["view"] != "AP"], a + b)
    tr += pick[:a]; cal += pick[a:]
api = sorted((D / "pilot/api_256").glob("*.png"))
sets = {"real_tr": [D / r["out256"] for r in tr], "real_cal": [D / r["out256"] for r in cal],
        "sx_tr": [D / r["out256"] for r in sx[:1700]], "sx_te": [D / r["out256"] for r in sx[1700:]],
        "openi": [D / r["out256"] for r in oi], "api": api}
allres = {}
for name in sys.argv[1:]:
    cache = D / f"pilot/emb_pooled_{name}.npz"
    if cache.exists():
        Z = dict(np.load(cache))
    else:
        enc = Encoder(name)
        Z = {k: enc.embed(v) for k, v in sets.items()}
        np.savez(cache, **Z)
    Xtr = np.r_[Z["real_tr"], Z["sx_tr"]].astype(np.float32)
    ytr = np.r_[np.zeros(len(Z["real_tr"])), np.ones(len(Z["sx_tr"]))].astype(int)
    for use_lbr in (False, True):
        rs = []
        for seed in range(5):
            det = LBRDetector(use_lbr=use_lbr, seed=seed).fit(Xtr, ytr)
            s = {k: det.score(Z[k].astype(np.float32)) for k in ("real_cal", "openi", "api", "sx_te")}
            thr = threshold_at_fpr(s["real_cal"], 0.01)
            rs.append({"tpr_api": tpr_at_threshold(s["api"], thr), "tpr_synthchex": tpr_at_threshold(s["sx_te"], thr),
                       "fpr_openi": fpr_at_threshold(s["openi"], thr), "auc_openi_vs_api": auc(s["openi"], s["api"]),
                       "auc_openi_vs_synthchex": auc(s["openi"], s["sx_te"])})
        key = f"{name}/{'LBR' if use_lbr else 'noLBR'}"
        allres[key] = {k: (float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs]))) for k in rs[0]}
        print(key, {m: f"{x[0]:.3f}±{x[1]:.3f}" for m, x in allres[key].items()}, flush=True)
json.dump(allres, open(ROOT / "results/pilot_encoders.json", "w"), indent=1)
