"""Export pilot flag counts for the figure-style demo (PILOT ONLY, not paper results).
Uses cached embeddings from pilot_encoders.py. Threshold: 1% FPR on even-indexed held-out
seen-hospital reals; seen FPR evaluated on odd-indexed ones. Seed 0 only.
Output: results/pilot_demo_counts.csv (encoder, lbr, set, k, n)."""
import csv
from pathlib import Path
import numpy as np
from cxrf.detectors import LBRDetector
from cxrf.metrics import threshold_at_fpr

ROOT = Path(__file__).resolve().parents[1]; D = ROOT / "data/pilot"
rows = []
for enc in ("dinov3", "rad-dino", "biomedclip"):
    Z = dict(np.load(D / f"emb_pooled_{enc}.npz"))
    X = np.r_[Z["real_tr"], Z["sx_tr"]].astype(np.float32)
    y = np.r_[np.zeros(len(Z["real_tr"])), np.ones(len(Z["sx_tr"]))].astype(int)
    for lbr in (False, True):
        det = LBRDetector(use_lbr=lbr, seed=0).fit(X, y)
        cal = det.score(Z["real_cal"].astype(np.float32))
        thr = threshold_at_fpr(cal[0::2], 0.01)
        sets = {"Seen hospitals (held-out)": (cal[1::2], 0), "Open-I (unseen hospital)": (det.score(Z["openi"].astype(np.float32)), 0),
                "SynthCheX (seen generator)": (det.score(Z["sx_te"].astype(np.float32)), 1),
                "GPT-image + Gemini (unseen)": (det.score(Z["api"].astype(np.float32)), 1)}
        for name, (s, lab) in sets.items():
            rows.append({"encoder": enc, "lbr": "LBR" if lbr else "No LBR", "set": name,
                         "kind": "real" if lab == 0 else "fake", "k": int((s > thr).sum()), "n": len(s)})
out = ROOT / "results/pilot_demo_counts.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(open(out).read())
