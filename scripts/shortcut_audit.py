"""Run the shortcut audit on preprocessed 256-px images.

Usage: python shortcut_audit.py <data_root> <out_csv> [--max-per-source 3000]
Pairs: every fake source vs every real source and vs pooled reals (the gate), plus every
real-vs-real site pair (reference: how much format alone identifies a hospital).
Each pair is run with all features and without the marker block.
"""
import argparse
import csv
import itertools
from pathlib import Path

import numpy as np
from PIL import Image

from cxrf.audit import audit_auc, features

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("out")
ap.add_argument("--max-per-source", type=int, default=3000)
args = ap.parse_args()
root = Path(args.root)

inv = {r["image_id"]: r for r in csv.DictReader(open(root / "meta/inventory_raw.csv"))}
qc = [r for r in csv.DictReader(open(root / "meta/preprocess_qc.csv"))
      if not r["error"] and float(r["out_std"]) >= 10.0]
rng = np.random.default_rng(0)
by_src = {}
for r in qc:
    by_src.setdefault(r["source"], []).append(r)
data = {}
for s, rows in sorted(by_src.items()):
    if len(rows) > args.max_per_source:
        rows = [rows[i] for i in sorted(rng.choice(len(rows), args.max_per_source, replace=False))]
    imgs = np.stack([np.asarray(Image.open(root / r["out256"])) for r in rows])
    F = features(imgs)
    groups = np.array([inv[r["image_id"]]["group_id"] for r in rows])
    data[s] = (F, groups, inv[rows[0]["image_id"]]["cls"])
    print(s, len(rows), data[s][2], flush=True)

reals = [s for s, v in data.items() if v[2] == "real"]
fakes = [s for s, v in data.items() if v[2] != "real"]
pairs = [(r, f) for f in fakes for r in reals] + [("ALL_REAL", f) for f in fakes] \
    + list(itertools.combinations(reals, 2))


def stack(src):
    if src == "ALL_REAL":  # pooled reals, equal-sized subsample per site
        k = min(len(data[s][1]) for s in reals)
        parts = []
        for s in reals:
            i = rng.choice(len(data[s][1]), k, replace=False)
            parts.append(({b: v[i] for b, v in data[s][0].items()}, data[s][1][i]))
        return ({b: np.concatenate([p[0][b] for p in parts]) for b in parts[0][0]},
                np.concatenate([p[1] for p in parts]))
    return data[src][0], data[src][1]


out = []
for a, b in pairs:
    (Fa, ga), (Fb, gb) = stack(a), stack(b)
    y = np.r_[np.zeros(len(ga)), np.ones(len(gb))].astype(int)
    g = np.r_[ga, gb]
    for name, blocks in [("all", ["hist", "stats", "fft_low", "marker"]),
                         ("no_marker", ["hist", "stats", "fft_low"]), ("marker_only", ["marker"])]:
        X = np.c_[tuple(np.r_[Fa[k], Fb[k]] for k in blocks)]
        res = audit_auc(X, y, g)
        kind = "real_vs_fake" if b in fakes else "real_vs_real"
        out.append({"kind": kind, "a": a, "b": b, "features": name, **res})
        print(f"{kind:12s} {a:10s} vs {b:10s} {name:11s} AUC {res['auc']:.3f} "
              f"[{res['ci_lo']:.3f}, {res['ci_hi']:.3f}]", flush=True)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
