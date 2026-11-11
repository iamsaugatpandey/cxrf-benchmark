"""Content-based near-duplicate check: NODE21 vs our NIH + Open-I frontal images.

Each preprocessed 256-px image -> 32x32 area-downsampled thumbnail -> z-scored vector;
similarity = Pearson correlation. For each NODE21 image, the best match among NIH/Open-I.
Known overlaps (NODE21 nodule images named as NIH originals we hold) validate the threshold.
Usage: python dedup_node21.py <data_root> [threshold=0.98]
Writes data/meta/node21_dups.csv: node21 id, best match id, corr, known_match.
"""
import csv, json, sys
from pathlib import Path
import numpy as np
from PIL import Image

root = Path(sys.argv[1]); thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.98
inv = {r["image_id"]: r for r in csv.DictReader(open(root / "meta/inventory_raw.csv"))}
qc = {r["image_id"]: r for r in csv.DictReader(open(root / "meta/preprocess_qc.csv")) if not r["error"]}


def thumbs(ids):
    X = np.stack([np.asarray(Image.open(root / qc[i]["out256"]).resize((32, 32), Image.Resampling.BOX),
                             np.float32).ravel() for i in ids])
    X -= X.mean(1, keepdims=True)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-6
    return X


ref_ids = [i for i in qc if inv[i]["source"] in ("nih", "openi")]
n21_ids = [i for i in qc if inv[i]["source"] == "node21"]
R, N = thumbs(ref_ids), thumbs(n21_ids)
best, arg = np.empty(len(N)), np.empty(len(N), int)
for s in range(0, len(N), 512):
    S = N[s:s + 512] @ R.T
    arg[s:s + 512] = S.argmax(1); best[s:s + 512] = S.max(1)
rows = []
for k, nid in enumerate(n21_ids):
    ex = json.loads(inv[nid]["extra"])
    known = ""
    if ex["orig_dataset"] == "chestxray14" and f"nih:{ex['orig_name']}" in inv:
        known = f"nih:{ex['orig_name']}"
    elif ex["orig_dataset"] == "openi" and f"openi:CXR{ex['orig_name']}" in inv:
        known = f"openi:CXR{ex['orig_name']}"
    rows.append({"node21_id": nid, "orig_dataset": ex["orig_dataset"], "best_match": ref_ids[arg[k]],
                 "corr": f"{best[k]:.4f}", "known_match": known,
                 "dup": int(best[k] >= thr)})
with open(root / "meta/node21_dups.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
kn = [r for r in rows if r["known_match"]]
print(f"known overlaps: {len(kn)}; best match == known: {sum(r['best_match'] == r['known_match'] for r in kn)}; "
      f"their corr min/median: {min(float(r['corr']) for r in kn):.4f}/{np.median([float(r['corr']) for r in kn]):.4f}")
unk = np.array([float(r["corr"]) for r in rows if not r["known_match"]])
print("unknown-origin corr histogram [0,.8,.9,.95,.97,.98,.99,1]:", np.histogram(unk, [0, .8, .9, .95, .97, .98, .99, 1.01])[0])
print(f"flagged dup (corr >= {thr}):", sum(r["dup"] for r in rows),
      "by best-match source:", {s: sum(r["dup"] and r["best_match"].startswith(s) for r in rows) for s in ("nih", "openi")})
