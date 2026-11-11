"""Preregistered sensitivity analyses that need no new detector runs (OSF 5vwkj, 4.6):
  S1 PA-only            : reals cells restricted to PA views.
  S6 pooled threshold   : per fit, threshold = 99th percentile of validation scores of ALL real sources
                          (not only the training hospitals); flags recomputed on test images.
  S7 no memorized fakes : fakes whose 32x32 standardized thumbnail correlates r >= 0.98 with any real image
                          (any split) are dropped (memorization list saved).
  S9 G4/G5 text in test : commercial text fakes restricted to those whose Open-I prompt report is in the test split.
Outputs: data/analysis/sens/<S>_{reals,fakes}_cells.parquet (same schema as the primary cells), consumed by
fit_h1_cells.R / fit_fakes_cells.R with an input-path argument; data/analysis/sens/memorized_fakes.csv.
Usage: python analysis/sensitivity_cells.py
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from cxrf.splits import split_of

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
A = D / "analysis"
OUT = A / "sens"
OUT.mkdir(parents=True, exist_ok=True)
RC = pd.read_parquet(A / "reals_cells.parquet")
FC = pd.read_parquet(A / "fakes_cells.parquet")
KEYS_R = ["site", "view", "family", "encoder", "config", "seed", "novelty_site"]
KEYS_F = ["generator", "conditioning", "cond_site", "family", "encoder", "config", "seed", "novelty_gen"]

# S1
RC[RC.view == "PA"].to_parquet(OUT / "S1_reals_cells.parquet", index=False)

# S6: pooled-real thresholds
m = pd.read_csv(D / "meta/images.csv", keep_default_na=False, dtype=str)
real_ids = set(m.image_id[(m.exclude == "") & (m.cls == "real")])
R = pd.read_parquet(A / "reals_long.parquet", columns=KEYS_R[:-1] + ["novelty_site", "image_id", "score"])
F = pd.read_parquet(A / "fakes_long.parquet", columns=KEYS_F + ["image_id", "score"])
thr = {}
files = [(f, *Path(f).parts[-3:-1]) for f in glob.glob(str(ROOT / "results/scores/embed/*/*/*.parquet"))
         if "__" not in Path(f).parts[-3]]
files += [(f, "none", "resnet") for f in glob.glob(str(ROOT / "results/scores/resnet/resnet18/*.parquet"))]
for f, enc, det in files:
    d = pd.read_parquet(f, columns=["seed", "image_id", "split", "score"])
    v = d[(d.split == "val") & d.image_id.isin(real_ids)]
    fam = {"logit": "fft" if enc == "pix_fft" else "format"}.get(det, det)
    e = "none" if enc.startswith("pix_") else enc
    hosp, gen = Path(f).stem.split("__")
    for seed, g in v.groupby("seed"):
        thr[(f"{fam}|{e}|{hosp}|{gen}", seed)] = float(np.quantile(g.score, 0.99, method="higher"))
t = pd.Series(thr, name="thr6")
t.index.names = ["config", "seed"]
for kind, X, keys in [("reals", R, KEYS_R), ("fakes", F, KEYS_F)]:
    X = X.join(t, on=["config", "seed"])
    X["flag"] = (X.score > X.thr6).astype(int)
    X.groupby(keys, observed=True).flag.agg(k="sum", n="size").reset_index() \
        .to_parquet(OUT / f"S6_{kind}_cells.parquet", index=False)
    del X

# S7: memorized fakes
inc = m[m.exclude == ""]


def thumbs(paths):
    T = np.stack([np.asarray(Image.open(D / p).resize((32, 32), Image.BILINEAR), np.float32).ravel() for p in paths])
    T -= T.mean(1, keepdims=True)
    return T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-6)


fk = inc[inc.cls == "fake"]
rl = inc[inc.cls == "real"]
Tf, Tr = thumbs(fk.out256), thumbs(rl.out256)
best = np.full(len(fk), -1.0, np.float32)
arg = np.zeros(len(fk), int)
for i in range(0, len(Tr), 8000):
    c = Tf @ Tr[i:i + 8000].T
    j = c.argmax(1)
    v = c[np.arange(len(fk)), j]
    upd = v > best
    best[upd], arg[upd] = v[upd], j[upd] + i
mem = pd.DataFrame({"image_id": fk.image_id.values, "max_r": best, "nearest_real": rl.image_id.values[arg]})
mem.to_csv(OUT / "memorized_fakes.csv", index=False)
memo = set(mem.image_id[mem.max_r >= 0.98])
print(f"S7: {len(memo)} of {len(fk)} fakes have r >= 0.98 to a real image;",
      mem.assign(src=fk.source.values)[mem.max_r >= 0.98].groupby("src").size().to_dict())
F7 = F[~F.image_id.isin(memo)].copy()
F7 = F7.merge(pd.read_parquet(A / "fakes_long.parquet", columns=["image_id", "config", "seed", "flag"]),
              on=["image_id", "config", "seed"])
F7.groupby(KEYS_F, observed=True).flag.agg(k="sum", n="size").reset_index().to_parquet(OUT / "S7_fakes_cells.parquet", index=False)
del F7

# S9: commercial text fakes whose prompt group (Open-I report) is in the test split
grp = m.set_index("image_id").group_id
FL = pd.read_parquet(A / "fakes_long.parquet", columns=KEYS_F + ["image_id", "flag"])
keep = ~((FL.conditioning == "text") & FL.generator.isin(["G4", "G5"])) | \
       FL.image_id.map(lambda i: split_of(grp[i]) == "test")
F9 = FL[keep]
print("S9: commercial text fakes kept per generator:",
      F9[(F9.conditioning == "text") & F9.generator.isin(["G4", "G5"])].groupby("generator").image_id.nunique().to_dict())
F9.groupby(KEYS_F, observed=True).flag.agg(k="sum", n="size").reset_index().to_parquet(OUT / "S9_fakes_cells.parquet", index=False)
print("wrote", OUT)
