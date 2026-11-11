"""Exploratory H1 analysis (added after the main analyses, not preregistered): real test images WITHOUT reported disease only,
to check whether the hospital false-positive increase reflects patient population (disease mix) rather than
acquisition. Same "normal" rules as the source radiographs of the image-conditioned arm (scripts/gen_api.py):
  NIH        : NIH label 'No Finding'
  Open-I     : impression mentions 'normal' or 'no acute'
  CheXpert   : impression mentions 'normal' or 'no acute' (keyword proxy; CheXpert Plus has no label file here)
  Shenzhen, Montgomery : tuberculosis label 0
  JSRT, PadChest       : excluded (NODE21 contributed only nodule cases)
Output: data/analysis/sens/N_reals_cells.parquet (schema of reals_cells.parquet) and
        results/tables/normals_counts.csv (images per site). Usage: python analysis/normals_cells.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "data/analysis"
m = pd.read_csv(ROOT / "data/meta/images.csv", dtype=str, keep_default_na=False)
m = m[(m.cls == "real") & (m.exclude == "")]
ex = m.extra.map(lambda e: json.loads(e) if e.startswith("{") else {})
imp = ex.map(lambda e: str(e.get("impression", "")).lower())
normal = (((m.source == "nih") & ex.map(lambda e: e.get("findings") == "No Finding"))
          | (m.source.isin(["openi", "chexpert"]) & (imp.str.contains("normal") | imp.str.contains("no acute")))
          | (m.source.isin(["shenzhen", "montgomery"]) & ex.map(lambda e: e.get("tb") == 0)))
ids = set(m.image_id[normal])
cnt = pd.DataFrame({"all": m.groupby("source").size(), "normal": m[normal].groupby("source").size()}).fillna(0)
cnt["test_normal"] = m[normal & (m.split == "test")].groupby("source").size()
cnt.to_csv(ROOT / "results/tables/normals_counts.csv")
print(cnt)
KEYS = ["site", "view", "family", "encoder", "config", "seed", "novelty_site"]
R = pd.read_parquet(A / "reals_long.parquet", columns=KEYS + ["image_id", "flag"])
ref = pd.read_parquet(A / "reals_cells.parquet")
R = R[R.image_id.isin(ids) & R.site.isin(ref.site.unique())]
C = R.groupby(KEYS, observed=True).flag.agg(k="sum", n="size").reset_index()
C.to_parquet(A / "sens/N_reals_cells.parquet", index=False)
print(len(C), "cells;", int(C.n.sum()), "decisions")
