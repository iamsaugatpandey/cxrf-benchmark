"""Freeze the preregistered LLM evaluation subset (OSF 5vwkj, Measured variables / H5).
No API calls. Deterministic (seed 20261109). CheXpert is excluded (data-use agreement).

Strata (test split only):
  reals: NIH 400 (PA 240 / AP 160), Open-I 300, Shenzhen 148, NODE21-PadChest 76,
         NODE21-JSRT 41, Montgomery 27, NODE21 undocumented-origin controls 8
  fakes: G1 SynthCheX 200, G2 MONAI 200, G4 text 150, G5 text 150, G4 cond 150, G5 cond 150
Order: within each stratum a random rank r; the run order is by (r + U) / n_stratum, so any
prefix (a budget-truncated run) stays stratum-balanced.
Subsets: retest = items with r < 20% of their stratum (≈400); fourshot = r < 25% (≈500).
4-shot examples (fixed, training split): 2 reals (NIH PA, Open-I) + 2 fakes (G1, G2).
Output: data/meta/llm_subset.csv, data/meta/llm_fewshot_examples.csv
Usage: python scripts/freeze_llm_subset.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
SEED = 20261109
m = pd.read_csv(D / "meta/images.csv", keep_default_na=False, dtype=str)
m = m[m.exclude == ""]
m["n21"] = np.where(m.source == "node21", m.extra.map(lambda e: json.loads(e).get("orig_dataset", "") if e else ""), "")
m["cond"] = m.image_id.str.contains(":cond_")
t = m[m.split == "test"]

S = {
    "nih_PA": (t.source == "nih") & (t.view == "PA"), "nih_AP": (t.source == "nih") & (t.view == "AP"),
    "openi": t.source == "openi", "shenzhen": t.source == "shenzhen", "montgomery": t.source == "montgomery",
    "node21_padchest": t.n21 == "padchest", "node21_jsrt": t.n21 == "jsrt", "node21_unknown": t.n21 == "unknown",
    "g1": t.source == "synthchex", "g2": t.source == "monai_ldm",
    "g4_text": (t.source == "gpt_image") & ~t.cond, "g5_text": (t.source == "gemini_image") & ~t.cond,
    "g4_cond": (t.source == "gpt_image") & t.cond, "g5_cond": (t.source == "gemini_image") & t.cond,
}
N = {"nih_PA": 240, "nih_AP": 160, "openi": 300, "shenzhen": 148, "montgomery": 27, "node21_padchest": 76,
     "node21_jsrt": 41, "node21_unknown": 8, "g1": 200, "g2": 200, "g4_text": 150, "g5_text": 150,
     "g4_cond": 150, "g5_cond": 150}
rng = np.random.default_rng(SEED)
parts = []
for k, mask in S.items():
    pool = t[mask].sort_values("image_id")
    assert len(pool) >= N[k], f"{k}: only {len(pool)}"
    pick = pool.iloc[rng.permutation(len(pool))[:N[k]]].copy()
    pick["stratum"], pick["rank"] = k, np.arange(N[k])
    pick["order_key"] = (pick["rank"] + rng.uniform(size=N[k])) / N[k]
    pick["retest"] = pick["rank"] < np.ceil(0.20 * N[k])
    pick["fourshot"] = pick["rank"] < np.ceil(0.25 * N[k])
    parts.append(pick)
sub = pd.concat(parts).sort_values("order_key").reset_index(drop=True)
sub["run_order"] = np.arange(len(sub))
cols = ["run_order", "image_id", "stratum", "cls", "source", "site", "view", "retest", "fourshot", "raw_path", "out256"]
sub[cols].to_csv(D / "meta/llm_subset.csv", index=False)
assert len(sub) == 2000 and (sub.cls == "real").sum() == 1000 and not (sub.source == "chexpert").any()

tr = m[m.split == "train"]
ex = []
for k, mask in {"real_nih": (tr.source == "nih") & (tr.view == "PA"), "real_openi": tr.source == "openi",
                "fake_g1": tr.source == "synthchex", "fake_g2": tr.source == "monai_ldm"}.items():
    pool = tr[mask].sort_values("image_id")
    r = pool.iloc[rng.integers(len(pool))]
    ex.append({"slot": k, "image_id": r.image_id, "cls": r.cls, "out256": r.out256})
pd.DataFrame(ex).to_csv(D / "meta/llm_fewshot_examples.csv", index=False)
print(sub.groupby("stratum").agg(n=("image_id", "size"), retest=("retest", "sum"), fourshot=("fourshot", "sum")))
print("total", len(sub), "| retest", sub.retest.sum(), "| fourshot", sub.fourshot.sum())
print(pd.DataFrame(ex))
