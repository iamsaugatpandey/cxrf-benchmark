"""Assemble per-image TEST-split detector outcomes into analysis tables (Phase 8).

Inputs : results/scores/embed/<enc>/<det>/<hosp>__<gen>.parquet   (frozen encoders + pixel 'logit')
         results/scores/resnet/resnet18/<hosp>__g12.parquet
         data/meta/images.csv (frozen, sha acbb483e…)
Coding (OSF 5vwkj, Transformations):
  family      : mlp (reference), mlp_lbr, knn, ocsvm, fft (pix_fft logit), format (pix_format logit), resnet
  encoder     : dinov3 | rad-dino | biomedclip | none (pixel / resnet)
  site        : nih, openi, shenzhen, chexpert, montgomery, jsrt, padchest  (NODE21 undocumented-origin
                controls and NODE21 copies of ChestX-ray14 are descriptive only -> site 'node21_other', not modelled)
  novelty_site: 0 = the site's training-split images were in the detector's training reals (seen), 1 = unseen.
                NODE21-derived sites (jsrt, padchest) are 'seen' in leave-one-hospital-out configs (auxiliary
                training reals) and 'unseen' in nih_only. Montgomery is always unseen.
  novelty_gen : 0 = generator in the training fakes, 1 = unseen (G4/G5 always unseen)
  view        : PA = 0, AP = 1 (reals); fakes have no view
Outputs (data/analysis/):
  reals_long.parquet  one row per real test image x fit x seed   (flag, score)
  fakes_long.parquet  one row per fake test image x fit x seed
  reals_agg.parquet   per image x family x encoder x novelty_site: k flags of n fits (binomial)
  fakes_agg.parquet   per image x family x encoder x novelty_gen: k of n
  reals_cells.parquet site x view x config x seed x novelty_site: k of n (primary H1 fit)
  fakes_cells.parquet generator x conditioning x cond_site x config x seed x novelty_gen: k of n
Usage: python analysis/build_tidy.py [mask|jpeg]   (variant -> data/analysis/sens/S2_*/S3_*)
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
OUT = D / "analysis"
OUT.mkdir(exist_ok=True)
CORE = ["nih", "openi", "shenzhen", "chexpert"]
TRAIN_REALS = {"nih_only": ["nih"], **{f"loho_{h}": [x for x in CORE if x != h] + ["node21"] for h in CORE}}
TRAIN_FAKES = {"g1": ["synthchex"], "g2": ["monai_ldm"], "g12": ["synthchex", "monai_ldm"]}
GEN = {"synthchex": "G1", "monai_ldm": "G2", "gpt_image": "G4", "gemini_image": "G5"}

m = pd.read_csv(D / "meta/images.csv", keep_default_na=False, dtype=str)
m = m[(m.exclude == "") & (m.split == "test")].copy()
n21 = m.extra.map(lambda e: json.loads(e).get("orig_dataset", "") if e.startswith("{") else "")
m["site"] = np.where(m.source != "node21", m.source,
                     np.select([n21 == "jsrt", n21 == "padchest"], ["jsrt", "padchest"], "node21_other"))
m["generator"] = m.source.map(GEN).fillna("")
m["conditioning"] = np.where(m.image_id.str.contains(":cond_"), "image", np.where(m.cls == "fake", "text", ""))
# hospital of the conditioning image for image-conditioned fakes (H4)
m["cond_site"] = np.where(m.conditioning == "image", m.image_id.str.split(":").str[1].str.split("_").str[1], "")
meta = m.set_index("image_id")[["cls", "site", "source", "view", "patient_id", "group_id", "generator",
                                "conditioning", "cond_site"]]

import sys
VARIANT = sys.argv[1] if len(sys.argv) > 1 else ""  # "", or a sensitivity variant (mask -> S2, jpeg -> S3)
allf = [(f, *Path(f).parts[-3:-1]) for f in glob.glob(str(ROOT / "results/scores/embed/*/*/*.parquet"))]
if VARIANT:  # only <encoder>__<variant> folders; encoder name restored without the suffix
    files = [(f, e.split("__")[0], d) for f, e, d in allf if e.endswith("__" + VARIANT)]
    OUT = OUT / "sens"
    OUT.mkdir(exist_ok=True)
else:  # primary: never read sensitivity folders
    files = [(f, e, d) for f, e, d in allf if "__" not in e]
    files += [(f, "none", "resnet") for f in glob.glob(str(ROOT / "results/scores/resnet/resnet18/*.parquet"))]
PFX = {"mask": "S2_", "jpeg": "S3_", "512": "S4_"}.get(VARIANT, "")
parts = []
for f, enc, det in files:
    hosp, gen = Path(f).stem.split("__")
    d = pd.read_parquet(f, columns=["seed", "image_id", "split", "score", "thr", "flag"])
    d = d[d.split == "test"].drop(columns="split")
    fam = {"logit": "fft" if enc == "pix_fft" else "format"}.get(det, det)
    d["family"], d["encoder"] = fam, ("none" if enc.startswith("pix_") else enc)
    d["hosp_cfg"], d["gen_cfg"] = hosp, gen
    parts.append(d)
L = pd.concat(parts, ignore_index=True)
L = L.join(meta, on="image_id")
L["config"] = L.family + "|" + L.encoder + "|" + L.hosp_cfg + "|" + L.gen_cfg

seen_src = L.hosp_cfg.map(TRAIN_REALS)
is_seen_site = [(s in tr) or (s in ("jsrt", "padchest", "node21_other") and "node21" in tr)
                for s, tr in zip(L.site, seen_src)]
L["novelty_site"] = np.where(L.cls == "real", 1 - np.array(is_seen_site, int), -1)
seen_gen = [src in TRAIN_FAKES[g] for src, g in zip(L.source, L.gen_cfg)]
L["novelty_gen"] = np.where(L.cls == "fake", 1 - np.array(seen_gen, int), -1)

R, F = L[L.cls == "real"], L[L.cls == "fake"]
R.to_parquet(OUT / f"{PFX}reals_long.parquet", index=False)
F.to_parquet(OUT / f"{PFX}fakes_long.parquet", index=False)

keys_r = ["image_id", "site", "view", "patient_id", "family", "encoder", "novelty_site"]
Ra = R[R.site != "node21_other"].groupby(keys_r, observed=True).flag.agg(k="sum", n="size").reset_index()
Ra.to_parquet(OUT / f"{PFX}reals_agg.parquet", index=False)
keys_f = ["image_id", "generator", "conditioning", "cond_site", "group_id", "family", "encoder", "novelty_gen"]
Fa = F.groupby(keys_f, observed=True).flag.agg(k="sum", n="size").reset_index()
Fa.to_parquet(OUT / f"{PFX}fakes_agg.parquet", index=False)
# cell level (primary H1 fit): site x view x fitted detector (config) x seed x novelty -> k of n
keys_c = ["site", "view", "family", "encoder", "config", "seed", "novelty_site"]
Rc = R[R.site != "node21_other"].groupby(keys_c, observed=True).flag.agg(k="sum", n="size").reset_index()
Rc.to_parquet(OUT / f"{PFX}reals_cells.parquet", index=False)
keys_fc = ["generator", "conditioning", "cond_site", "family", "encoder", "config", "seed", "novelty_gen"]
Fc = F.groupby(keys_fc, observed=True).flag.agg(k="sum", n="size").reset_index()
Fc.to_parquet(OUT / f"{PFX}fakes_cells.parquet", index=False)
print(f"reals_cells {len(Rc):,}  fakes_cells {len(Fc):,}")
print(f"reals_long {len(R):,}  fakes_long {len(F):,}  reals_agg {len(Ra):,}  fakes_agg {len(Fa):,}")
print("fits per family:", L.groupby("family").config.nunique().to_dict())
