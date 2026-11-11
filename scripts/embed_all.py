"""Embed every included image with each frozen encoder (Phase 6). No scoring happens here, so
nothing is unblinded: embeddings are inputs, not outcomes.

Reads data/meta/images.csv (exclude == ''), uses the 256-px preprocessed PNGs. Writes
  data/embeddings/<encoder>/emb.npy   (float16, N x 1536)
  data/embeddings/<encoder>/ids.txt   (row order = image_id)
Incremental: images already embedded are skipped (rows appended), so new sources such as
CheXpert can be added later without recomputing.
Usage: python scripts/embed_all.py dinov3 rad-dino biomedclip [--batch 32]
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from cxrf.embed import Encoder
from cxrf.env import load_env

load_env()
ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"

ap = argparse.ArgumentParser()
ap.add_argument("encoders", nargs="+")
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--chunk", type=int, default=2048)
ap.add_argument("--variant", default="", help="sensitivity variant (mask|jpeg): reads data/processed_variants/<v>/index.csv, writes data/embeddings/<encoder>__<v>/")
a = ap.parse_args()

rows = [r for r in csv.DictReader(open(D / "meta/images.csv")) if not r["exclude"] and r["out256"]]
if a.variant:
    vmap = {r["image_id"]: r["out256"] for r in csv.DictReader(open(D / "processed_variants" / a.variant / "index.csv"))}
    rows = [{**r, "out256": vmap[r["image_id"]]} for r in rows if r["image_id"] in vmap]  # S4 excludes Open-I
for name in a.encoders:
    out = D / "embeddings" / (name + (f"__{a.variant}" if a.variant else ""))
    out.mkdir(parents=True, exist_ok=True)
    ids_f, emb_f = out / "ids.txt", out / "emb.npy"
    have = ids_f.read_text().split() if ids_f.exists() else []
    E = [np.load(emb_f)] if emb_f.exists() else []
    seen = set(have)
    todo = [r for r in rows if r["image_id"] not in seen]
    print(f"{name}: {len(have)} done, {len(todo)} to embed", flush=True)
    if not todo:
        continue
    S4_SIZE = {"dinov3": 512, "rad-dino": 518, "biomedclip": 224}  # S4: native-resolution input from the 512-px image
    enc = Encoder(name, size=S4_SIZE[name] if a.variant == "512" else None)
    for i in range(0, len(todo), a.chunk):
        part = todo[i:i + a.chunk]
        E.append(enc.embed([D / r["out256"] for r in part], a.batch))
        have += [r["image_id"] for r in part]
        np.save(emb_f, np.concatenate(E))  # checkpoint after every chunk
        ids_f.write_text("\n".join(have) + "\n")
        print(f"  {name}: {len(have)}/{len(rows)}", flush=True)
