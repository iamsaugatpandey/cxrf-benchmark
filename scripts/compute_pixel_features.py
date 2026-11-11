"""Pixel-level baseline features for every included image (256-px preprocessed PNG).
  pix_fft    : 64-bin azimuthally averaged log power spectrum (cxrf.detectors.fft_features)
  pix_format : format features from the shortcut audit (histogram, stats, low FFT, marker block)
Saved in the embedding layout so run_embed_grid.py can use them as 'encoders':
  data/embeddings/<name>/emb.npy (float32), data/embeddings/<name>/ids.txt
Usage: python scripts/compute_pixel_features.py [variant]
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from cxrf.audit import features
from cxrf.detectors import fft_features

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
m = pd.read_csv(D / "meta/images.csv", keep_default_na=False, dtype=str)
m = m[m.exclude == ""]
import sys
VAR = sys.argv[1] if len(sys.argv) > 1 else ""  # sensitivity variant (mask|jpeg)
if VAR:
    v = pd.read_csv(D / "processed_variants" / VAR / "index.csv").set_index("image_id").out256
    m = m.assign(out256=m.image_id.map(v)).dropna(subset=["out256"])  # S4 (512) excludes Open-I
F_fft, F_fmt = [], []
for i in range(0, len(m), 2000):
    part = m.iloc[i:i + 2000]
    imgs = np.stack([np.asarray(Image.open(D / p)) for p in part.out256])
    F_fft.append(fft_features(imgs))
    f = features(imgs)
    F_fmt.append(np.c_[f["hist"], f["stats"], f["fft_low"], f["marker"]].astype(np.float32))
    print(i + len(part), "/", len(m), flush=True)
for name, F in [("pix_fft", F_fft), ("pix_format", F_fmt)]:
    out = D / "embeddings" / (name + (f"__{VAR}" if VAR else ""))
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "emb.npy", np.concatenate(F))
    (out / "ids.txt").write_text("\n".join(m.image_id) + "\n")
    print(name, np.concatenate(F).shape)
