"""Image variants for preregistered sensitivity analyses (OSF 5vwkj, analysis plan 4.6).

  mask : S2 marker-masked. From the primary 256-px image: the upper 20% of the frame and an 8% border on
         every side are set to 0 (following Lee et al. 2026's masking of burned-in annotations).
  jpeg : S3 JPEG-harmonized. From the RAW file: decode -> intensity window (8-bit) -> JPEG quality 90
         encode/decode (Pillow, 4:2:0 default subsampling irrelevant for grayscale) -> border crop -> centre square
         -> 256 px. JPEG needs 8-bit input, so it is applied right after windowing, before all geometric steps.
Writes data/processed_variants/<variant>/256/<source>/<name>.png (same relative names as the primary set)
and data/processed_variants/<variant>/index.csv (image_id, out256).
Usage: python scripts/make_variants.py mask|jpeg [--workers 6]
"""
import argparse
import io
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from cxrf.preprocess import _resize, center_square, crop_border, load_any, save_png, window

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"


def do_mask(row):
    a = np.asarray(Image.open(D / row["out256"])).copy()
    h, w = a.shape
    b = int(round(0.08 * h))
    a[: int(round(0.20 * h)), :] = 0
    a[:b, :] = 0; a[-b:, :] = 0; a[:, :b] = 0; a[:, -b:] = 0
    return a


def do_jpeg(row):
    a = window(load_any(D / row["raw_path"]))  # float32 in [0, 1]
    a8 = np.clip(np.rint(a * 255), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a8).save(buf, format="JPEG", quality=90)
    a = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("L"), np.float32) / 255.0  # back to [0, 1]
    return _resize(center_square(crop_border(a)), 256)


def work(args):
    variant, row = args
    dst = D / "processed_variants" / variant / Path(row["out256"]).relative_to("processed")
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        save_png((do_mask if variant == "mask" else do_jpeg)(row), dst)
    return row["image_id"], str(dst.relative_to(D))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=["mask", "jpeg"])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    m = pd.read_csv(D / "meta/images.csv", keep_default_na=False, dtype=str)
    m = m[m.exclude == ""]
    rows = m[["image_id", "out256", "raw_path"]].to_dict("records")
    out = []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(work, [(a.variant, r) for r in rows], chunksize=64)):
            out.append(r)
            if (i + 1) % 5000 == 0:
                print(f"{a.variant}: {i + 1}/{len(rows)}", flush=True)
    pd.DataFrame(out, columns=["image_id", "out256"]).to_csv(D / "processed_variants" / a.variant / "index.csv", index=False)
    print(f"{a.variant}: done {len(out)}")
