"""Extract SynthCheX parquet rows to lossless PNGs (pixels unchanged, RGB as stored) plus a
labels CSV. Raw parquet stays untouched. Usage: python extract_synthchex.py <synthchex_dir>"""
import csv, sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
from PIL import Image

root = Path(sys.argv[1]); out = root / "png"; out.mkdir(exist_ok=True)
rows = []
for shard in sorted((root / "data").glob("train-*.parquet")):
    t = pq.read_table(shard)
    for fn, lab, img in zip(t["image_filename"].to_pylist(), t["label"].to_pylist(), t["image"]):
        a = np.asarray(img.as_py(), dtype=np.uint8)
        assert a.shape == (1024, 1024, 3), (fn, a.shape)
        assert "/" not in fn and ".." not in fn
        Image.fromarray(a, "RGB").save(out / fn, format="PNG")
        rows.append({"image_filename": fn, "shard": shard.name, "label": lab})
    print(shard.name, len(rows), flush=True)
with open(root / "labels.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["image_filename", "shard", "label"]); w.writeheader(); w.writerows(rows)
print("EXTRACTED", len(rows))
