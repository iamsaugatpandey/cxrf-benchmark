"""QC contact sheets: preprocess a random sample of raw images per source and tile them.

Usage: python contact_sheets.py <data_root> <out_dir> [n_per_source=40] [size=256]
Sources come from the inventory scanners; Open-I is restricted to frontal views when
data/meta/openi_views.csv exists. One PNG per source: <out_dir>/sheet_<source>.png
"""
import csv
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from cxrf.preprocess import preprocess

root, out = Path(sys.argv[1]), Path(sys.argv[2])
n = int(sys.argv[3]) if len(sys.argv) > 3 else 40
size = int(sys.argv[4]) if len(sys.argv) > 4 else 256
out.mkdir(parents=True, exist_ok=True)

frontal_openi = None
vf = root / "meta" / "openi_views.csv"
if vf.exists():
    frontal_openi = {r["image_id"][6:] for r in csv.DictReader(open(vf)) if r["view"] == "PA"}

SOURCES = {
    "nih": sorted((root / "raw/nih/images").glob("*/images/*.png")),
    "montgomery": sorted((root / "raw/montgomery/CXR_png").glob("*.png")),
    "shenzhen": sorted((root / "raw/shenzhen/CXR_png").glob("*.png")),
    "openi": [p for p in sorted((root / "raw/openi/png").rglob("*.png"))
              if frontal_openi is None or p.stem in frontal_openi],
    "synthchex": sorted((root / "raw/synthchex/png").glob("*.png")),
}
cols, t = 8, 192
for name, files in SOURCES.items():
    if not files:
        continue
    random.seed(0)
    pick = random.sample(files, min(n, len(files)))
    rows = (len(pick) + cols - 1) // cols
    S = Image.new("L", (cols * t, rows * t), 0)
    d = ImageDraw.Draw(S)
    for i, p in enumerate(pick):
        im = Image.fromarray(preprocess(p, size=size)).resize((t, t))
        x, y = (i % cols) * t, (i // cols) * t
        S.paste(im, (x, y))
        d.text((x + 3, y + t - 12), p.stem[:24], fill=255)
    S.save(out / f"sheet_{name}.png")
    print(name, len(files), "files; sheet of", len(pick))
