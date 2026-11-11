"""Download a patient-level subset of CheXpert Plus (Stanford AIMI, Redivis) with reports.

Source: Redivis dataset aimi.chexpert_plus:5yyj:v1_0.
Tables: df_chexpert_plus_240401 (metadata + sectioned reports), png_train (file index).
Selection (deterministic): frontal views (PA + AP); patients ordered by sha256(salt:patient_id);
<= 3 frontal images per patient; stop at N images. Files verified against Redivis md5.
License: Stanford Dataset Research Use Agreement: no redistribution; images must NOT be sent to
third-party APIs (excluded from image-conditioned generation and LLM evaluation).

Output: data/raw/chexpert_plus/{meta_full.parquet, files.parquet, selection.csv, png/<patient>/<study>/<view>.png,
        MANIFEST.sha256}
Usage: python scripts/fetch_chexpert_plus.py [N=8000] [--threads 8]
"""
import argparse
import base64
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from cxrf.env import load_env

load_env()
os.environ.setdefault("REDIVIS_API_TOKEN", os.environ.get("REDIVIS_API_KEY", ""))
import redivis  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/raw/chexpert_plus"
REF = "chexpert_plus:5yyj:v1_0"

ap = argparse.ArgumentParser()
ap.add_argument("n", type=int, nargs="?", default=8000)
ap.add_argument("--threads", type=int, default=8)
a = ap.parse_args()
OUT.mkdir(parents=True, exist_ok=True)
ds = redivis.organization("aimi").dataset(REF)

meta_p, files_p = OUT / "meta_full.parquet", OUT / "files.parquet"
if not meta_p.exists():
    ds.table("df_chexpert_plus_240401:bavj").to_pandas_dataframe().to_parquet(meta_p)
if not files_p.exists():
    ds.table("png_train:s6cj").to_pandas_dataframe().to_parquet(files_p)
meta, files = pd.read_parquet(meta_p), pd.read_parquet(files_p)
print("meta", meta.shape, "| files", files.shape, flush=True)

# path_to_image is 'train/patientX/studyY/viewZ_frontal.jpg'; file index names 'patientX/studyY/viewZ_frontal.png'
meta["file_name"] = meta["path_to_image"].str.replace(r"^(train|valid)/", "", regex=True).str.replace(".jpg", ".png")
m = meta[meta["frontal_lateral"] == "Frontal"].merge(files, on="file_name", how="inner")
m["prank"] = m["deid_patient_id"].map(lambda p: hashlib.sha256(f"cxp-subset-v1:{p}".encode()).hexdigest())
m = m.sort_values(["prank", "patient_report_date_order", "file_name"])
m = m.groupby("deid_patient_id", sort=False).head(3)
sel = m.head(a.n).copy()
print(f"selected {len(sel)} images from {sel['deid_patient_id'].nunique()} patients; "
      f"{sel['size'].sum() / 1e9:.1f} GB; views {sel['ap_pa'].value_counts().to_dict()}", flush=True)
sel.drop(columns=["prank"]).to_csv(OUT / "selection.csv", index=False)


PNG = ds.table("png_train:s6cj")


def md5_b64(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return base64.b64encode(h.digest()).decode()


def get(row):
    dst = OUT / "png" / row.file_name
    if dst.exists() and md5_b64(dst) == row.md5_hash:
        return "skip"
    dst.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            PNG.file(row.file_name).download(str(dst), overwrite=True)
            if md5_b64(dst) == row.md5_hash:
                return "ok"
        except Exception as e:  # retry transient errors
            err = e
    return f"FAIL {row.file_name}: {err if 'err' in dir() else 'md5 mismatch'}"


done = 0
with ThreadPoolExecutor(a.threads) as ex:
    for r in ex.map(get, sel.itertuples()):
        done += 1
        if r.startswith("FAIL"):
            print(r, flush=True)
        if done % 250 == 0:
            print(f"{done}/{len(sel)}", flush=True)

with open(OUT / "MANIFEST.sha256", "w") as f:
    for p in sorted((OUT / "png").rglob("*.png")):
        f.write(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(OUT)}\n")
print("CHEXPERT_PLUS_OK", sum(1 for _ in (OUT / 'png').rglob('*.png')), "png")
