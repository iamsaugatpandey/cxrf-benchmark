"""Batch preprocessing of every frontal image in the inventory (Pipeline step 1).

Usage: python preprocess_all.py <data_root> [--workers 8] [--sources a,b]
Reads data/meta/inventory_raw.csv; skips lateral views. Writes
  data/processed/256/<source>/<file_id>.png   (all sources)
  data/processed/512/<source>/<file_id>.png   (sources whose short side >= 512 px)
  data/meta/preprocess_qc.csv                  (one row per image: QC stats + output paths)
Already-processed images are skipped, so the script can be resumed.
"""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

from cxrf.preprocess import preprocess_multi, save_png

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--sources", default="")
args = ap.parse_args()
root = Path(args.root)
QC_COLS = ["image_id", "source", "dyn_range", "out_std", "raw_h", "raw_w", "crop_h", "crop_w",
           "out256", "out512", "error"]


def file_id(image_id: str) -> str:
    return image_id.split(":", 1)[1].replace("/", "_")


def work(r):
    src = root / r["raw_path"]
    fid = file_id(r["image_id"])
    o256 = root / "processed/256" / r["source"] / f"{fid}.png"
    o512 = root / "processed/512" / r["source"] / f"{fid}.png"
    want512 = min(int(r["width"]), int(r["height"])) >= 512
    out = {"image_id": r["image_id"], "source": r["source"], "error": ""}
    try:
        sizes = (256, 512) if want512 else (256,)
        imgs, qc = preprocess_multi(src, sizes)
        save_png(imgs[256], o256)
        if want512:
            save_png(imgs[512], o512)
        out.update(qc, out256=str(o256.relative_to(root)),
                   out512=str(o512.relative_to(root)) if want512 else "")
    except Exception as e:  # recorded, never silently dropped
        out["error"] = f"{type(e).__name__}: {e}"
    return out


if __name__ == "__main__":
    inv = [r for r in csv.DictReader(open(root / "meta/inventory_raw.csv")) if r["view"] != "lateral"]
    if args.sources:
        keep = set(args.sources.split(","))
        inv = [r for r in inv if r["source"] in keep]
    qc_path = root / "meta/preprocess_qc.csv"
    done = {}
    if qc_path.exists():
        done = {r["image_id"]: r for r in csv.DictReader(open(qc_path)) if not r["error"]}
    todo = [r for r in inv if r["image_id"] not in done]
    print(f"{len(inv)} frontal images, {len(done)} already done, {len(todo)} to process", flush=True)
    results = list(done.values())
    with ProcessPoolExecutor(args.workers) as ex:
        for i, res in enumerate(ex.map(work, todo, chunksize=16), 1):
            results.append(res)
            if i % 2000 == 0:
                print(i, flush=True)
    with open(qc_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=QC_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    errs = [r for r in results if r.get("error")]
    print(f"done: {len(results)} rows, {len(errs)} errors", errs[:3])
