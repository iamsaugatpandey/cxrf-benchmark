"""Build data/meta/inventory_raw.csv from all sources present on disk.
Usage: python build_inventory.py <data_root> [source ...]   (default: every scanner)"""
import csv, sys
from collections import Counter
from pathlib import Path
from cxrf.inventory import SCANNERS, scan_openi, write_inventory

root = Path(sys.argv[1]); names = sys.argv[2:] or list(SCANNERS)
views = {r["image_id"]: r["view"] for r in csv.DictReader(open(root / "meta/openi_views.csv"))}
def rows():
    for n in names:
        yield from (scan_openi(root, views) if n == "openi" else SCANNERS[n](root))
out = root / "meta/inventory_raw.csv"
print("rows:", write_inventory(rows(), out))
inv = list(csv.DictReader(open(out)))
print(Counter((r["source"], r["cls"], r["view"]) for r in inv))
print("unique groups per source:", {s: len({r["group_id"] for r in inv if r["source"] == s}) for s in names})
dup = [h for h, c in Counter(r["raw_sha256"] for r in inv).items() if c > 1]
print("exact duplicate files (sha256):", len(dup))
