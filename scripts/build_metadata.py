"""Join inventory + preprocessing QC + splits into data/meta/images.csv (the master table).

Usage: python build_metadata.py <data_root>
Adds: split, exclude (reason or ''), out256, out512, dyn_range. Prints counts per
source x split for included images and the exclusion tally, and checks that no group
spans two splits.
"""
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

from cxrf.inventory import COLUMNS
from cxrf.splits import exclusion_reason, split_of

TEST_ONLY = {"gpt_image", "gemini_image"}  # G4, G5

root = Path(sys.argv[1])
inv = list(csv.DictReader(open(root / "meta/inventory_raw.csv")))
qc = {r["image_id"]: r for r in csv.DictReader(open(root / "meta/preprocess_qc.csv"))}
# NODE21 content de-duplication (scripts/dedup_node21.py): corr >= LINK -> share the matched
# image's group (possible same patient); corr >= DUP or a known source match -> exclude the
# NODE21 copy from the real pool (the original stays).
LINK, DUP = 0.95, 0.98
gid_of = {r["image_id"]: r["group_id"] for r in inv}
n21 = {}
dp = root / "meta/node21_dups.csv"
if dp.exists():
    for d in csv.DictReader(open(dp)):
        c = float(d["corr"])
        match = d["known_match"] or d["best_match"]
        n21[d["node21_id"]] = (match if (d["known_match"] or c >= LINK) else "",
                               bool(d["known_match"]) or c >= DUP)
for r in inv:
    m = n21.get(r["image_id"])
    if m and m[0]:
        r["group_id"] = gid_of[m[0]]
# Author-confirmed QC exclusions of generated images (preregistration §4.5). Only the CONFIRMED file is
# applied; qc_exclusions_proposed.csv is the automated + reviewed proposal awaiting the author's sign-off.
qc_ex = {}
qp = root / "meta/qc_exclusions_confirmed.csv"
if qp.exists():
    qc_ex = {r["image_id"]: r["reason"] for r in csv.DictReader(open(qp))}
seen = set()
rows = []
for r in sorted(inv, key=lambda r: r["image_id"]):  # deterministic "first copy wins"
    q = qc.get(r["image_id"])
    m = n21.get(r["image_id"])
    reason = exclusion_reason(r, q, seen)
    if not reason and m and m[1]:
        reason = "near_duplicate_of_" + m[0].split(":")[0]
    if not reason and r["image_id"] in qc_ex:
        reason = "qc_" + qc_ex[r["image_id"]]
    # commercial generators are held-out, test-only by design (preregistration 3.2): never trained on.
    # Their text prompts reuse Open-I report groups, so the group hash would otherwise send 80% to train/val.
    split = "test" if r["source"] in TEST_ONLY else split_of(r["group_id"])
    rows.append({**r, "split": split,
                 "exclude": reason,
                 "out256": q["out256"] if q else "", "out512": q["out512"] if q else "",
                 "dyn_range": q["dyn_range"] if q else "", "out_std": q["out_std"] if q else ""})

g2s = defaultdict(set)
for r in rows:
    if r["source"] not in TEST_ONLY:
        g2s[r["group_id"]].add(r["split"])
assert all(len(s) == 1 for s in g2s.values()), "a group spans splits"
assert all(r["split"] == "test" for r in rows if r["source"] in TEST_ONLY)

out = root / "meta/images.csv"
cols = COLUMNS + ["split", "exclude", "out256", "out512", "dyn_range", "out_std"]
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)
inc = [r for r in rows if not r["exclude"]]
print(f"{out}: {len(rows)} rows, {len(inc)} included")
print("excluded:", Counter((r["source"], r["exclude"]) for r in rows if r["exclude"]))
t = Counter((r["source"], r["split"]) for r in inc)
for s in sorted({r["source"] for r in inc}):
    print(f"  {s:10s} train {t[(s, 'train')]:6d}  val {t[(s, 'val')]:6d}  test {t[(s, 'test')]:6d}")
nih_pa_val = sum(1 for r in inc if r["source"] == "nih" and r["split"] == "val" and r["view"] == "PA")
print("NIH PA-only val images (for the 1% FPR threshold):", nih_pa_val)
