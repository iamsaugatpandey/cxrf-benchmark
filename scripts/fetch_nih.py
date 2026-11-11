"""Download NIH ChestX-ray14 parts + CSVs from the HF mirror alkzar90/NIH-Chest-X-ray-dataset
(original NIH zips repackaged), pin the revision, unzip, and write a sha256 manifest.
Usage: python fetch_nih.py <dest_dir> [parts=1,2,3]"""
import hashlib, sys, zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi

REPO = "alkzar90/NIH-Chest-X-ray-dataset"
dest = Path(sys.argv[1]); parts = [int(p) for p in (sys.argv[2] if len(sys.argv) > 2 else "1,2,3").split(",")]
rev = HfApi().dataset_info(REPO).sha
print("revision", rev, flush=True)
meta = ["README.md", "data/BBox_List_2017.csv", "data/Data_Entry_2017_v2020.csv",
        "data/test_list.txt", "data/train_val_list.txt"]
for f in meta + [f"data/images/images_{p:03d}.zip" for p in parts]:
    hf_hub_download(REPO, f, repo_type="dataset", revision=rev, local_dir=dest)
    print("ok", f, flush=True)
for p in parts:
    z = dest / f"data/images/images_{p:03d}.zip"
    out = dest / "images" / f"images_{p:03d}"
    if not out.exists():
        with zipfile.ZipFile(z) as zf:
            bad = [n for n in zf.namelist() if n.startswith("/") or ".." in n]
            assert not bad, bad[:3]
            zf.extractall(out)
    print("unzipped", z.name, sum(1 for _ in out.rglob("*.png")), "png", flush=True)
(dest / "REVISION.txt").write_text(f"{REPO}@{rev}\n")
with open(dest / "MANIFEST.sha256", "w") as m:
    for q in sorted(x for x in dest.rglob("*") if x.is_file() and ".cache" not in x.parts and not x.name.startswith("MANIFEST")):
        h = hashlib.sha256()
        with open(q, "rb") as fh:
            for c in iter(lambda: fh.read(1 << 22), b""): h.update(c)
        m.write(f"{h.hexdigest()}  {q.relative_to(dest)}\n")
print("NIH_OK")
