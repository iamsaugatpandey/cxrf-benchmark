"""Download the first N SynthCheX-75K-v2 parquet shards (G1) and write a sha256 manifest.
Usage: python fetch_synthchex.py <dest_dir> [n_shards=50]"""
import hashlib, sys
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi

REPO = "raman07/SynthCheX-75K-v2"
dest = Path(sys.argv[1]); n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
rev = HfApi().dataset_info(REPO).sha  # pin the exact dataset commit
print("revision", rev, flush=True)
files = ["README.md"] + [f"data/train-{i:05d}-of-01112.parquet" for i in range(n)]
for f in files:
    hf_hub_download(REPO, f, repo_type="dataset", revision=rev, local_dir=dest)
    print("ok", f, flush=True)
(dest / "REVISION.txt").write_text(f"{REPO}@{rev}\n")
with open(dest / "MANIFEST.sha256", "w") as m:
    for p in sorted(x for x in dest.rglob("*") if x.is_file() and ".cache" not in x.parts and not x.name.startswith("MANIFEST")):
        m.write(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(dest)}\n")
print("SYNTHCHEX_OK", len(files))
