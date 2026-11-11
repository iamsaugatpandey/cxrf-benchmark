"""Pre-scoring QC of generated images (preregistration §4.5): flag images that are not a single frontal
chest radiograph, for the AUTHOR to confirm before any detector scores exist.

Automated flags (any one triggers review):
  * BiomedCLIP zero-shot: p(frontal chest X-ray) < 0.5 against alternatives (screenshot of software,
    photograph, illustration/diagram, grid of several images, lateral chest X-ray);
  * colour: mean chroma of the raw RGB image > 8 (radiographs are grayscale);
  * shape: raw aspect ratio outside [0.8, 1.25].
Outputs: data/meta/qc_fakes.csv (one row per generated image, with scores and flags) and contact
sheets in figures/qc/fakes/: <set>_flagged.png (all flagged, labelled) and <set>_sample.png
(random 80 unflagged, for a visual sanity check).
Usage: python scripts/qc_fakes.py
"""
import csv
import random
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
OUT = ROOT / "figures/qc/fakes"
OUT.mkdir(parents=True, exist_ok=True)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
REPO = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
LABELS = {"cxr": ["a frontal chest X-ray radiograph", "a chest radiograph, posteroanterior view"],
          "screenshot": ["a screenshot of medical imaging software with a user interface"],
          "photo": ["a photograph of a person", "a photograph of a computer screen"],
          "diagram": ["an anatomical illustration or diagram", "a 3D rendering of the chest"],
          "grid": ["several X-ray images arranged side by side in a grid"],
          "lateral": ["a lateral chest X-ray"]}

SETS = {
    "G2_monai": sorted((D / "raw/gen/G2").glob("p*.png")),
    "G4_text": sorted((D / "raw/api/G4/text").glob("p*.*")),
    "G5_text": sorted((D / "raw/api/G5/text").glob("p*.*")),
    "G4_cond": sorted((D / "raw/api/G4/cond_v2").glob("*.*")),
    "G5_cond": sorted((D / "raw/api/G5/cond_v2").glob("*.*")),
}

model, pre = open_clip.create_model_from_pretrained(REPO)
tok = open_clip.get_tokenizer(REPO)
model = model.to(DEV).eval()
with torch.no_grad():
    keys, T = list(LABELS), []
    for k in keys:
        e = model.encode_text(tok(LABELS[k]).to(DEV))
        e = e / e.norm(dim=-1, keepdim=True)
        T.append(e.mean(0))
    T = torch.stack(T)
    T = T / T.norm(dim=-1, keepdim=True)


def score(paths, bs=48):
    out = []
    for i in range(0, len(paths), bs):
        x = torch.stack([pre(Image.open(p).convert("RGB")) for p in paths[i:i + bs]]).to(DEV)
        with torch.no_grad():
            f = model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
            out += (model.logit_scale.exp() * f @ T.T).softmax(-1).cpu().tolist()
    return out


def chroma(p):
    a = np.asarray(Image.open(p).convert("RGB").resize((128, 128)), np.float32)
    return float(np.abs(a - a.mean(2, keepdims=True)).mean())


def sheet(items, dst, cols=10, t=150):
    if not items:
        return
    rows = (len(items) + cols - 1) // cols
    S = Image.new("L", (cols * t, rows * (t + 16)), 255)
    d = ImageDraw.Draw(S)
    for i, (p, lab) in enumerate(items):
        im = Image.open(p).convert("L")
        im.thumbnail((t, t))
        x, y = (i % cols) * t, (i // cols) * (t + 16)
        S.paste(im, (x, y))
        d.text((x + 2, y + t + 2), lab[:28], fill=0)
    S.save(dst)


rows_out = []
for name, paths in SETS.items():
    probs = score(paths)
    flagged, ok = [], []
    for p, pr in zip(paths, probs):
        pc = pr[0]
        top = keys[int(np.argmax(pr))]
        with Image.open(p) as im:
            w, h = im.size
        ch = chroma(p)
        reasons = [r for r, c in [(f"clip:{top}", pc < 0.5), ("colour", ch > 8), ("aspect", not 0.8 <= w / h <= 1.25)] if c]
        rows_out.append({"set": name, "file": str(p.relative_to(D)), "p_cxr": round(pc, 4), "top_label": top,
                         "chroma": round(ch, 2), "w": w, "h": h, "flag": ";".join(reasons)})
        (flagged if reasons else ok).append((p, f"{p.stem} {';'.join(reasons)}"))
    sheet(flagged, OUT / f"{name}_flagged.png")
    random.seed(0)
    sheet(random.sample(ok, min(80, len(ok))), OUT / f"{name}_sample.png")
    print(f"{name}: {len(paths)} images, flagged {len(flagged)}", flush=True)

with open(D / "meta/qc_fakes.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
    w.writeheader()
    w.writerows(rows_out)
print("wrote data/meta/qc_fakes.csv")
