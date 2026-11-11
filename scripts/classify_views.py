"""Zero-shot frontal vs lateral view labels for datasets without view metadata (Open-I).

The study is frontal-only, and Open-I mixes frontal and lateral images with no per-image view
field. This uses BiomedCLIP zero-shot (no manual labels): prompts "a frontal/PA chest X-ray"
vs "a lateral chest X-ray". As a sanity check it also scores a sample of NIH images, which
are all frontal (PA/AP), and reports the lateral rate there (should be ~0).

Usage: python classify_views.py <data_root> <out_csv> [n_nih_check=500]
Writes image_id, p_lateral, view (lateral if p_lateral >= 0.97 else PA).
"""
import csv
import random
import sys
from pathlib import Path

import open_clip
import torch
from PIL import Image

root, out = Path(sys.argv[1]), Path(sys.argv[2])
LATERAL_THR = 0.97  # set from the bimodal score histogram + QC sheets (WORKLOG 2026-11-08)
n_check = int(sys.argv[3]) if len(sys.argv) > 3 else 500
dev = "mps" if torch.backends.mps.is_available() else "cpu"
REPO = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
model, pre = open_clip.create_model_from_pretrained(REPO)
tok = open_clip.get_tokenizer(REPO)
model = model.to(dev).eval()
prompts = {"PA": ["a frontal chest X-ray", "a posteroanterior chest radiograph"],
           "lateral": ["a lateral chest X-ray", "a lateral view chest radiograph"]}
with torch.no_grad():
    T = []
    for k in ("PA", "lateral"):
        e = model.encode_text(tok(prompts[k]).to(dev))
        e = e / e.norm(dim=-1, keepdim=True)
        T.append(e.mean(0))
    T = torch.stack(T)
    T = T / T.norm(dim=-1, keepdim=True)


def p_lateral(paths, bs=64):
    out = []
    for i in range(0, len(paths), bs):
        x = torch.stack([pre(Image.open(p).convert("RGB")) for p in paths[i:i + bs]]).to(dev)
        with torch.no_grad():
            f = model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
            logits = model.logit_scale.exp() * f @ T.T
            out += logits.softmax(-1)[:, 1].cpu().tolist()
    return out


openi = sorted((root / "raw/openi/png").rglob("*.png"))
po = p_lateral(openi)
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["image_id", "p_lateral", "view"])
    for p, s in zip(openi, po):
        w.writerow([f"openi:{p.stem}", f"{s:.4f}", "lateral" if s >= LATERAL_THR else "PA"])
n_lat = sum(s >= LATERAL_THR for s in po)
amb = sum(0.2 < s < 0.8 for s in po)
print(f"Open-I: {len(po)} images, lateral {n_lat}, frontal {len(po) - n_lat}, ambiguous(0.2-0.8) {amb}")

random.seed(0)
nih = random.sample(sorted((root / "raw/nih/images").glob("*/images/*.png")), n_check)
pn = p_lateral(nih)
print(f"NIH sanity ({n_check} frontal images): predicted lateral {sum(s >= LATERAL_THR for s in pn)}")
