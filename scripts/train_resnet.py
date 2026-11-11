"""Phase 6, confirmatory: ResNet-18 fine-tuned end-to-end at 256 px (OSF 5vwkj: 3 seeds).
Grid: hospital in {nih_only, loho_nih, loho_openi, loho_shenzhen, loho_chexpert} x generator g12
(both open generators), 3 seeds = 15 fine-tunes. (Embedding probes cover the full generator factor;
the CNN is the expensive family, so it gets the pooled-generator configuration only.)
Recipe: ImageNet weights, grayscale replicated to 3 channels, ImageNet normalisation, random
horizontal flip, AdamW lr 1e-4 wd 1e-2, batch 64, 3 epochs, BCE with fakes weighted n_real/n_fake.
Training sets, threshold set and outputs follow run_embed_grid.py exactly:
  results/scores/resnet/resnet18/<hospital>__g12.parquet  (seed, image_id, split, score, thr, flag)
  results/scores/resnet/runs.csv
Usage: python scripts/train_resnet.py [--smoke]
"""
import argparse
import csv
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

from cxrf.metrics import threshold_at_fpr

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
OUT = ROOT / "results/scores/resnet"
FREEZE = "acbb483e7568d1bd1286c1a46fc7d34bee21793e8967aeff1408da6e1342c589"
CORE = ["nih", "openi", "shenzhen", "chexpert"]
HOSPITALS = {"nih_only": ["nih"], **{f"loho_{h}": [x for x in CORE if x != h] for h in CORE}}
FAKES = ["synthchex", "monai_ldm"]
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

class DS(Dataset):
    def __init__(self, paths, labels, train):
        self.p, self.y, self.train = list(paths), np.asarray(labels, np.float32), train

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        x = torch.from_numpy(np.asarray(Image.open(D / self.p[i]), np.float32) / 255.0)[None].expand(3, -1, -1)
        if self.train and torch.rand(1).item() < 0.5:
            x = x.flip(-1)
        return (x - MEAN) / STD, self.y[i]


def scores(model, paths):
    dl = DataLoader(DS(paths, np.zeros(len(paths)), False), batch_size=128, num_workers=4)
    out = []
    model.eval()
    with torch.no_grad():
        for x, _ in dl:
            out.append(torch.sigmoid(model(x.to(DEV)).squeeze(-1)).float().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=3)
    a = ap.parse_args()

    sha = hashlib.sha256((D / "meta/images.csv").read_bytes()).hexdigest()
    assert sha == FREEZE, "images.csv changed since the data freeze"
    m = pd.read_csv(D / "meta/images.csv", keep_default_na=False, dtype=str)
    m = m[m.exclude == ""].reset_index(drop=True)
    ev = m[m.split.isin(["val"] if a.smoke else ["val", "test"])]
    OUT.mkdir(parents=True, exist_ok=True)
    for hosp, real_src in HOSPITALS.items():
        dst = OUT / "resnet18" / f"{hosp}__g12.parquet"
        if dst.exists() and not a.smoke:
            continue
        tr_src = real_src + ([] if hosp == "nih_only" else ["node21"])
        tr = m[(m.split == "train") & (((m.cls == "real") & m.source.isin(tr_src)) | m.source.isin(FAKES))]
        y = (tr.cls != "real").astype(int).values
        if a.smoke:
            tr = pd.concat([tr[y == 0].sample(256, random_state=0), tr[y == 1].sample(256, random_state=0)])
            y = (tr.cls != "real").astype(int).values
        thr_mask = ((ev.cls == "real") & ev.source.isin(real_src)).values
        parts = []
        for seed in range(1 if a.smoke else a.seeds):
            t0 = time.time()
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            model.fc = nn.Linear(model.fc.in_features, 1)
            model = model.to(DEV)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
            lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((y == 0).sum() / (y == 1).sum()), dtype=torch.float32, device=DEV))
            g = torch.Generator().manual_seed(seed)
            dl = DataLoader(DS(tr.out256, y, True), batch_size=64, shuffle=True, generator=g, num_workers=6,
                            persistent_workers=True)
            for ep in range(1 if a.smoke else a.epochs):
                model.train()
                tot = 0.0
                for i, (x, yb) in enumerate(dl):
                    opt.zero_grad()
                    loss = lossf(model(x.to(DEV)).squeeze(-1), yb.to(DEV))
                    loss.backward()
                    opt.step()
                    tot += loss.item()
                print(f"{hosp} seed {seed} epoch {ep}: mean loss {tot / (i + 1):.4f} ({time.time() - t0:.0f}s)", flush=True)
            s = scores(model, ev.out256)
            thr = threshold_at_fpr(s[thr_mask], 0.01)
            parts.append(pd.DataFrame({"seed": seed, "image_id": ev.image_id.values, "split": ev.split.values,
                                       "score": s, "thr": thr, "flag": (s > thr).astype(np.int8)}))
            row = {"model": "resnet18", "hospital": hosp, "generator": "g12", "seed": seed,
                   "n_train_real": int((y == 0).sum()), "n_train_fake": int((y == 1).sum()),
                   "n_thr": int(thr_mask.sum()), "thr": thr, "seconds": round(time.time() - t0), "freeze_sha": sha[:12]}
            print(row, flush=True)
            if a.smoke:
                assert np.isfinite(s).all() and abs(np.mean(s[thr_mask] > thr) - 0.01) < 0.01
                print("SMOKE OK: ResNet-18 code path ran on train/val only")
                return
            new = not (OUT / "runs.csv").exists()
            with open(OUT / "runs.csv", "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(row))
                if new:
                    w.writeheader()
                w.writerow(row)
        dst.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(parts).to_parquet(dst, index=False)


if __name__ == "__main__":
    main()
