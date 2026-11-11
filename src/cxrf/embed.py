"""Frozen-encoder embeddings (Pipeline step 4): CLS token ++ mean of patch tokens, float16.

Input is always a preprocessed 8-bit grayscale PNG from the shared pipeline. Each encoder
gets the image replicated to RGB, resized to its input size (bicubic) and normalised with
the encoder's own mean/std. Input sizes:
  dinov3     256  (the primary analysis resolution; ViT-B/16 -> 16x16 patches)
  rad-dino   518  (its native size; 256-px images are upsampled, so the resolution rule
                   "every image passes through the same 256-px bottleneck" still holds)
  biomedclip 224  (its fixed input size)
DINOv3 register tokens are excluded from the patch mean.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

ENCODERS = {
    "dinov3": ("hf", "facebook/dinov3-vitb16-pretrain-lvd1689m", 256),
    "rad-dino": ("hf", "microsoft/rad-dino", 518),
    "biomedclip": ("openclip", "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", 224),
}


class Encoder:
    def __init__(self, name: str, device: str | None = None, size: int | None = None):
        kind, repo, size0 = ENCODERS[name]
        size = size or size0  # override for the 512-px sensitivity analysis (S4)
        self.name, self.kind, self.size = name, kind, size
        self.dev = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        if kind == "hf":
            from transformers import AutoImageProcessor, AutoModel
            proc = AutoImageProcessor.from_pretrained(repo)
            self.mean, self.std = proc.image_mean, proc.image_std
            self.model = AutoModel.from_pretrained(repo).to(self.dev).eval()
            self.n_reg = getattr(self.model.config, "num_register_tokens", 0) or 0
        else:
            import open_clip
            model, pre = open_clip.create_model_from_pretrained(repo)
            norm = [t for t in pre.transforms if t.__class__.__name__ == "Normalize"][0]
            self.mean, self.std = list(norm.mean), list(norm.std)
            self.model = model.visual.trunk.to(self.dev).eval()
            self.n_reg = 0
        self._m = torch.tensor(self.mean).view(1, 3, 1, 1)
        self._s = torch.tensor(self.std).view(1, 3, 1, 1)

    def _load(self, path) -> torch.Tensor:
        im = Image.open(path).convert("L").resize((self.size, self.size), Image.Resampling.BICUBIC)
        x = torch.from_numpy(np.asarray(im, np.float32) / 255.0)
        return x.expand(3, -1, -1)

    @torch.no_grad()
    def embed(self, paths, batch=32) -> np.ndarray:
        out = []
        for i in range(0, len(paths), batch):
            x = torch.stack([self._load(p) for p in paths[i:i + batch]])
            x = ((x - self._m) / self._s).to(self.dev)
            if self.kind == "hf":
                h = self.model(pixel_values=x).last_hidden_state
            else:
                h = self.model.forward_features(x)
            e = torch.cat([h[:, 0], h[:, 1 + self.n_reg:].mean(1)], dim=1)
            out.append(e.half().cpu().numpy())
        return np.concatenate(out)


def embed_to_disk(name: str, ids: list[str], paths: list[Path], out_dir: Path, batch=32):
    """Writes <out_dir>/<name>.npy (float16) and <name>_ids.txt (row order)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    E = Encoder(name).embed(paths, batch)
    np.save(out_dir / f"{name}.npy", E)
    (out_dir / f"{name}_ids.txt").write_text("\n".join(ids) + "\n")
    return E
