"""Local (free) text-to-image generation on Apple-silicon MPS for the open generators.

  G2  MONAI CXR latent diffusion (MIMIC-CXR-trained, report-conditioned; Apache-2.0)
      HF: MONAI/cxr_image_synthesis_latent_diffusion_model. DDIM 50 steps, guidance 7.0,
      v-prediction, latent 3x64x64 -> 512x512, exactly as the bundle's inference.json.
  G3  Stable Diffusion 1.5 prompted as a "naive forger" (general web-trained), 512x512,
      DPM-Solver++ 30 steps, guidance 7.5. Safety checker disabled (outputs are radiographs;
      the checker blacks out false positives, which would silently change the sample).

Prompts come from data/meta/prompts.csv (same prompt_id across generators = content matched).
Each image gets a fixed seed = int(sha256(f"{gen}:{prompt_id}")[:8], 16). Raw outputs are
saved untouched as PNG under data/raw/gen/<GEN>/; a per-image log (seed, steps, guidance,
seconds) goes to data/raw/gen/<GEN>/gen_log.csv. Resumable.

Usage: python scripts/gen_open.py G2|G3 [n] [--start k]
"""
import argparse
import csv
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def seed_of(gen, pid):
    return int(hashlib.sha256(f"{gen}:{pid}".encode()).hexdigest()[:8], 16)


class MonaiCXR:
    REPO = "MONAI/cxr_image_synthesis_latent_diffusion_model"
    STEPS, GUIDANCE, SCALE = 50, 7.0, 0.3

    def __init__(self):
        from huggingface_hub import hf_hub_download
        from monai.networks.nets import AutoencoderKL, DiffusionModelUNet
        from monai.networks.schedulers import DDIMScheduler
        from transformers import CLIPTextModel, CLIPTokenizer
        self.rev = "main"
        ae = AutoencoderKL(spatial_dims=2, in_channels=1, out_channels=1, latent_channels=3,
                           channels=[64, 128, 128, 128], num_res_blocks=2, norm_num_groups=32,
                           norm_eps=1e-6, attention_levels=[False] * 4,
                           with_encoder_nonlocal_attn=False, with_decoder_nonlocal_attn=False)
        ae.load_old_state_dict(torch.load(hf_hub_download(self.REPO, "models/autoencoder.pt"), map_location="cpu"))
        un = DiffusionModelUNet(spatial_dims=2, in_channels=3, out_channels=3, channels=[256, 512, 768],
                                num_res_blocks=2, attention_levels=[False, True, True], norm_num_groups=32,
                                norm_eps=1e-6, resblock_updown=False, num_head_channels=[0, 512, 768],
                                with_conditioning=True, transformer_num_layers=1, cross_attention_dim=1024)
        un.load_old_state_dict(torch.load(hf_hub_download(self.REPO, "models/model.pt"), map_location="cpu"))
        self.ae, self.unet = ae.to(DEV).eval(), un.to(DEV).eval()
        self.sched = DDIMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015,
                                   beta_end=0.0205, prediction_type="v_prediction", clip_sample=False)
        # stabilityai/stable-diffusion-2-1-base was removed from HF; this mirror's text-encoder
        # weights are byte-identical (sha256 cce6febb… / c3e254d7…) across 3 independent mirrors.
        base, rev = "sd-research/stable-diffusion-2-1-base", "0708cecd370b4d1c3a6ff3f7332f5e9aea78896f"
        self.tok = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer", revision=rev)
        self.te = CLIPTextModel.from_pretrained(base, subfolder="text_encoder", revision=rev).to(DEV).eval()
        self.params = {"steps": self.STEPS, "guidance": self.GUIDANCE, "scheduler": "DDIM v-pred",
                       "size": 512, "text_encoder": f"{base}@{rev}"}

    @torch.no_grad()
    def __call__(self, prompt, seed):
        ids = self.tok(["", prompt], padding="max_length", max_length=self.tok.model_max_length,
                       truncation=True, return_tensors="pt").input_ids.to(DEV)
        ctx = self.te(ids)[0]
        g = torch.Generator().manual_seed(seed)
        x = torch.randn((1, 3, 64, 64), generator=g).to(DEV)
        self.sched.set_timesteps(num_inference_steps=self.STEPS)
        for t in self.sched.timesteps:
            out = self.unet(torch.cat([x] * 2), timesteps=torch.tensor([t], device=DEV).long(), context=ctx)
            u, c = out.chunk(2)
            x, _ = self.sched.step(u + self.GUIDANCE * (c - u), t, x)
        img = self.ae.decode_stage_2_outputs(x / self.SCALE)
        a = img[0, 0].float().clamp(0, 1).cpu().numpy()
        return Image.fromarray((a * 255).round().astype(np.uint8), "L")


class SD15:
    REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    STEPS, GUIDANCE = 30, 7.5

    def __init__(self):
        from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
        p = StableDiffusionPipeline.from_pretrained(self.REPO, revision="451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
                                                    torch_dtype=torch.float16,
                                                    safety_checker=None, requires_safety_checker=False)
        p.scheduler = DPMSolverMultistepScheduler.from_config(p.scheduler.config)
        p.enable_attention_slicing()
        p.set_progress_bar_config(disable=True)
        self.p = p.to(DEV)
        self.params = {"steps": self.STEPS, "guidance": self.GUIDANCE, "scheduler": "DPM-Solver++",
                       "size": 512, "dtype": "fp16", "safety_checker": "disabled"}

    def __call__(self, prompt, seed):
        # v2 prompt (v1 produced 3-D anatomy renders): radiographic vocabulary + strong negatives
        text = f"PA chest radiograph, digital radiography film, hospital x-ray, grayscale, {prompt}"
        neg = ("3d render, illustration, anatomy model, skeleton art, transparent body, cgi, drawing, "
               "photo, color, text, user interface, border, frame, cartoon")
        g = torch.Generator(device="cpu").manual_seed(seed)
        im = self.p(text, negative_prompt=neg, num_inference_steps=self.STEPS, guidance_scale=self.GUIDANCE,
                    height=512, width=512, generator=g).images[0]
        return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gen", choices=["G2", "G3"])
    ap.add_argument("n", type=int, nargs="?", default=10)
    ap.add_argument("--start", type=int, default=0)
    a = ap.parse_args()
    prompts = list(csv.DictReader(open(D / "meta/prompts.csv")))[a.start:a.start + a.n]
    out = D / "raw/gen" / a.gen
    out.mkdir(parents=True, exist_ok=True)
    logp = out / "gen_log.csv"
    new = not logp.exists()
    model = MonaiCXR() if a.gen == "G2" else SD15()
    with open(logp, "a", newline="") as lf:
        w = csv.DictWriter(lf, fieldnames=["prompt_id", "group_id", "seed", "seconds", "params", "file"])
        if new:
            w.writeheader()
        for p in prompts:
            dst = out / f"{p['prompt_id']}.png"
            if dst.exists():
                continue
            s = seed_of(a.gen, p["prompt_id"])
            t = time.time()
            im = model(p["text_clean"], s)
            im.save(dst, format="PNG")
            w.writerow({"prompt_id": p["prompt_id"], "group_id": p["group_id"], "seed": s,
                        "seconds": round(time.time() - t, 2), "params": str(model.params), "file": dst.name})
            lf.flush()
            print(f"{a.gen} {p['prompt_id']} {time.time() - t:.1f}s", flush=True)


if __name__ == "__main__":
    main()
