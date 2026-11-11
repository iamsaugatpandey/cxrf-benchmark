"""Phase 2 pilot: API generation (G4 GPT-image, G5 Gemini) + MLLM detection, all logged.

Usage (from the project root):
  python scripts/pilot_api.py prompts    # 10 content-matched prompts from Open-I train reports
  python scripts/pilot_api.py generate   # 10 images per generator -> data/raw/api/<gen>/pilot/
  python scripts/pilot_api.py detect     # 20 real + 20 fake -> one flagship MLLM per provider
  python scripts/pilot_api.py report     # parse results, cost summary

Rules followed (study plan): raw API files kept untouched; analysis images only via the shared
preprocessing; every call logged with apilog (image bytes never inlined, only path + sha256);
pilot draws only from split == "train".
"""
from __future__ import annotations

import base64
import csv
import io
import json
import random
import re
import sys
import time
from pathlib import Path

from cxrf.apilog import LOG_DIR, log_call
from cxrf.env import load_env
from cxrf.preprocess import preprocess, save_png

load_env()
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PILOT = DATA / "pilot"
GEN_MODELS = {"G4": ("openai", "gpt-image-2"), "G5": ("gemini", "gemini-3.1-flash-image")}
DET_MODELS = [("anthropic", "claude-opus-5-5"), ("openai", "gpt-6-astra"), ("gemini", "gemini-4-argon")]
DET_FALLBACK = {"gpt-6-astra": "gpt-5.5"}

VERSION = "v2"  # v1 said "as stored in a hospital PACS" -> Gemini drew PACS-viewer screenshots
PROMPT_TEMPLATE = (
    "A single realistic frontal posteroanterior (PA) chest X-ray radiograph of an adult, grayscale. "
    "The radiograph fills the entire image: chest only, from the lung apices to the costophrenic "
    "angles. No viewer software, no user interface, no borders, no second image, not a photograph "
    "of a screen. The radiology report for this image reads: "
    "Findings: {findings} Impression: {impression}"
)
DETECT_PROMPT = (
    "You are assisting an image-forensics research study on the provenance of medical images. "
    "This is NOT a diagnostic task. The attached chest radiograph is either an authentic clinical "
    "acquisition or was generated or edited by an AI model. Estimate the probability, from 0 to 100, "
    "that the image is AI-generated or AI-edited. Reply with JSON only, no other text: "
    '{"probability": <integer 0-100>, "reason": "<one short sentence>"}'
)

# USD prices from the providers' pricing pages, read 2026-11-08 (standard, non-batch).
PRICE_PER_MTOK = {  # (input, output)
    "claude-opus-5-5": (4.0, 20.0), "gpt-6-astra": (10.0, 50.0), "gpt-5.5": (None, None),
    "gemini-4-argon": (2.0, 10.0),
}
GPT_IMAGE_PER_MTOK = {"text_in": 5.0, "image_in": 8.0, "image_out": 30.0}
GEMINI_IMAGE_PER_IMG = {"gemini-3.1-flash-image": 0.067}


def _clean(t: str) -> str:
    t = re.sub(r"\bX+-?X*\b", "", t)  # IU de-identification tokens (XXXX)
    return re.sub(r"\s+", " ", t).strip()


def cmd_prompts():
    rows = [r for r in csv.DictReader(open(DATA / "meta/images.csv"))
            if r["source"] == "openi" and not r["exclude"] and r["split"] == "train"]
    by_report = {}
    for r in rows:
        ex = json.loads(r["extra"])
        if len(ex.get("findings", "")) > 40:
            by_report.setdefault(ex["report_id"], ex)
    random.seed(20261108)
    pick = random.sample(sorted(by_report), 10)
    out = [{"prompt_id": f"pilot{i:02d}", "openi_report": rid,
            "prompt": PROMPT_TEMPLATE.format(findings=_clean(by_report[rid]["findings"]),
                                             impression=_clean(by_report[rid]["impression"]))}
           for i, rid in enumerate(pick)]
    PILOT.mkdir(parents=True, exist_ok=True)
    with open(PILOT / f"prompts_{VERSION}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print(f"wrote {len(out)} prompts -> {PILOT / f'prompts_{VERSION}.csv'}")


def _gen_openai(model, prompt):
    from openai import OpenAI
    params = {"size": "1024x1024", "quality": "medium", "n": 1}
    r = OpenAI().images.generate(model=model, prompt=prompt, **params)
    img = base64.b64decode(r.data[0].b64_json)
    meta = r.model_dump()
    for d in meta.get("data", []):
        d.pop("b64_json", None)
    usage = meta.get("usage")
    return img, "png", params, meta, usage, getattr(r, "model", None)


def _gen_gemini(model, prompt):
    import os
    from google import genai
    from google.genai import types
    c = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    params = {"response_modalities": ["IMAGE"], "aspect_ratio": "1:1"}
    r = c.models.generate_content(model=model, contents=prompt, config=types.GenerateContentConfig(
        response_modalities=["IMAGE"], image_config=types.ImageConfig(aspect_ratio="1:1")))
    img, mime = None, None
    for part in r.candidates[0].content.parts:
        if getattr(part, "inline_data", None) and part.inline_data.data:
            img, mime = part.inline_data.data, part.inline_data.mime_type
            break
    meta = r.model_dump(mode="json", exclude_none=True)
    for cand in meta.get("candidates", []):
        for part in cand.get("content", {}).get("parts", []):
            if "inline_data" in part:
                part["inline_data"]["data"] = f"<{len(img or b'')} bytes, see output_images>"
    usage = meta.get("usage_metadata")
    if img is None:
        raise RuntimeError("no image returned: " + json.dumps(meta)[:500])
    ext = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}.get(mime, "bin")
    return img, ext, params, meta, usage, meta.get("model_version")


def cmd_generate():
    prompts = list(csv.DictReader(open(PILOT / f"prompts_{VERSION}.csv")))
    for gen, (prov, model) in GEN_MODELS.items():
        out_dir = DATA / "raw/api" / gen / f"pilot_{VERSION}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in prompts:
            if list(out_dir.glob(f"{p['prompt_id']}.*")):
                continue
            t = time.time()
            try:
                fn = _gen_openai if prov == "openai" else _gen_gemini
                img, ext, params, meta, usage, ret_model = fn(model, p["prompt"])
                dst = out_dir / f"{p['prompt_id']}.{ext}"
                dst.write_bytes(img)  # untouched raw bytes
                log_call(experiment="pilot_generate", provider=prov, endpoint="images.generate"
                         if prov == "openai" else "models.generate_content", requested_model=model,
                         returned_model=ret_model, prompt=p["prompt"], params=params,
                         response=meta, output_images=[dst], usage=usage)
                print(f"{gen} {p['prompt_id']} ok {time.time() - t:.1f}s -> {dst.name}", flush=True)
            except Exception as e:
                log_call(experiment="pilot_generate", provider=prov, endpoint="generate",
                         requested_model=model, returned_model=None, prompt=p["prompt"], params={},
                         response=None, error=f"{type(e).__name__}: {e}")
                print(f"{gen} {p['prompt_id']} FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)


def _detect_set():
    """20 NIH train PA reals + all pilot fakes, preprocessed to 512 px, metadata-free PNG."""
    det_dir = PILOT / "detect_512"
    items = []
    rows = [r for r in csv.DictReader(open(DATA / "meta/images.csv"))
            if r["source"] == "nih" and not r["exclude"] and r["split"] == "train" and r["view"] == "PA"]
    random.seed(7)
    for r in random.sample(rows, 20):
        items.append({"item": r["image_id"].replace(":", "_"), "label": 0, "source": "nih",
                      "path": DATA / r["out512"]})
    for gen in GEN_MODELS:
        for f in sorted((DATA / "raw/api" / gen / f"pilot_{VERSION}").glob("pilot*.*")):
            dst = det_dir / f"{gen}_{f.stem}.png"
            if not dst.exists():
                save_png(preprocess(f, size=512), dst)
            items.append({"item": f"{gen}_{f.stem}", "label": 1, "source": gen, "path": dst})
    return items


def _b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode()


def _det_anthropic(model, path):
    import anthropic
    params = {"max_tokens": 2000}  # no temperature in SDK 2.6 for Claude 5.x; thinking is on by default
    r = anthropic.Anthropic().messages.create(
        model=model, **params,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _b64(path)}},
            {"type": "text", "text": DETECT_PROMPT}]}])
    text = "".join(b.text for b in r.content if b.type == "text")
    return text, params, r.model_dump(), r.usage.model_dump(), r.model


def _det_openai(model, path):
    from openai import OpenAI
    params = {"max_output_tokens": 2000, "reasoning": {"effort": "low"}}
    r = OpenAI().responses.create(model=model, **params, input=[{"role": "user", "content": [
        {"type": "input_image", "image_url": f"data:image/png;base64,{_b64(path)}"},
        {"type": "input_text", "text": DETECT_PROMPT}]}])
    return r.output_text, params, r.model_dump(), r.usage.model_dump() if r.usage else None, r.model


def _det_gemini(model, path):
    import os
    from google import genai
    from google.genai import types
    c = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    params = {"temperature": 0, "max_output_tokens": 2000}
    r = c.models.generate_content(
        model=model, config=types.GenerateContentConfig(**params),
        contents=[types.Part.from_bytes(data=Path(path).read_bytes(), mime_type="image/png"),
                  DETECT_PROMPT])
    meta = r.model_dump(mode="json", exclude_none=True)
    return r.text or "", params, meta, meta.get("usage_metadata"), meta.get("model_version")


def cmd_detect():
    items = _detect_set()
    out_path = PILOT / "detect_results.jsonl"
    done = set()
    if out_path.exists():
        done = {(json.loads(l)["model"], json.loads(l)["item"]) for l in open(out_path)}
    fns = {"anthropic": _det_anthropic, "openai": _det_openai, "gemini": _det_gemini}
    for prov, model in DET_MODELS:
        use = model
        for it in items:
            if (model, it["item"]) in done:
                continue
            t = time.time()
            try:
                text, params, meta, usage, ret = fns[prov](use, it["path"])
                err = None
            except Exception as e:
                if use in DET_FALLBACK and ("model" in str(e).lower() or "image" in str(e).lower()):
                    print(f"{use} failed ({str(e)[:150]}); falling back to {DET_FALLBACK[use]}")
                    use = DET_FALLBACK[use]
                    text, params, meta, usage, ret = fns[prov](use, it["path"])
                    err = None
                else:
                    text, params, meta, usage, ret, err = "", {}, None, None, None, f"{type(e).__name__}: {e}"
            log_call(experiment="pilot_detect", provider=prov, endpoint="detect", requested_model=use,
                     returned_model=ret, prompt=DETECT_PROMPT, params=params, response=meta,
                     input_images=[it["path"]], usage=usage, error=err)
            m = re.search(r'"probability"\s*:\s*(\d+(?:\.\d+)?)', text or "")
            rec = {"model": model, "used_model": use, "item": it["item"], "label": it["label"],
                   "source": it["source"], "prob": float(m.group(1)) if m else None,
                   "raw_text": text, "error": err, "usage": usage, "secs": round(time.time() - t, 2)}
            with open(out_path, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
            print(f"{use:16s} {it['item']:28s} label={it['label']} prob={rec['prob']} "
                  f"{'ERR ' + err[:80] if err else ''}", flush=True)


def _tok(u, *keys):
    for k in keys:
        if u and u.get(k) is not None:
            return u[k]
    return 0


def cmd_report():
    import numpy as np
    from sklearn.metrics import roc_auc_score
    # generation costs
    gen_cost = {}
    for l in open(LOG_DIR / "pilot_generate.jsonl"):
        r = json.loads(l)
        if r["error"]:
            continue
        m = r["requested_model"]
        if r["provider"] == "openai":
            u = r["usage"] or {}
            det = u.get("input_tokens_details") or {}
            c = (det.get("text_tokens", u.get("input_tokens", 0)) * GPT_IMAGE_PER_MTOK["text_in"]
                 + det.get("image_tokens", 0) * GPT_IMAGE_PER_MTOK["image_in"]
                 + u.get("output_tokens", 0) * GPT_IMAGE_PER_MTOK["image_out"]) / 1e6
        else:
            c = GEMINI_IMAGE_PER_IMG.get(m, 0.067)
        gen_cost.setdefault(m, []).append(c)
    for m, cs in gen_cost.items():
        print(f"GEN {m:24s} n={len(cs)}  cost/img ${np.mean(cs):.4f}  total ${sum(cs):.3f}")
    gen_fail = sum(1 for l in open(LOG_DIR / "pilot_generate.jsonl") if json.loads(l)["error"])
    print("generation failures/refusals:", gen_fail)
    # detection
    res = [json.loads(l) for l in open(PILOT / "detect_results.jsonl")]
    for model in dict.fromkeys(r["model"] for r in res):
        rs = [r for r in res if r["model"] == model]
        ok = [r for r in rs if r["prob"] is not None]
        y = np.array([r["label"] for r in ok]); s = np.array([r["prob"] for r in ok])
        auc = roc_auc_score(y, s) if len(set(y)) == 2 else float("nan")
        acc = float(np.mean((s >= 50) == (y == 1))) if len(ok) else float("nan")
        cost = 0.0
        for r in rs:
            u = r["usage"] or {}
            pin, pout = PRICE_PER_MTOK.get(r["used_model"]) or (None, None)
            if pin is None:
                continue
            ti = _tok(u, "input_tokens", "prompt_token_count")
            to = _tok(u, "output_tokens", "candidates_token_count") + _tok(u, "thoughts_token_count")
            cost += (ti * pin + to * pout) / 1e6
        print(f"DET {model:18s} (used {rs[0]['used_model']}) n={len(rs)} parsed={len(ok)} "
              f"AUC={auc:.3f} acc@50={acc:.2f} mean p(real)={s[y == 0].mean():.1f} "
              f"mean p(fake)={s[y == 1].mean():.1f} cost=${cost:.3f}")


if __name__ == "__main__":
    {"prompts": cmd_prompts, "generate": cmd_generate, "detect": cmd_detect,
     "report": cmd_report}[sys.argv[1]]()
