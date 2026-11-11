"""Commercial-API generation for the confirmatory study (Phase 4). DRY-RUN BY DEFAULT.

Arms
  text   : text-to-image from data/meta/prompts.csv (prompt v2 wording; same prompt_ids as the
           open generators -> content matched). G4 = gpt-image-2 (medium, 1024^2, PNG);
           G5 = gemini-3.1-flash-image (1:1).
  cond   : image-conditioned edit of a NORMAL test-split radiograph from each hospital
           ("make this radiograph show <findings>"), so the fake inherits that hospital's
           appearance. Source normals: NIH 'No Finding', Open-I reports with normal impression,
           Shenzhen/Montgomery tb == 0. CheXpert is EXCLUDED from this arm until its research-use
           agreement is checked for uploading images to third-party APIs.

Every call -> apilog (model snapshot, params, usage, sha256 of input/output); raw outputs saved
untouched under data/raw/api/<GEN>/<arm>/. Resumable. Threads for concurrency.

Usage:
  python scripts/gen_api.py text --n 600            # prints plan + cost estimate only
  python scripts/gen_api.py cond --per-site 100      # prints plan + cost estimate only
  add --execute to actually call the APIs
"""
import argparse
import base64
import csv
import io
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cxrf.apilog import log_call
from cxrf.env import load_env

load_env()
ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
PRICE = {"G4": 0.0534, "G5": 0.067}  # measured per-image cost in the pilot (USD, standard tier)
MODELS = {"G4": "gpt-image-2", "G5": "gemini-3.1-flash-image"}
GPT_IMAGE_PER_MTOK = {"text_in": 5.0, "image_in": 8.0, "image_out": 30.0}  # pricing page, 2026-11-08
SPENT = {"G4": 0.0, "G5": 0.0}
LOCK = threading.Lock()


def actual_cost(gen, resp):
    """USD for one successful call, from the usage the API returned (G4) or the per-image price (G5)."""
    if gen == "G5":
        return 0.067
    u = getattr(resp, "usage", None)
    u = u.model_dump() if hasattr(u, "model_dump") else (u or {})
    det = u.get("input_tokens_details") or {}
    txt = det.get("text_tokens", u.get("input_tokens", 0) - det.get("image_tokens", 0))
    return (txt * GPT_IMAGE_PER_MTOK["text_in"] + det.get("image_tokens", 0) * GPT_IMAGE_PER_MTOK["image_in"]
            + u.get("output_tokens", 0) * GPT_IMAGE_PER_MTOK["image_out"]) / 1e6

TEXT_TEMPLATE = (
    "A single realistic frontal chest X-ray radiograph of an adult, grayscale. The radiograph fills the "
    "entire image: chest only, from the lung apices to the costophrenic angles. No viewer software, no user "
    "interface, no borders, no second image, not a photograph of a screen. The radiology report for this "
    "image reads: {text}"
)
COND_TEMPLATE = (
    "Edit this chest X-ray radiograph so that it shows the following findings: {text} "
    "Keep everything else unchanged: the same patient anatomy, positioning, exposure, contrast, noise, "
    "burned-in markers and image style. Return one radiograph only, same framing, no annotations."
)


def _openai_text(prompt):
    from openai import OpenAI
    params = {"size": "1024x1024", "quality": "medium", "n": 1}
    r = OpenAI().images.generate(model=MODELS["G4"], prompt=prompt, **params)
    return base64.b64decode(r.data[0].b64_json), "png", params, r


def _png_bytes(src):
    """Raw file -> PNG bytes, pixels unchanged (grayscale kept; palette/other modes -> RGB)."""
    from PIL import Image
    im = Image.open(src)
    if im.mode not in ("L", "RGB", "RGBA", "I;16"):
        im = im.convert("RGB")
    b = io.BytesIO()
    im.save(b, format="PNG")
    return b.getvalue()


def _openai_cond(prompt, src):
    from openai import OpenAI
    params = {"size": "1024x1024", "quality": "medium", "n": 1}
    f = ("source.png", _png_bytes(src), "image/png")
    r = OpenAI().images.edit(model=MODELS["G4"], image=f, prompt=prompt, **params)
    return base64.b64decode(r.data[0].b64_json), "png", params, r


def _gemini(prompt, src=None):
    import os
    from google import genai
    from google.genai import types
    c = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    params = {"response_modalities": ["IMAGE"], "aspect_ratio": "1:1"}
    contents = [prompt] if src is None else [types.Part.from_bytes(data=_png_bytes(src), mime_type="image/png"), prompt]
    r = c.models.generate_content(model=MODELS["G5"], contents=contents, config=types.GenerateContentConfig(
        response_modalities=["IMAGE"], image_config=types.ImageConfig(aspect_ratio="1:1")))
    cand = (r.candidates or [None])[0]
    if cand is None or cand.content is None or not cand.content.parts:
        fr = getattr(cand, "finish_reason", None) if cand is not None else None
        pf = getattr(r, "prompt_feedback", None)
        raise RuntimeError(f"no image (refusal/block): finish_reason={fr}, prompt_feedback={pf}")
    for part in cand.content.parts:
        if getattr(part, "inline_data", None) and part.inline_data.data:
            ext = {"image/png": "png", "image/jpeg": "jpg"}.get(part.inline_data.mime_type, "bin")
            return part.inline_data.data, ext, params, r
    raise RuntimeError("no image returned (refusal or text-only response)")


def _redact(resp):
    try:
        m = resp.model_dump(mode="json", exclude_none=True)
    except TypeError:
        m = resp.model_dump()
    s = json.dumps(m, default=str)
    if len(s) > 20000:  # drop inlined image bytes
        for d in m.get("data", []) or []:
            d.pop("b64_json", None)
        for cand in m.get("candidates", []) or []:
            for p in cand.get("content", {}).get("parts", []):
                if "inline_data" in p:
                    p["inline_data"]["data"] = "<omitted>"
    return m


def job(gen, arm, item, out_dir, cap):
    dst_stem = out_dir / item["id"]
    if list(out_dir.glob(item["id"] + ".*")):
        return "skip"
    with LOCK:
        if SPENT[gen] >= cap:
            return f"BUDGET-STOP (spent ${SPENT[gen]:.2f} >= cap ${cap:.2f})"
    prompt = (TEXT_TEMPLATE if arm == "text" else COND_TEMPLATE).format(text=item["text"])
    src = item.get("src")
    t = time.time()
    try:
        if gen == "G4":
            img, ext, params, r = _openai_text(prompt) if arm == "text" else _openai_cond(prompt, src)
        else:
            img, ext, params, r = _gemini(prompt, src)
        dst = Path(f"{dst_stem}.{ext}")
        dst.write_bytes(img)
        log_call(experiment=f"gen_{arm}", provider="openai" if gen == "G4" else "gemini", endpoint=arm,
                 requested_model=MODELS[gen], returned_model=getattr(r, "model", None) or getattr(r, "model_version", None),
                 prompt=prompt, params=params, response=_redact(r), input_images=[src] if src else None,
                 output_images=[dst], usage=getattr(r, "usage", None) and _redact(r).get("usage"))
        c = actual_cost(gen, r)
        with LOCK:
            SPENT[gen] += c
        return f"ok {time.time() - t:.0f}s ${c:.4f} (total ${SPENT[gen]:.2f})"
    except Exception as e:
        log_call(experiment=f"gen_{arm}", provider="openai" if gen == "G4" else "gemini", endpoint=arm,
                 requested_model=MODELS[gen], returned_model=None, prompt=prompt, params={}, response=None,
                 input_images=[src] if src else None, error=f"{type(e).__name__}: {e}")
        return f"FAIL {type(e).__name__}: {str(e)[:120]}"


def text_items(n):
    rows = list(csv.DictReader(open(D / "meta/prompts.csv")))[:n]
    return [{"id": r["prompt_id"], "text": r["text_clean"], "group_id": r["group_id"]} for r in rows]


def cond_items(per_site, seed=7):
    img = list(csv.DictReader(open(D / "meta/images.csv")))
    prompts = [p for p in csv.DictReader(open(D / "meta/prompts.csv"))
               if not any(k in p["text_clean"].lower() for k in ("no acute", "normal chest", "lungs are clear. no"))]
    rng = random.Random(seed)
    items = []
    for site, ok in [("nih", lambda r: json.loads(r["extra"])["findings"] == "No Finding"),
                     ("openi", lambda r: "normal" in json.loads(r["extra"]).get("impression", "").lower()
                      or "no acute" in json.loads(r["extra"]).get("impression", "").lower()),
                     ("shenzhen", lambda r: json.loads(r["extra"]).get("tb") == 0),
                     ("montgomery", lambda r: json.loads(r["extra"]).get("tb") == 0)]:
        pool = [r for r in img if r["source"] == site and not r["exclude"] and r["split"] == "test" and ok(r)]
        pick = rng.sample(pool, min(per_site, len(pool)))
        for r in pick:
            p = rng.choice(prompts)
            items.append({"id": f"{site}_{r['image_id'].split(':', 1)[1]}", "text": p["text_clean"],
                          # the RAW original file, as a forger would use (not our preprocessed copy,
                          # which would put cond fakes through the pipeline twice)
                          "src": str(D / r["raw_path"]),
                          "group_id": r["group_id"], "site": site, "prompt_id": p["prompt_id"]})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=["text", "cond"])
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--per-site", type=int, default=100)
    ap.add_argument("--gens", default="G4,G5")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--cap-g4", type=float, default=55.0, help="max USD this run on gpt-image (OpenAI balance $80)")
    ap.add_argument("--cap-g5", type=float, default=60.0, help="max USD this run on Gemini image")
    ap.add_argument("--limit", type=int, default=0, help="only the first k items (smoke test)")
    a = ap.parse_args()
    items = text_items(a.n) if a.arm == "text" else cond_items(a.per_site)
    if a.limit:
        items = items[:a.limit]
    gens = a.gens.split(",")
    cost = sum(PRICE[g] for g in gens) * len(items)
    print(f"arm={a.arm} items={len(items)} generators={gens} calls={len(items) * len(gens)} est. cost ${cost:.2f}")
    if a.arm == "cond":
        from collections import Counter
        print("per site:", Counter(i["site"] for i in items))
    manifest = D / "raw/api" / f"manifest_{a.arm}.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for i in items for k in i}))
        w.writeheader()
        w.writerows(items)
    print("manifest:", manifest)
    if not a.execute:
        print("DRY RUN: no API calls made. Re-run with --execute after approval.")
        return
    for g in gens:
        out = D / "raw/api" / g / ("cond_v2" if a.arm == "cond" else a.arm)
        out.mkdir(parents=True, exist_ok=True)
        # cumulative cap: count spend already on disk for this generator x arm (measured unit prices)
        unit = {"G4": 0.0532 if a.arm == "text" else 0.0615, "G5": 0.067}[g]
        SPENT[g] = unit * sum(1 for _ in out.iterdir())
        print(f"== {g} {a.arm}: already on disk ≈ ${SPENT[g]:.2f}", flush=True)
        with ThreadPoolExecutor(a.threads) as ex:
            cap = a.cap_g4 if g == "G4" else a.cap_g5
            for it, res in zip(items, ex.map(lambda it: job(g, a.arm, it, out, cap), items)):
                print(g, it["id"], res, flush=True)
        print(f"== {g} {a.arm}: cumulative ≈ ${SPENT[g]:.2f}", flush=True)


if __name__ == "__main__":
    main()
