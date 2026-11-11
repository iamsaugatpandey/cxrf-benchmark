"""Phase 7, confirmatory: frontier multimodal LLMs as zero-shot / 4-shot detectors (OSF 5vwkj, H5).

Protocol (fixed before any call; see docs/llm_protocol.md for the rationale and citations)
  images   : data/meta/llm_subset.csv (2,000 test images, frozen; CheXpert excluded). Each image is
             the same 256-px preprocessed, metadata-free PNG that the trained detectors score.
  models   : claude-opus-5-5, claude-haiku-5-5, gpt-6-astra, gpt-5.4-mini, gemini-4-argon,
             gemini-3.1-flash-lite (snapshot IDs returned by the APIs are stored).
  prompt   : SYSTEM + USER below, byte-identical for every model; one image per call. The model
             is told the set contains authentic and AI-generated/edited radiographs, prevalence
             undisclosed (as in Lee et al. 2026), and returns a verbalised probability (0-100)
             plus a one-sentence cue, as JSON enforced by each provider's structured-output API.
  settings : provider default sampling (no temperature/top_p set anywhere); reasoning effort
             'low' on every model (uniform across providers; see protocol note).
  runs     : zeroshot r1 on all 2,000 (primary); zeroshot r2, r3 on the retest subset (403);
             fourshot r1 on the 4-shot subset (503) with 4 fixed labelled training-split examples.
  budget   : batch APIs (50% price). Chunks are submitted in preregistered order
             (r1 -> r2 -> r3 -> fourshot); before each chunk the projected spend is checked
             against the provider cap (OpenAI $25, Anthropic $25, Google $25). If a chunk would
             exceed the cap, that provider stops there and only completed chunks are analysed.

Files
  data/llm/requests/<model>/<chunk>.jsonl     exact request bodies (images base64)   [build]
  data/llm/raw/<model>/<chunk>.jsonl          raw provider results, untouched        [run]
  data/llm/state_<provider>.json              batch ids, status, cost per chunk
  results/scores/llm/llm_scores.csv      parsed: model, chunk, run, image_id, probability,
                                              reason, status, tokens, cost            [parse]
Usage
  python scripts/run_llm.py build                 # write request files + cost projection
  python scripts/run_llm.py smoke                 # 2 synchronous calls per model on TRAIN images
  python scripts/run_llm.py run <provider>        # anthropic | openai | google (blocking; resumable)
  python scripts/run_llm.py parse
"""
import base64
import json
import sys
import time
from pathlib import Path

import pandas as pd

from cxrf.env import load_env

load_env()
ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
L = D / "llm"
OUTCSV = ROOT / "results/scores/llm/llm_scores.csv"

PROMPT_VERSION = "detect-v3 (2026-11-09)"
SYSTEM = ("You are taking part in an image-forensics research study on the provenance of chest radiographs. "
          "This is not a diagnostic task, and no clinical decision depends on your answer.")
USER = ("The chest radiograph below comes from a set that contains both authentic radiographs acquired from "
        "patients and radiographs that were generated or edited by AI models. The proportion of each is not "
        "disclosed. Every image in the set has been converted to the same format (256 x 256 pixels, grayscale, "
        "borders cropped, no metadata), so file format and size carry no information.\n"
        "Estimate the probability, from 0 to 100, that this radiograph was generated or edited by AI. "
        "0 means certainly authentic, 100 means certainly AI-generated or AI-edited, and 50 means you cannot "
        "tell. Then state the main visual cue for your judgment in one short sentence.")
FEWSHOT_INTRO = ("First, here are four labelled example radiographs from the same set, to show what authentic and "
                 "AI-generated images in this set look like.")
SCHEMA = {"type": "object",
          "properties": {"probability": {"type": "integer"},  # range 0-100 stated in the prompt, checked in parse_one
                         "reason": {"type": "string"}},
          "required": ["probability", "reason"], "additionalProperties": False}

MODELS = {  # model: (provider, $/MTok input, $/MTok output) at standard price; batch = 50%
    "claude-opus-5-5": ("anthropic", 4.0, 20.0), "claude-haiku-5-5": ("anthropic", 0.10, 0.50),
    "gpt-6-astra": ("openai", 10.0, 50.0), "gpt-5.4-mini": ("openai", 0.75, 4.50),
    "gemini-4-argon": ("google", 2.0, 10.0), "gemini-3.1-flash-lite": ("google", 0.25, 1.50),
}
CAP = {"anthropic": 25.0, "openai": 25.0, "google": 25.0}
BATCH_DISCOUNT = 0.5
MAX_TOK = 4000
OPENAI_FILE_MB = 90


def b64(rel):
    return base64.b64encode((D / rel).read_bytes()).decode()


def chunks():
    """Preregistered order. Returns [(chunk_name, condition, run, DataFrame)]."""
    s = pd.read_csv(D / "meta/llm_subset.csv").sort_values("run_order")
    return [("zeroshot_r1", "zeroshot", 1, s), ("zeroshot_r2", "zeroshot", 2, s[s.retest]),
            ("zeroshot_r3", "zeroshot", 3, s[s.retest]), ("fourshot_r1", "fourshot", 1, s[s.fourshot])]


def fewshot_blocks():
    ex = pd.read_csv(D / "meta/llm_fewshot_examples.csv").set_index("slot")
    order = [("real_nih", "authentic"), ("fake_g1", "AI-generated"), ("real_openi", "authentic"),
             ("fake_g2", "AI-generated")]
    return [(f"Example {i + 1}: {lab} radiograph.", ex.loc[k, "out256"]) for i, (k, lab) in enumerate(order)]


def content_parts(img_rel, condition):
    """Provider-neutral list of ('text', str) / ('image', rel_path) in presentation order."""
    parts = []
    if condition == "fourshot":
        parts.append(("text", FEWSHOT_INTRO))
        for t, p in fewshot_blocks():
            parts += [("text", t), ("image", p)]
        parts.append(("text", "Now the radiograph to assess:"))
    parts += [("image", img_rel), ("text", USER)]
    return parts


def req_anthropic(model, parts):
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64(v)}}
               if k == "image" else {"type": "text", "text": v} for k, v in parts]
    return {"model": model, "max_tokens": MAX_TOK, "system": SYSTEM,
            "messages": [{"role": "user", "content": content}],
            "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}}}


def req_openai(model, parts):
    content = [{"type": "input_image", "image_url": f"data:image/png;base64,{b64(v)}"}
               if k == "image" else {"type": "input_text", "text": v} for k, v in parts]
    return {"model": model, "instructions": SYSTEM, "max_output_tokens": MAX_TOK,
            "reasoning": {"effort": "low"},
            "input": [{"role": "user", "content": content}],
            "text": {"format": {"type": "json_schema", "name": "authenticity", "schema": SCHEMA, "strict": True}}}


def req_google(model, parts):
    p = [{"inlineData": {"mimeType": "image/png", "data": b64(v)}} if k == "image" else {"text": v}
         for k, v in parts]
    schema = {k: v for k, v in SCHEMA.items() if k != "additionalProperties"}
    return {"systemInstruction": {"parts": [{"text": SYSTEM}]},
            "contents": [{"role": "user", "parts": p}],
            "generationConfig": {"maxOutputTokens": MAX_TOK, "responseMimeType": "application/json",
                                 "responseJsonSchema": schema, "thinkingConfig": {"thinkingLevel": "low"}}}


BUILD = {"anthropic": req_anthropic, "openai": req_openai, "google": req_google}


def cid(chunk, run_order):
    return f"{chunk}-{int(run_order):04d}"


def state_path(prov):  # one file per provider: the three runs are separate processes
    return L / f"state_{prov}.json"


def load_state(prov):
    p = state_path(prov)
    return json.loads(p.read_text()) if p.exists() else {}


def save_state(prov, st):
    p = state_path(prov)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1))
    tmp.replace(p)  # atomic


# ------------------------------------------------------------------ build
PILOT_TOK = {"claude-opus-5-5": (516, 250), "gpt-6-astra": (409, 64), "gemini-4-argon": (1221, 63)}  # pilot means


def projected(model, cond, n):
    """Build-time cost projection (USD, batch price) from pilot token counts (512-px images: an upper bound)."""
    _, pin, pout = MODELS[model]
    tin, tout = PILOT_TOK.get(model, (900, 250))
    return n * (tin * (5 if cond == "fourshot" else 1) * pin + tout * pout) / 1e6 * BATCH_DISCOUNT


def cmd_build():
    tot = {}
    for model, (prov, pin, pout) in MODELS.items():
        out = L / "requests" / model
        out.mkdir(parents=True, exist_ok=True)
        for name, cond, run, df in chunks():
            with open(out / f"{name}.jsonl", "w") as f:
                for r in df.itertuples():
                    f.write(json.dumps({"custom_id": cid(name, r.run_order),
                                        "body": BUILD[prov](model, content_parts(r.out256, cond))}) + "\n")
            c = projected(model, cond, len(df))
            tot.setdefault(prov, 0.0)
            tot[prov] += c
            print(f"{model:22s} {name:12s} n={len(df):5d}  projected ${c:6.2f}")
    print("projected batch cost by provider:", {k: round(v, 2) for k, v in tot.items()}, "caps:", CAP)


# ------------------------------------------------------------------ smoke (sync, TRAIN images only)
def call_sync(model, body):
    prov = MODELS[model][0]
    if prov == "anthropic":
        import anthropic
        r = anthropic.Anthropic().messages.create(**body)
        return r.model_dump(mode="json")
    if prov == "openai":
        from openai import OpenAI
        return OpenAI().responses.create(**body).model_dump(mode="json")
    import os
    import requests
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    for wait in (0, 20, 60, 120, 240):  # back off on 429 (synchronous smoke calls only)
        time.sleep(wait)
        r = requests.post(url, json=body, headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]}, timeout=300)
        if r.status_code != 429:
            break
    r.raise_for_status()
    return r.json()


def cmd_smoke():
    m = pd.read_csv(D / "meta/images.csv", keep_default_na=False, dtype=str)
    ex = set(pd.read_csv(D / "meta/llm_fewshot_examples.csv").image_id)
    tr = m[(m.exclude == "") & (m.split == "train") & ~m.image_id.isin(ex)]
    picks = [tr[tr.source == "nih"].iloc[7].out256, tr[tr.source == "synthchex"].iloc[7].out256]
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else list(MODELS)
    for model in only:
        for i, p in enumerate(picks):
            cond = "fourshot" if i else "zeroshot"
            body = BUILD[MODELS[model][0]](model, content_parts(p, cond))
            raw = call_sync(model, body)
            rec = parse_one(model, raw)
            print(f"SMOKE {model:22s} {cond:8s} prob={rec['probability']} status={rec['status']} "
                  f"tok={rec['in_tok']}/{rec['out_tok']} reason={str(rec['reason'])[:80]!r}", flush=True)


# ------------------------------------------------------------------ parsing
def parse_one(model, raw):
    """raw = one provider response body (dict). Returns probability, reason, status, tokens, snapshot."""
    prov = MODELS[model][0]
    text, status, snap, tin, tout = "", "ok", None, 0, 0
    try:
        if prov == "anthropic":
            snap = raw.get("model")
            u = raw.get("usage") or {}
            tin, tout = u.get("input_tokens", 0), u.get("output_tokens", 0)
            if raw.get("stop_reason") == "refusal":
                status = "refusal"
            text = "".join(b.get("text", "") for b in raw.get("content", []) if b.get("type") == "text")
        elif prov == "openai":
            snap = raw.get("model")
            u = raw.get("usage") or {}
            tin, tout = u.get("input_tokens", 0), u.get("output_tokens", 0)
            for o in raw.get("output", []):
                for c in o.get("content", []) or []:
                    if c.get("type") == "output_text":
                        text += c.get("text", "")
                    elif c.get("type") == "refusal":
                        status = "refusal"
        else:
            snap = raw.get("modelVersion") or raw.get("model_version")
            u = raw.get("usageMetadata") or raw.get("usage_metadata") or {}
            tin = u.get("promptTokenCount", u.get("prompt_token_count", 0))
            tout = (u.get("candidatesTokenCount", u.get("candidates_token_count", 0)) or 0) + \
                   (u.get("thoughtsTokenCount", u.get("thoughts_token_count", 0)) or 0)
            cands = raw.get("candidates") or []
            if not cands or not (cands[0].get("content") or {}).get("parts"):
                status = "refusal"
            else:
                text = "".join(p.get("text", "") for p in cands[0]["content"]["parts"] if not p.get("thought"))
        j = json.loads(text) if text and status == "ok" else {}
        prob = j.get("probability")
        if status == "ok" and (not isinstance(prob, (int, float)) or not 0 <= prob <= 100):
            status = "unparseable"
    except (json.JSONDecodeError, AttributeError, TypeError):
        j, prob, status = {}, None, "unparseable"
    pin, pout = MODELS[model][1:]
    return {"probability": prob, "reason": j.get("reason"), "status": status, "snapshot": snap,
            "in_tok": tin, "out_tok": tout, "raw_text": text[:2000],
            "cost": (tin * pin + tout * pout) / 1e6 * BATCH_DISCOUNT}


# ------------------------------------------------------------------ batch backends
def submit_anthropic(model, name):
    import anthropic
    reqs = [json.loads(l) for l in open(L / "requests" / model / f"{name}.jsonl")]
    b = anthropic.Anthropic().messages.batches.create(
        requests=[{"custom_id": r["custom_id"], "params": r["body"]} for r in reqs])
    return [b.id]


def poll_anthropic(ids):
    import anthropic
    return all(anthropic.Anthropic().messages.batches.retrieve(i).processing_status == "ended" for i in ids)


def fetch_anthropic(ids, dst):
    import anthropic
    with open(dst, "w") as f:
        for i in ids:
            for r in anthropic.Anthropic().messages.batches.results(i):
                d = r.model_dump(mode="json")
                f.write(json.dumps({"custom_id": d["custom_id"], "type": d["result"]["type"],
                                    "response": d["result"].get("message"), "error": d["result"].get("error")}) + "\n")


def submit_openai(model, name):
    from openai import OpenAI
    c = OpenAI()
    lines = open(L / "requests" / model / f"{name}.jsonl").read().splitlines()
    ids, part, size, k = [], [], 0, 0

    def flush(part, k):
        p = L / "requests" / model / f"{name}.openai{k}.jsonl"
        with open(p, "w") as f:
            for l in part:
                r = json.loads(l)
                f.write(json.dumps({"custom_id": r["custom_id"], "method": "POST", "url": "/v1/responses",
                                    "body": r["body"]}) + "\n")
        fo = c.files.create(file=open(p, "rb"), purpose="batch")
        return c.batches.create(input_file_id=fo.id, endpoint="/v1/responses", completion_window="24h").id

    for l in lines:
        if size + len(l) > OPENAI_FILE_MB * 1e6 and part:
            ids.append(flush(part, k))
            part, size, k = [], 0, k + 1
        part.append(l)
        size += len(l)
    ids.append(flush(part, k))
    return ids


def poll_openai(ids):
    from openai import OpenAI
    st = [OpenAI().batches.retrieve(i).status for i in ids]
    bad = [s for s in st if s in ("failed", "expired", "cancelled")]
    if bad:
        raise RuntimeError(f"OpenAI batch status {st}")
    return all(s == "completed" for s in st)


def fetch_openai(ids, dst):
    from openai import OpenAI
    c = OpenAI()
    with open(dst, "w") as f:
        for i in ids:
            b = c.batches.retrieve(i)
            for fid in [b.output_file_id, b.error_file_id]:
                if not fid:
                    continue
                for l in c.files.content(fid).text.splitlines():
                    r = json.loads(l)
                    resp = (r.get("response") or {})
                    f.write(json.dumps({"custom_id": r["custom_id"],
                                        "type": "succeeded" if resp.get("status_code") == 200 else "errored",
                                        "response": resp.get("body"), "error": r.get("error")}) + "\n")


def _genai():
    import os
    from google import genai
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def submit_google(model, name):
    c = _genai()
    p = L / "requests" / model / f"{name}.google.jsonl"
    with open(p, "w") as f:
        for l in open(L / "requests" / model / f"{name}.jsonl"):
            r = json.loads(l)
            f.write(json.dumps({"key": r["custom_id"], "request": r["body"]}) + "\n")
    fo = c.files.upload(file=str(p), config={"mime_type": "jsonl", "display_name": f"{model}-{name}"})
    b = c.batches.create(model=f"models/{model}", src=fo.name, config={"display_name": f"cxrf-{model}-{name}"})
    return [b.name]


def poll_google(ids):
    c = _genai()
    st = [c.batches.get(name=i).state.name for i in ids]
    if any(s in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED") for s in st):
        raise RuntimeError(f"Google batch state {st}")
    return all(s == "JOB_STATE_SUCCEEDED" for s in st)


def fetch_google(ids, dst):
    c = _genai()
    with open(dst, "w") as f:
        for i in ids:
            b = c.batches.get(name=i)
            data = c.files.download(file=b.dest.file_name).decode()
            for l in data.splitlines():
                r = json.loads(l)
                f.write(json.dumps({"custom_id": r.get("key"), "type": "errored" if r.get("error") else "succeeded",
                                    "response": r.get("response"), "error": r.get("error")}) + "\n")


BACK = {"anthropic": (submit_anthropic, poll_anthropic, fetch_anthropic),
        "openai": (submit_openai, poll_openai, fetch_openai),
        "google": (submit_google, poll_google, fetch_google)}


def chunk_cost(model, dst):
    return sum(parse_one(model, r["response"])["cost"] for r in map(json.loads, open(dst)) if r["response"])


def cmd_run(prov):
    submit, poll, fetch = BACK[prov]
    models = [m for m, v in MODELS.items() if v[0] == prov]
    st = load_state(prov)
    for name, cond, run, df in chunks():
        spent = sum(v.get("cost", 0) for k, v in st.items() if MODELS[k.split("|")[0]][0] == prov)
        # projected cost of this chunk for all of this provider's models: measured per-call cost if we
        # have it (x1.25 for 4-shot = 5 images), else the build-time projection
        proj = 0.0
        for model in models:
            done = [v for k, v in st.items() if k.startswith(model + "|") and v.get("cost") is not None]
            per = (sum(v["cost"] for v in done) / sum(v["n"] for v in done)) if done else None
            if not st.get(f"{model}|{name}", {}).get("fetched"):
                proj += per * (4.0 if cond == "fourshot" else 1.0) * len(df) if per else projected(model, cond, len(df))
        if spent + proj > CAP[prov]:
            print(f"[{prov}] STOP before {name}: spent ${spent:.2f} + projected ${proj:.2f} > cap ${CAP[prov]}",
                  flush=True)
            st[f"_{prov}_stopped_before"] = {"chunk": name, "spent": spent, "projected": proj}
            save_state(prov, st)
            return
        for model in models:  # submit all of this provider's models for this chunk together
            k = f"{model}|{name}"
            if k not in st:
                ids = submit(model, name)
                st[k] = {"batch_ids": ids, "submitted": time.strftime("%Y-%m-%d %H:%M:%S"), "n": len(df)}
                save_state(prov, st)
                print(f"[{prov}] submitted {k}: {ids}", flush=True)
        for model in models:
            k = f"{model}|{name}"
            while not st[k].get("fetched"):
                try:
                    ready = poll(st[k]["batch_ids"])
                except (OSError, ConnectionError) as e:  # transient network errors: retry, never resubmit
                    print(f"[{prov}] poll error ({type(e).__name__}); retrying", flush=True)
                    ready = False
                except Exception as e:
                    if "ReadError" in type(e).__name__ or "Connect" in type(e).__name__ or "Timeout" in type(e).__name__:
                        print(f"[{prov}] poll error ({type(e).__name__}); retrying", flush=True)
                        ready = False
                    else:
                        raise
                if ready:
                    dst = L / "raw" / model / f"{name}.jsonl"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    fetch(st[k]["batch_ids"], dst)
                    st[k].update(fetched=time.strftime("%Y-%m-%d %H:%M:%S"), cost=chunk_cost(model, dst))
                    save_state(prov, st)
                    print(f"[{prov}] fetched {k}: ${st[k]['cost']:.2f}", flush=True)
                else:
                    time.sleep(120)
    print(f"[{prov}] all chunks done", flush=True)


def cmd_parse():
    sub = pd.read_csv(D / "meta/llm_subset.csv").set_index("run_order")
    rows = []
    for model in MODELS:
        for f in sorted((L / "raw" / model).glob("*.jsonl")):
            cond, run = f.stem.split("_r")
            for r in map(json.loads, open(f)):
                ro = int(r["custom_id"].rsplit("-", 1)[1])
                base = {"model": model, "condition": cond, "run": int(run), "run_order": ro,
                        "image_id": sub.loc[ro, "image_id"], "stratum": sub.loc[ro, "stratum"],
                        "cls": sub.loc[ro, "cls"]}
                if r["type"] != "succeeded" or not r["response"]:
                    rows.append({**base, "status": f"api_{r['type']}", "error": json.dumps(r.get("error"))[:300]})
                else:
                    rows.append({**base, **parse_one(model, r["response"])})
    OUTCSV.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUTCSV, index=False)
    print(out.groupby(["model", "condition", "run"]).agg(n=("image_id", "size"),
          ok=("status", lambda s: (s == "ok").sum()), cost=("cost", "sum")))


if __name__ == "__main__":
    {"build": cmd_build, "smoke": cmd_smoke, "parse": cmd_parse,
     "run": lambda: cmd_run(sys.argv[2])}[sys.argv[1]]()
