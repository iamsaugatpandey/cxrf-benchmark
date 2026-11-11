"""Build the content-matched prompt table for all generators (Phase 4).

Source: Open-I reports with a non-trivial FINDINGS section and at least one frontal image.
Order: deterministic (sha256 of report id), so the first k prompts are the same for every
generator -> content is matched ACROSS generators as well as to the real images.
Cleaning: IU de-identification removed words as "XXXX". A text model (Claude Haiku 5.5) rewrites
each report into a short, grammatical findings + impression text without adding findings.
Every call is logged with apilog. Output: data/meta/prompts.csv
  prompt_id, report_id, group_id, findings_raw, impression_raw, text_clean
Usage: python scripts/build_prompts.py [n=2000]
"""
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

from cxrf.apilog import log_call
from cxrf.env import load_env
from cxrf.inventory import openi_reports

load_env()
ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
MODEL = "claude-haiku-5-5"
CHUNK = 20
OUT = D / "meta/prompts.csv"

SYSTEM = (
    "You clean de-identified radiology report text for use as image-generation prompts. "
    "Words removed by de-identification appear as XXXX. For each report, write ONE short, "
    "grammatical text (max 60 words) with the findings followed by the impression. Rules: keep every "
    "stated finding and its laterality; never add a finding; if an XXXX gap cannot be resolved from "
    "context, drop that phrase; remove dates, comparisons to prior studies and recommendations. "
    "Return JSON only: a list of objects {\"id\": <id>, \"text\": <cleaned text>} in the input order."
)


def main():
    rep = openi_reports(D / "raw/openi")
    frontal = {r["image_id"][6:] for r in csv.DictReader(open(D / "meta/openi_views.csv")) if r["view"] == "PA"}
    by_report = {}
    for img, r in rep.items():
        if img in frontal and len(r["findings"].strip()) > 40:
            by_report.setdefault(r["report_id"], r)
    order = sorted(by_report, key=lambda k: hashlib.sha256(f"prompts-v1:{k}".encode()).hexdigest())[:N]

    done = {}
    if OUT.exists():
        done = {r["report_id"]: r for r in csv.DictReader(open(OUT))}
    todo = [k for k in order if k not in done]
    print(f"{len(order)} reports selected; {len(done)} already cleaned; {len(todo)} to do", flush=True)

    import anthropic
    client = anthropic.Anthropic()
    for i in range(0, len(todo), CHUNK):
        batch = todo[i:i + CHUNK]
        payload = [{"id": k, "findings": by_report[k]["findings"], "impression": by_report[k]["impression"]}
                   for k in batch]
        prompt = json.dumps(payload, ensure_ascii=False)
        params = {"max_tokens": 8000, "system": SYSTEM}
        r = client.messages.create(model=MODEL, messages=[{"role": "user", "content": prompt}], **params)
        text = "".join(b.text for b in r.content if b.type == "text")
        log_call(experiment="build_prompts", provider="anthropic", endpoint="messages.create",
                 requested_model=MODEL, returned_model=r.model, prompt=SYSTEM + "\n\n" + prompt,
                 params={"max_tokens": 8000}, response=r.model_dump(), usage=r.usage.model_dump())
        m = re.search(r"\[.*\]", text, re.S)
        items = json.loads(m.group(0)) if m else []
        got = {str(x["id"]): x["text"].strip() for x in items if "id" in x and "text" in x}
        for k in batch:
            if k in got and "XXXX" not in got[k]:
                done[k] = {"report_id": k, "text_clean": got[k]}
        print(f"  {min(i + CHUNK, len(todo))}/{len(todo)} cleaned (this chunk {len(got)}/{len(batch)})", flush=True)

    rows = []
    for j, k in enumerate(order):
        if k not in done:
            continue
        rows.append({"prompt_id": f"p{j:05d}", "report_id": k, "group_id": f"openi:{k}",
                     "findings_raw": by_report[k]["findings"], "impression_raw": by_report[k]["impression"],
                     "text_clean": done[k]["text_clean"]})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} prompts -> {OUT}")


if __name__ == "__main__":
    main()
