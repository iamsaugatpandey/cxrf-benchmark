"""Raw-image inventory (Pipeline step 3, first half): one row per raw image.

Columns
  image_id     globally unique, "<source>:<native id>"
  source       dataset name (nih, montgomery, shenzhen, openi, synthchex, ...)
  site         acquisition site for reals; "synthetic" for fakes
  cls          "real" | "fake" | "tampered"
  generator    "" for reals, G1..G6 for fakes
  patient_id   patient key where the dataset has one ("" for fakes)
  group_id     resampling / split unit. Patient for reals; for fakes, the source
               report/study the generation prompt came from (so a prompt and its source
               never cross splits and bootstrap resamples them together)
  view         PA | AP | lateral | unknown
  raw_path     relative to the data root
  raw_sha256   from the dataset's MANIFEST.sha256 when present, else computed
  width, height, mode   as stored on disk
  extra        dataset-specific JSON (findings, TB label, MIMIC labels, ...)
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

from PIL import Image

COLUMNS = ["image_id", "source", "site", "cls", "generator", "patient_id", "group_id", "view",
           "raw_path", "raw_sha256", "width", "height", "mode", "extra"]


def _manifest(dirpath: Path) -> dict[str, str]:
    m = dirpath / "MANIFEST.sha256"
    out = {}
    if m.exists():
        for line in m.read_text().splitlines():
            h, p = line.split(maxsplit=1)
            out[str((dirpath / p.strip()).resolve())] = h
    return out


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def _row(root: Path, p: Path, man: dict, **kw) -> dict:
    with Image.open(p) as im:
        w, h, mode = im.width, im.height, im.mode
    sha = man.get(str(p.resolve())) or _sha(p)
    r = {c: "" for c in COLUMNS}
    r.update(raw_path=str(p.relative_to(root)), raw_sha256=sha, width=w, height=h, mode=mode)
    r.update({k: (json.dumps(v) if k == "extra" else v) for k, v in kw.items()})
    return r


def scan_nih(root: Path):
    d = root / "raw" / "nih"
    meta = {}
    with open(d / "data" / "Data_Entry_2017_v2020.csv") as f:
        for r in csv.DictReader(f):
            meta[r["Image Index"]] = r
    man = _manifest(d)
    for p in sorted((d / "images").glob("images_*/images/*.png")):
        m = meta[p.name]
        pid = f"nih:{int(m['Patient ID'])}"
        yield _row(root, p, man, image_id=f"nih:{p.stem}", source="nih", site="NIH-CC", cls="real",
                   patient_id=pid, group_id=pid, view=m["View Position"],
                   extra={"findings": m["Finding Labels"], "age": m["Patient Age"],
                          "sex": m["Patient Gender"], "followup": m["Follow-up #"]})


def _scan_nlm_tb(root: Path, name: str, site: str):
    d = root / "raw" / name
    man = _manifest(d)
    for p in sorted((d / "CXR_png").glob("*.png")):
        pid = f"{name}:{p.stem}"
        yield _row(root, p, man, image_id=pid, source=name, site=site, cls="real",
                   patient_id=pid, group_id=pid, view="PA",
                   extra={"tb": int(p.stem.rsplit("_", 1)[1])})


def scan_montgomery(root: Path):
    yield from _scan_nlm_tb(root, "montgomery", "Montgomery-US")


def scan_shenzhen(root: Path):
    yield from _scan_nlm_tb(root, "shenzhen", "Shenzhen-CN")


def openi_reports(d: Path) -> dict[str, dict]:
    """image stem -> {report_id, findings, impression}. IU has no patient IDs; one report
    = one study, so the report is the patient-level unit."""
    out = {}
    for x in (d / "reports" / "ecgen-radiology").glob("*.xml"):
        t = x.read_text(errors="replace")
        sec = dict(re.findall(r'<AbstractText Label="([A-Z]+)">([^<]*)', t))
        for img in re.findall(r'<parentImage id="([^"]+)"', t):
            out[img] = {"report_id": x.stem, "findings": sec.get("FINDINGS", ""),
                        "impression": sec.get("IMPRESSION", "")}
    return out


def scan_openi(root: Path, views: dict[str, str] | None = None):
    d = root / "raw" / "openi"
    rep = openi_reports(d)
    views = views or {}
    for p in sorted((d / "png").rglob("*.png")):
        r = rep.get(p.stem, {})
        gid = f"openi:{r.get('report_id', p.stem)}"
        yield _row(root, p, {}, image_id=f"openi:{p.stem}", source="openi", site="Indiana-US",
                   cls="real", patient_id=gid, group_id=gid,
                   view=views.get(f"openi:{p.stem}", "unknown"), extra=r)


def scan_synthchex(root: Path):
    d = root / "raw" / "synthchex"
    labels = {}
    with open(d / "labels.csv") as f:
        for r in csv.DictReader(f):
            labels[r["image_filename"]] = r
    for p in sorted((d / "png").glob("*.png")):
        src = p.stem.removeprefix("SyntheticImg_")  # MIMIC dicom id of the source report
        yield _row(root, p, {}, image_id=f"synthchex:{p.stem}", source="synthchex",
                   site="synthetic", cls="fake", generator="G1", group_id=f"mimic:{src}",
                   view="unknown", extra={"shard": labels[p.name]["shard"],
                                     "labels": labels[p.name]["label"]})


def scan_node21(root: Path):
    """NODE21 originals (.mha, 1024 px, 8-bit). Nodule images (n*) map to their source
    dataset; those from NIH / Open-I inherit the source's group_id so a NODE21 copy always
    shares the split of its original. Controls (c*) have no published mapping: they are
    de-duplicated against our sets by image content (scripts/dedup_node21.py)."""
    d = root / "raw" / "node21" / "cxr_images" / "original_data"
    src = {}
    with open(d / "filenames_orig_and_new.csv") as f:
        for r in csv.DictReader(f):
            src[r["node21_img_id"]] = (r["orig_dataset"], r["original_image_name"])
    boxes: dict[str, list] = {}
    with open(d / "metadata.csv") as f:
        for r in csv.DictReader(f):
            if r["label"] == "1":
                boxes.setdefault(Path(r["img_name"]).stem, []).append(
                    [int(r["x"]), int(r["y"]), int(r["width"]), int(r["height"])])
    nih_pid = {}
    with open(root / "raw/nih/data/Data_Entry_2017_v2020.csv") as f:
        for r in csv.DictReader(f):
            nih_pid[Path(r["Image Index"]).stem] = f"nih:{int(r['Patient ID'])}"
    openi_rep = openi_reports(root / "raw" / "openi")
    for p in sorted((d / "images").glob("*.mha")):
        nid = p.stem
        ds, orig = src.get(nid, ("unknown", ""))
        if ds == "chestxray14" and orig in nih_pid:
            gid = nih_pid[orig]
        elif ds == "openi" and f"CXR{orig}" in openi_rep:
            gid = f"openi:{openi_rep[f'CXR{orig}']['report_id']}"
        else:
            gid = f"node21:{nid}"
        r = {c: "" for c in COLUMNS}
        r.update(image_id=f"node21:{nid}", source="node21", site=f"NODE21-{ds}", cls="real",
                 patient_id=gid, group_id=gid, view="PA", raw_path=str(p.relative_to(root)),
                 raw_sha256=_sha(p), width=1024, height=1024, mode="L",
                 extra=json.dumps({"orig_dataset": ds, "orig_name": orig,
                                   "nodule_boxes_xywh": boxes.get(nid, [])}))
        yield r


def scan_generated(root: Path, gen: str, source: str):
    """Locally generated fakes (data/raw/gen/<GEN>/). group_id = the source report's group from the
    prompt table, so a fake always shares its split with the report that seeded it."""
    d = root / "raw" / "gen" / gen
    log = {r["prompt_id"]: r for r in csv.DictReader(open(d / "gen_log.csv"))} if (d / "gen_log.csv").exists() else {}
    for p in sorted(d.glob("p*.png")):
        r = log.get(p.stem, {})
        yield _row(root, p, {}, image_id=f"{source}:{p.stem}", source=source, site="synthetic", cls="fake",
                   generator=gen, group_id=r.get("group_id", f"{source}:{p.stem}"), view="unknown",
                   extra={"prompt_id": p.stem, "seed": r.get("seed", ""), "params": r.get("params", "")})


def scan_chexpert(root: Path):
    """CheXpert Plus subset (Stanford, US; Redivis aimi.chexpert_plus:5yyj:v1_0). Full-resolution 8-bit PNGs.
    Patient = deid_patient_id. Rows whose ap_pa is not PA/AP (e.g. 'LL') get view='lateral' so they are excluded.
    License: no redistribution; never sent to third-party APIs."""
    d = root / "raw" / "chexpert_plus"
    man = _manifest(d)
    with open(d / "selection.csv", newline="") as f:
        for r in csv.DictReader(f):
            p = d / "png" / r["file_name"]
            if not p.exists():
                continue
            pid = f"chexpert:{r['deid_patient_id']}"
            view = r["ap_pa"] if r["ap_pa"] in ("PA", "AP") else "lateral"
            yield _row(root, p, man, image_id="chexpert:" + r["file_name"][:-4].replace("/", "_"),
                       source="chexpert", site="CheXpert-US", cls="real", patient_id=pid, group_id=pid,
                       view=view, extra={"age": r.get("age", ""), "sex": r.get("sex", ""),
                                         "findings": (r.get("section_findings") or "")[:2000],
                                         "impression": (r.get("section_impression") or "")[:2000]})


def _scan_api(root: Path, gen: str, source: str):
    """Commercial-API fakes. text: group = source report (prompts.csv); image-conditioned (cond_v2):
    group = the SOURCE real image's patient group, so an edit shares the split of its source image.
    Only cond_v2 is used (cond/ holds 3 smoke tests made from preprocessed sources: excluded)."""
    d = root / "raw" / "api" / gen
    prompts = {r["prompt_id"]: r for r in csv.DictReader(open(root / "meta" / "prompts.csv"))}
    for p in sorted((d / "text").glob("p*.*")):
        pr = prompts.get(p.stem, {})
        yield _row(root, p, {}, image_id=f"{source}:text_{p.stem}", source=source, site="synthetic", cls="fake",
                   generator=gen, group_id=pr.get("group_id", f"{source}:{p.stem}"), view="unknown",
                   extra={"arm": "text", "prompt_id": p.stem})
    man = {r["id"]: r for r in csv.DictReader(open(root / "raw" / "api" / "manifest_cond.csv"))}
    for p in sorted((d / "cond_v2").glob("*.*")):
        m = man.get(p.stem, {})
        yield _row(root, p, {}, image_id=f"{source}:cond_{p.stem}", source=source, site="synthetic", cls="fake",
                   generator=gen, group_id=m.get("group_id", f"{source}:{p.stem}"), view="unknown",
                   extra={"arm": "image_conditioned", "source_site": m.get("site", ""),
                          "source_image": p.stem, "prompt_id": m.get("prompt_id", "")})


def scan_gpt_image(root: Path):
    yield from _scan_api(root, "G4", "gpt_image")


def scan_gemini_image(root: Path):
    yield from _scan_api(root, "G5", "gemini_image")


def scan_monai_ldm(root: Path):
    yield from scan_generated(root, "G2", "monai_ldm")


SCANNERS = {"nih": scan_nih, "montgomery": scan_montgomery, "shenzhen": scan_shenzhen,
            "openi": scan_openi, "synthchex": scan_synthchex, "node21": scan_node21,
            "monai_ldm": scan_monai_ldm, "chexpert": scan_chexpert,
            "gpt_image": scan_gpt_image, "gemini_image": scan_gemini_image}


def write_inventory(rows, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
            n += 1
    return n
