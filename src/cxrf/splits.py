"""Group-level splits and the master metadata table (Pipeline step 3).

split(group_id) is a deterministic hash of the group id (patient for reals; source report or
study for fakes) with a fixed salt, so:
  * every image of a patient or prompt source lands in the same split,
  * adding new data never moves existing groups,
  * the assignment needs no stored state and is reproducible from the code alone.
Fractions: 60% train / 20% val / 20% test. Test-only sources (API generators G4/G5, sites
reserved for held-out evaluation) are handled by the experiment configs, not here: every
image gets a split, and experiments choose which splits of which sources to use.

The Phase 2 pilot draws ONLY from split == "train", so pilot images can never reach a test set.
"""
from __future__ import annotations

import hashlib

SALT = "cxrf-splits-v1"
FRACTIONS = (("train", 0.6), ("val", 0.2), ("test", 0.2))
BLANK_OUT_STD = 10.0  # final 256-px image with grey-level std below this is blank


def split_of(group_id: str, salt: str = SALT) -> str:
    h = int(hashlib.sha256(f"{salt}:{group_id}".encode()).hexdigest()[:12], 16) / 16**12
    acc = 0.0
    for name, frac in FRACTIONS:
        acc += frac
        if h < acc:
            return name
    return FRACTIONS[-1][0]


def exclusion_reason(inv_row: dict, qc_row: dict | None, seen_sha: set) -> str:
    """'' if the image is usable, else a short reason. Mutates seen_sha (first copy wins)."""
    if inv_row["view"] == "lateral":
        return "lateral"
    if qc_row is None:
        return "not_preprocessed"
    if qc_row.get("error"):
        return "preprocess_error"
    if float(qc_row["out_std"]) < BLANK_OUT_STD:
        return "blank"
    if inv_row["raw_sha256"] in seen_sha:
        return "exact_duplicate"
    seen_sha.add(inv_row["raw_sha256"])
    return ""
