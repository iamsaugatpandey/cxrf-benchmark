"""Packaging audit (the Phase 3 gate, Study Plan v3 §6).

Every analysis image, real or fake, must be indistinguishable at the packaging level:
same pixel size, 8-bit single channel, PNG container, and no metadata chunks. This test
reads only headers and chunk info, so it covers the WHOLE processed set in about a minute.
Skipped automatically when data/processed does not exist (e.g. on a fresh clone).
"""
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / "data" / "processed"


def _files(size):
    d = ROOT / str(size)
    return sorted(d.rglob("*.png")) if d.exists() else []


@pytest.mark.parametrize("size", [256, 512])
def test_every_processed_image_has_identical_packaging(size):
    files = _files(size)
    if not files:
        pytest.skip(f"no processed {size}px images")
    bad = []
    for f in files:
        with Image.open(f) as im:
            problems = []
            if im.format != "PNG":
                problems.append(f"format={im.format}")
            if im.size != (size, size):
                problems.append(f"size={im.size}")
            if im.mode != "L":
                problems.append(f"mode={im.mode}")
            meta = {k: v for k, v in im.info.items() if k not in ("dpi",)}
            if meta:
                problems.append(f"metadata={sorted(meta)}")
            if problems:
                bad.append((str(f.relative_to(ROOT)), problems))
    assert not bad, f"{len(bad)} of {len(files)} images fail packaging: {bad[:5]}"
