"""The single preprocessing function used for EVERY image, real or fake (Pipeline step 1).

raw file -> float array -> fixed percentile window -> 8-bit range -> border crop
-> centre square crop -> resize (256 primary / 512 secondary) -> lossless PNG.

Only pixels are written out, so embedded metadata (EXIF, C2PA, DICOM tags, PNG text
chunks) never reaches a detector.

Design decisions (revisit after the shortcut audit, Phase 3):
* Window: 0.5th-99.5th percentile of the image itself, so the windowing does not depend
  on the source's bit depth.
* Border crop (v2, 2026-11-08): drop edge rows/columns that are near-uniform OR >90%
  saturated dark/bright (collimation bands, scanner borders, padding, black bars on API
  outputs), at most 25% per side per pass, two passes. v1 (std only, 10%) left the
  collimation bands seen in Montgomery.
* Square: centre crop of the short side. No padding, because padding bars would differ
  between sources and become a shortcut. No stretching, because aspect ratio also differs
  by source.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

PCT_LO, PCT_HI = 0.5, 99.5
BORDER_STD = 0.02  # rows/cols with std below this (on 0-1 scale) count as border
DARK_LEVEL, BRIGHT_LEVEL = 0.05, 0.95  # saturated-dark / saturated-bright pixel levels
SAT_FRAC = 0.90  # a row/col that is >90% saturated dark or bright counts as border
MAX_CROP_FRAC = 0.25  # per side, per pass (v1 was 0.10, which left collimation bands)

# JSRT: 2048 x 2048, 12-bit stored in 16-bit big-endian, no header.
JSRT_SHAPE = (2048, 2048)


def load_any(src: str | Path | bytes, kind: str | None = None) -> np.ndarray:
    """Return a 2-D float32 array. `kind` overrides extension detection
    ('dicom', 'jsrt', 'mha', or None for anything Pillow reads)."""
    if isinstance(src, (bytes, bytearray)):
        return _to_gray(Image.open(io.BytesIO(src)))
    p = Path(src)
    ext = p.suffix.lower()
    kind = kind or {".dcm": "dicom", ".img": "jsrt", ".mha": "mha", ".mhd": "mha"}.get(ext)
    if kind == "dicom":
        import pydicom

        ds = pydicom.dcmread(str(p))
        a = ds.pixel_array.astype(np.float32)
        a = a * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            a = a.max() - a
        return a
    if kind == "jsrt":
        a = np.fromfile(p, dtype=">u2").reshape(JSRT_SHAPE).astype(np.float32)
        return a.max() - a  # JSRT stores high values as dark; invert to the usual look
    if kind == "mha":
        import SimpleITK as sitk

        a = sitk.GetArrayFromImage(sitk.ReadImage(str(p))).astype(np.float32)
        return np.squeeze(a)
    return _to_gray(Image.open(p))


def _to_gray(im: Image.Image) -> np.ndarray:
    if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        return np.asarray(im, dtype=np.float32)
    return np.asarray(im.convert("L"), dtype=np.float32)


def window(a: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(a, [PCT_LO, PCT_HI])
    if hi <= lo:
        return np.zeros_like(a, dtype=np.float32)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _is_border(lines: np.ndarray) -> np.ndarray:
    """lines: (n, L) rows (or transposed columns) on the 0-1 scale. A line is border if it is
    near-uniform, or mostly saturated dark (collimation bands) or bright (padding)."""
    return ((lines.std(axis=1) < BORDER_STD)
            | ((lines < DARK_LEVEL).mean(axis=1) > SAT_FRAC)
            | ((lines > BRIGHT_LEVEL).mean(axis=1) > SAT_FRAC))


def crop_border(a: np.ndarray) -> np.ndarray:
    """Strip border rows/columns from each edge inward, at most MAX_CROP_FRAC per side.
    Two passes, because removing rows changes column statistics and vice versa."""
    for _ in range(2):
        h, w = a.shape

        def span(flag: np.ndarray, n: int) -> tuple[int, int]:
            lim = int(n * MAX_CROP_FRAC)
            s = 0
            while s < lim and flag[s]:
                s += 1
            e = n
            while n - e < lim and flag[e - 1]:
                e -= 1
            return s, e

        r0, r1 = span(_is_border(a), h)
        c0, c1 = span(_is_border(a.T), w)
        a = a[r0:r1, c0:c1]
    return a


def center_square(a: np.ndarray) -> np.ndarray:
    h, w = a.shape
    s = min(h, w)
    r, c = (h - s) // 2, (w - s) // 2
    return a[r : r + s, c : c + s]


def _resize(a: np.ndarray, size: int) -> np.ndarray:
    im = Image.fromarray(a, mode="F").resize((size, size), Image.Resampling.LANCZOS)
    return np.clip(np.rint(np.asarray(im) * 255.0), 0, 255).astype(np.uint8)


def preprocess(src, size: int = 256, kind: str | None = None) -> np.ndarray:
    """Full pipeline; returns uint8 array of shape (size, size)."""
    return _resize(center_square(crop_border(window(load_any(src, kind)))), size)


def preprocess_multi(src, sizes=(256, 512), kind: str | None = None):
    """Same pipeline, several output sizes from one decode, plus QC stats:
    dyn_range = (p99.5 - p0.5) / nominal full scale (informational only: unreliable for
                16-bit files whose max is an outlier pixel),
    out_std = grey-level std of the first output size (blank images -> ~0; used for QC),
    crop_h, crop_w = side lengths after border crop (before the centre square),
    raw_h, raw_w."""
    raw = load_any(src, kind)
    full = 255.0 if raw.max() <= 255 else float(raw.max())
    lo, hi = np.percentile(raw, [PCT_LO, PCT_HI])
    c = crop_border(window(raw))
    sq = center_square(c)
    out = {s: _resize(sq, s) for s in sizes}
    qc = {"dyn_range": float((hi - lo) / full) if full > 0 else 0.0,
          "out_std": float(out[sizes[0]].std()),
          "raw_h": raw.shape[0], "raw_w": raw.shape[1], "crop_h": c.shape[0], "crop_w": c.shape[1]}
    return out, qc


def save_png(arr: np.ndarray, dst: str | Path) -> None:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(dst, format="PNG", optimize=True)  # no metadata
