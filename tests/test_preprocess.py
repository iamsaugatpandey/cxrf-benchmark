import sys
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cxrf.preprocess import preprocess, save_png  # noqa: E402


def _fake_cxr(h, w, dtype=np.uint8, border=0):
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:h, 0:w]
    a = 0.5 + 0.3 * np.sin(xx / 37) * np.cos(yy / 53) + 0.05 * rng.standard_normal((h, w))
    a = np.clip(a, 0, 1)
    if border:
        a[:border] = 0
        a[-border:] = 0
    scale = 255 if dtype == np.uint8 else 4095
    return (a * scale).astype(dtype)


def test_shapes_and_dtype(tmp_path):
    for size in (256, 512):
        p = tmp_path / "a.png"
        Image.fromarray(_fake_cxr(1200, 1000)).save(p)
        out = preprocess(p, size=size)
        assert out.shape == (size, size) and out.dtype == np.uint8
        # window spans the range (downscaling averages away the most extreme pixels)
        assert out.min() <= 20 and out.max() >= 235


def test_16bit_and_8bit_same_content_match(tmp_path):
    a8 = _fake_cxr(800, 800)
    a16 = (a8.astype(np.uint16) * 16)
    p8, p16 = tmp_path / "a8.png", tmp_path / "a16.png"
    Image.fromarray(a8).save(p8)
    Image.fromarray(a16).save(p16)
    d = np.abs(preprocess(p8).astype(int) - preprocess(p16).astype(int))
    assert d.max() <= 1  # bit depth must not leak into the output


def test_border_is_cropped(tmp_path):
    p = tmp_path / "b.png"
    Image.fromarray(_fake_cxr(1000, 1000, border=60)).save(p)
    out = preprocess(p)
    assert out[:3].mean() > 20 and out[-3:].mean() > 20


def test_metadata_is_dropped(tmp_path):
    src, dst = tmp_path / "m.png", tmp_path / "out.png"
    info = PngImagePlugin.PngInfo()
    info.add_text("Software", "SomeGenerator v1")
    Image.fromarray(_fake_cxr(600, 600)).save(src, pnginfo=info)
    save_png(preprocess(src), dst)
    assert not Image.open(dst).text


def test_noisy_collimation_band_is_cropped(tmp_path):
    # a dark band with noise (std above the uniform-border rule) covering 20% of the bottom
    a = _fake_cxr(1000, 1000).astype(np.float32)
    rng = np.random.default_rng(1)
    a[800:] = np.clip(rng.normal(4, 3, (200, 1000)), 0, 255)
    p = tmp_path / "band.png"
    Image.fromarray(a.astype(np.uint8)).save(p)
    out = preprocess(p)
    assert out[-5:].mean() > 20  # band gone


def test_split_is_deterministic_and_balanced():
    from collections import Counter
    from cxrf.splits import split_of
    ids = [f"g{i}" for i in range(20000)]
    assert [split_of(i) for i in ids[:50]] == [split_of(i) for i in ids[:50]]
    c = Counter(split_of(i) for i in ids)
    assert abs(c["train"] / 20000 - 0.6) < 0.02 and abs(c["test"] / 20000 - 0.2) < 0.02
