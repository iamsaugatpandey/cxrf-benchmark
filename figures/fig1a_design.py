"""Fig. 1a (v2) — study design as a three-step visual narrative, hand-laid-out vector SVG.

Design language borrowed from the Kay / Yang figures (Style Guide §6):
  * the artifacts themselves are the figure (radiograph thumbnails, shown large);
  * soft light-grey cards without outlines (Yang et al. 2023, Fig. 1);
  * numbered grey step markers + white callouts with a soft shadow (Yang, Hedayati & Kay 2023, Fig. 1);
  * a quantile dotplot (1 dot = 1% of images) for the operating point (Kay's frequency framing);
  * semantic color carried into the text: blue = seen, red = unseen, gold = highlight;
  * no legends, no badges, thin rules, generous whitespace; Helvetica at Nature sizes.
v1 (boxed columns) is kept as fig1a_design_v1.py.
Output: figures/out/fig1a_design.{svg,pdf,png} (183 mm wide). Usage: python figures/fig1a_design.py
"""
import base64
import csv
import io
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import norm

from cxrf.preprocess import preprocess

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
OUT = ROOT / "figures/out"
OUT.mkdir(parents=True, exist_ok=True)

C = dict(seen="#2B5C8A", unseen="#D1495B", enc="#1B7F79", pix="#C98B2B", llm="#5B3F8C", gold="#E3B23C",
         ink="#222222", ink2="#6B6B6B", ink3="#9A9A9A", rule="#D9D9D9", card="#F3F3F1", step="#8C8C8C",
         fake="#7A7A7A")
FONT = "Helvetica, Arial, sans-serif"
W, H = 1830, 905  # units of 0.1 mm -> 183 x 90.5 mm
PT = 3.528  # 1 pt = 0.3528 mm = 3.528 units


def fs(pt):
    return round(pt * PT, 2)


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def T(x, y, s, pt=6.5, w="normal", col=None, anchor="start", italic=False, raw=False):
    st = ' font-style="italic"' if italic else ""
    body = s if raw else esc(s)
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{fs(pt)}" font-weight="{w}" '
            f'fill="{col or C["ink"]}" text-anchor="{anchor}" xml:space="preserve"{st}>{body}</text>')


def span(s, col=None, w=None):
    a = (f' fill="{col}"' if col else "") + (f' font-weight="{w}"' if w else "")
    return f"<tspan{a}>{esc(s)}</tspan>"


def R(x, y, w, h, fill, r=18, stroke="none", sw=0, extra=""):
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}" {extra}/>')


def L(x1, y1, x2, y2, col=None, sw=2.2, dash=None, arrow=False):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = ' marker-end="url(#ah)"' if arrow else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{col or C["ink3"]}" '
            f'stroke-width="{sw}" stroke-linecap="round"{d}{m}/>')


def thumb_uri(path, px=300):
    a = preprocess(path, size=256)
    im = Image.fromarray(a).resize((px, px), Image.Resampling.LANCZOS)
    b = io.BytesIO()
    im.save(b, format="PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def IMG(x, y, s, uri, frame=None, fw=3.0):
    out = f'<image x="{x:.1f}" y="{y:.1f}" width="{s}" height="{s}" href="{uri}" preserveAspectRatio="xMidYMid slice"/>'
    if frame:
        out += f'<rect x="{x:.1f}" y="{y:.1f}" width="{s}" height="{s}" fill="none" stroke="{frame}" stroke-width="{fw}"/>'
    return out


def step(x, y, n, title, sub_parts):
    s = f'<circle cx="{x + 17}" cy="{y}" r="17" fill="{C["step"]}"/>'
    s += T(x + 17, y + 7.5, str(n), 6.5, "bold", "#FFFFFF", "middle")
    s += T(x + 46, y + 9, title, 8, "bold")
    s += T(x + 46, y + 44, "".join(sub_parts), 6.2, col=C["ink2"], raw=True)
    return s


def callout(x, y, w, h):
    return R(x, y, w, h, "#FFFFFF", r=14, extra='filter="url(#sh)"')


def pick(source, split="train", view=None, k=1, offset=0):
    rows = [r for r in csv.DictReader(open(D / "meta/images.csv"))
            if r["source"] == source and not r["exclude"] and r["split"] == split
            and (view is None or r["view"] == view)]
    return [D / r["raw_path"] for r in rows[offset:offset + k]]


def site_pick(site):
    for r in csv.DictReader(open(D / "meta/images.csv")):
        if r["site"] == site and not r["exclude"] and r["split"] == "train":
            return D / r["raw_path"]


def main():
    nih = [thumb_uri(p) for p in pick("nih", view="PA", k=3, offset=40)]
    sx = thumb_uri(pick("synthchex", offset=5)[0])
    g2 = thumb_uri(D / "raw/gen/G2/p00004.png")
    oi = thumb_uri(pick("openi", offset=12)[0])
    sz = thumb_uri(pick("shenzhen", offset=3)[0])
    js = thumb_uri(site_pick("NODE21-jsrt"))
    pc = thumb_uri(site_pick("NODE21-padchest"))
    g4 = thumb_uri(D / "raw/api/G4/text/p00001.png")
    g5 = thumb_uri(D / "raw/api/G5/text/p00001.jpg")
    src = thumb_uri(D / "raw/nih/images/images_002/images/00003221_011.png")
    edit = thumb_uri(D / "raw/api/G4/cond_v2/nih_00003221_011.png")

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="183mm" height="{H / 10}mm" viewBox="0 0 {W} {H}">',
         "<defs>",
         f'<marker id="ah" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6" markerHeight="6" orient="auto">'
         f'<path d="M0,1 L9,5 L0,9 z" fill="{C["ink3"]}"/></marker>',
         '<filter id="sh" x="-10%" y="-10%" width="120%" height="130%"><feGaussianBlur in="SourceAlpha" stdDeviation="5"/>'
         '<feOffset dy="3" result="b"/><feComponentTransfer><feFuncA type="linear" slope="0.18"/></feComponentTransfer>'
         '<feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge></filter>',
         "</defs>", R(0, 0, W, H, "#FFFFFF", r=0)]
    # single-panel figure: no panel tag
    #    s.append(T(0, 30, "a", 8, "bold"))
    Y0, CH = 56, 840

    # ---------------- step 1: train ----------------
    x1, w1 = 30, 560
    s.append(R(x1, Y0, w1, CH, C["card"]))
    s.append(step(x1 + 26, Y0 + 52, 1, "Train on seen sources",
                  [span("Seen hospital", C["seen"], "bold"), span(" vs "), span("open generators", C["seen"], "bold")]))
    ty, ts = Y0 + 140, 150
    for i, u in enumerate(nih):  # fanned stack (Yang et al. 2023, Fig. 1D)
        s.append(IMG(x1 + 40 + i * 30, ty + (2 - i) * 22, ts, u, C["seen"], 2.6))
    s.append(T(x1 + 40, ty + 245, "real", 6.5, "bold"))
    s.append(T(x1 + 40, ty + 272, "NIH ChestX-ray14 (US)", 6, col=C["ink2"]))
    s.append(T(x1 + 40, ty + 296, "PA + AP views", 6, col=C["ink2"]))
    fx = x1 + 300
    s.append(IMG(fx, ty + 22, ts, sx, C["seen"], 2.6))
    s.append(IMG(fx + 70, ty + 44, ts, g2, C["seen"], 2.6))
    s.append(T(fx, ty + 245, "fake", 6.5, "bold"))
    s.append(T(fx, ty + 272, "SynthCheX · MONAI LDM", 6, col=C["ink2"]))
    s.append(T(fx, ty + 296, "both MIMIC-trained", 6, col=C["ink2"]))
    s.append(L(x1 + w1 / 2, ty + 318, x1 + w1 / 2, ty + 365, arrow=True))
    cy, cw, chh = ty + 370, 490, 330
    s.append(callout(x1 + 35, cy, cw, chh))
    s.append(T(x1 + 60, cy + 44, "Detectors trained on these images", 7, "bold"))
    fam = [(C["enc"], "Frozen-encoder probes", "DINOv3 · RAD-DINO · BiomedCLIP (± LBR)"),
           (C["pix"], "Pixel baselines", "radial FFT · format features"),
           (C["pix"], "Fine-tuned CNN", "ResNet-18")]
    for i, (col, a, b) in enumerate(fam):
        yy = cy + 92 + i * 56
        s.append(f'<circle cx="{x1 + 68}" cy="{yy - 8}" r="9" fill="{col}"/>')
        s.append(T(x1 + 88, yy, a, 6.5, "bold", col))
        s.append(T(x1 + 88, yy + 24, b, 5.7, col=C["ink2"]))
    s.append(L(x1 + 60, cy + 250, x1 + cw + 10, cy + 250, C["rule"], 1.6))
    s.append(f'<circle cx="{x1 + 68}" cy="{cy + 276}" r="9" fill="{C["llm"]}"/>')
    s.append(T(x1 + 88, cy + 284, "Frontier LLMs", 6.5, "bold", C["llm"]))
    s.append(T(x1 + 88, cy + 308, "no training · zero-shot on the same test images", 5.7, col=C["ink2"]))

    # ---------------- step 2: operating point (quantile dotplot) ----------------
    x2, w2 = 610, 420
    s.append(R(x2, Y0, w2, CH, C["card"]))
    s.append(step(x2 + 26, Y0 + 52, 2, "Fix one operating point",
                  [span("Threshold frozen for "), span("every", None, "bold"), span(" test set")]))
    ax0, ax1, base_f, base_r = x2 + 40, x2 + w2 - 30, Y0 + 400, Y0 + 640
    d = 13.2  # dot diameter (units)

    def xs(z):  # score -> x
        return ax0 + (z + 3.0) / 7.2 * (ax1 - ax0)

    thr = norm.ppf(0.99)

    def dotplot(mu, base, colf):
        q = norm.ppf((np.arange(100) + 0.5) / 100, loc=mu)
        out, counts = [], {}
        for v in q:
            b = int(round((xs(v) - ax0) / d))
            k = counts.get(b, 0)
            counts[b] = k + 1
            out.append(f'<circle cx="{ax0 + b * d:.1f}" cy="{base - d / 2 - k * d:.1f}" r="{d / 2 - 1.1:.1f}" fill="{colf(v)}"/>')
        return out

    s += dotplot(3.0, base_f, lambda v: C["fake"] if v > thr else "#C9C9C9")
    s += dotplot(0.0, base_r, lambda v: C["gold"] if v > thr else C["seen"])
    s.append(L(ax0 - 6, base_f + 4, ax1, base_f + 4, C["rule"], 1.6))
    s.append(L(ax0 - 6, base_r + 4, ax1, base_r + 4, C["rule"], 1.6))
    tx = xs(thr) + d * 0.1
    s.append(L(tx, Y0 + 230, tx, base_r + 40, C["ink"], 2.2, dash="7 6"))
    s.append(T(tx + 10, Y0 + 228, "threshold", 6, "bold"))
    s.append(T(ax0, base_f - 150, "fakes · open generators", 6.2, "bold", C["fake"]))
    s.append(T(ax0, base_r - 168, "real · seen hospital", 6.2, "bold", C["seen"]))
    s.append(T(ax0, base_r + 34, "score  →  more fake", 5.6, col=C["ink3"]))
    ccx, ccy = x2 + 34, Y0 + 700
    s.append(callout(ccx, ccy, w2 - 68, 128))
    s.append(f'<circle cx="{ccx + 26}" cy="{ccy + 36}" r="8" fill="{C["gold"]}"/>')
    s.append(T(ccx + 44, ccy + 43, span("1 of 100", C["ink"], "bold") + span(" real images from", C["ink2"]), 6, raw=True))
    s.append(T(ccx + 44, ccy + 70, "the seen hospital is flagged:", 6, col=C["ink2"]))
    s.append(T(ccx + 44, ccy + 97, "a 1% false-positive rate", 6, col=C["ink2"]))
    s.append(T(ax0, base_f + 32, "1 dot = 1% of images", 5.6, col=C["ink3"]))

    # ---------------- step 3: test under shift ----------------
    x3, w3 = 1050, 750
    s.append(R(x3, Y0, w3, CH, C["card"]))
    s.append(step(x3 + 26, Y0 + 52, 3, "Test under shift",
                  [span("Same threshold on "), span("unseen", C["unseen"], "bold"), span(" hospitals and generators")]))
    ts3 = 112
    rx = x3 + 36
    oc_x = x3 + w3 - 210

    def outcome(y, text_parts):
        return T(rx, y, "".join(text_parts), 5.8, col=C["ink2"], raw=True)

    r1 = Y0 + 132
    s.append(T(rx, r1, span("Real", None, "bold") + span(" · ") + span("unseen hospitals", C["unseen"], "bold"), 6.5, raw=True))
    for i, (u, lab) in enumerate([(oi, "Open-I (US)"), (sz, "Shenzhen (CN)"), (js, "JSRT (JP)"), (pc, "PadChest (ES)")]):
        xx = rx + i * (ts3 + 24)
        s.append(IMG(xx, r1 + 40, ts3, u, C["unseen"], 2.4))
        s.append(T(xx, r1 + ts3 + 66, lab, 5.2, col=C["ink2"]))
    s.append(T(rx + 4 * (ts3 + 24) + 2, r1 + 92, "+ CheXpert", 5.6, col=C["ink2"]))
    s.append(T(rx + 4 * (ts3 + 24) + 2, r1 + 114, "(US), not", 5.2, col=C["ink3"], italic=True))
    s.append(T(rx + 4 * (ts3 + 24) + 2, r1 + 134, "shown (license)", 5.2, col=C["ink3"], italic=True))
    s.append(outcome(r1 + 26, [span("flagged", C["unseen"], "bold"), span(" = false positive → "), span("hospital shift", C["unseen"], "bold"), span(" (primary endpoint)")]))

    r2 = Y0 + 362
    s.append(L(rx, r2 - 34, x3 + w3 - 36, r2 - 34, C["rule"], 1.6))
    s.append(T(rx, r2, span("Fake", None, "bold") + span(" · ") + span("unseen commercial generators", C["unseen"], "bold"), 6.5, raw=True))
    for i, (u, lab) in enumerate([(g4, "gpt-image-2"), (g5, "Gemini image")]):
        xx = rx + i * (ts3 + 18)
        s.append(IMG(xx, r2 + 40, ts3, u, C["unseen"], 2.4))
        s.append(T(xx, r2 + ts3 + 66, lab, 5.4, col=C["ink2"]))
    s.append(T(rx + 2 * (ts3 + 18) + 4, r2 + 100, "prompted with the same report", 5.6, col=C["ink2"]))
    s.append(T(rx + 2 * (ts3 + 18) + 4, r2 + 122, "text as the open-generator fakes", 5.6, col=C["ink2"]))
    s.append(outcome(r2 + 26, [span("missed", C["unseen"], "bold"), span(" = false negative → "), span("generator shift", C["unseen"], "bold")]))

    r3 = Y0 + 592
    s.append(L(rx, r3 - 34, x3 + w3 - 36, r3 - 34, C["rule"], 1.6))
    s.append(T(rx, r3, span("Fake", None, "bold") + span(" · ") + span("image-conditioned", C["unseen"], "bold")
               + span(" (edits a hospital's own radiograph)"), 6.5, raw=True))
    s.append(outcome(r3 + 26, [span("missed", C["unseen"], "bold"), span(" = false negative → tests the "), span("appearance shortcut", C["unseen"], "bold")]))
    s.append(IMG(rx, r3 + 40, ts3, src, C["seen"], 2.4))
    s.append(T(rx, r3 + ts3 + 66, "real source", 5.4, col=C["ink2"]))
    ex = rx + ts3 + 110
    s.append(L(rx + ts3 + 12, r3 + 40 + ts3 / 2, ex - 12, r3 + 40 + ts3 / 2, arrow=True))
    s.append(T(rx + ts3 + 55, r3 + 40 + ts3 / 2 - 12, "edit", 5.6, "bold", C["ink2"], "middle"))
    s.append(IMG(ex, r3 + 40, ts3, edit, C["unseen"], 2.4))
    mx, my = ex + ts3 * 85 / 256, r3 + 40 + ts3 * 72 / 256
    s.append(f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="15" fill="none" stroke="{C["gold"]}" stroke-width="2.6" stroke-dasharray="5 3.5"/>')
    s.append(T(ex, r3 + ts3 + 66, "gpt-image-2 edit", 5.4, col=C["ink2"]))
    nx = ex + ts3 + 22
    s.append(L(mx + 16, my, nx - 4, my, C["gold"], 1.8))
    s.append(T(nx, my + 6, span("added:", C["ink"], "bold") + span(" 3 cm right-upper-lobe mass", C["ink2"]), 5.6, raw=True))
    s.append(T(nx, my + 40, span("kept:", C["ink"], "bold") + span(" markers, exposure, anatomy", C["ink2"]), 5.6, raw=True))
    s.append(T(nx, my + 62, "so the fake inherits the hospital's look", 5.6, col=C["ink2"]))

    s.append("</svg>")
    svg = OUT / "fig1a_design.svg"
    svg.write_text("\n".join(s))
    for fmt, extra in [("pdf", []), ("png", ["-d", "300", "-p", "300"])]:
        subprocess.run(["rsvg-convert", "-f", fmt, *extra, "-o", str(OUT / f"fig1a_design.{fmt}"), str(svg)], check=True)
    print("wrote", OUT / "fig1a_design.{svg,pdf,png}")


if __name__ == "__main__":
    main()
