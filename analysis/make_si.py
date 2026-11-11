"""Build the Supplementary Information (manuscript/supplementary.tex) and its figures from saved results.

Everything numeric is read from results/{models,tables}; nothing is typed by hand except the deviations table
and the descriptions. Re-run after any refit:  .venv/bin/python analysis/make_si.py
Figures: manuscript/figures/si_montage.png (test images per source; CheXpert excluded by licence),
         manuscript/figures/si_ppc_h1.pdf (posterior predictive check of the primary H1 model),
         manuscript/figures/si_prior_predictive.pdf (copied from admin/).
"""
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "results/models"
T = ROOT / "results/tables"
OUT = ROOT / "manuscript"
FIG = OUT / "figures"

FAM = {"mlp": "Encoder probe (MLP)", "mlp_lbr": "MLP + latent blending", "resnet": "ResNet-18",
       "knn": "Encoder kNN", "format": "Format logistic", "ocsvm": "One-class SVM", "fft": "Spectrum logistic"}
FAM_ORDER = list(FAM)
AN = {"cells": "Primary", "S1": "S1 PA only", "S2": "S2 masked", "S3": "S3 JPEG", "S4": "S4 512 px",
      "S5": "S5 wide priors", "S6": "S6 pooled threshold", "S7": "S7 memorized removed", "S9": "S9 test prompts", "": "Primary"}
MODEL_LAB = {"claude-opus-5-5": "Claude Opus 5.5", "claude-haiku-5-5": "Claude Haiku 5.5", "gpt-6-astra": "GPT-6 Astra",
             "gpt-5.4-mini": "GPT-5.4 mini", "gemini-4-argon": "Gemini 4 Argon",
             "gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite"}


def tex(s):
    return str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


def ci(m, lo, hi, d=1):
    return f"${m:+.{d}f}$ [{lo:.{d}f}, {hi:.{d}f}]".replace("$+-", "$-")


def table(caption, label, cols, rows, align=None, size=r"\footnotesize", fit=False):
    align = align or "l" + "r" * (len(cols) - 1)
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    return (f"\\begin{{table}}[!htbp]\n\\centering\n{size}\n\\caption{{{caption}}}\\label{{{label}}}\n"
            + (r"\resizebox{\textwidth}{!}{" if fit else "")
            + f"\\begin{{tabular}}{{{align}}}\n\\toprule\n{' & '.join(cols)} \\\\\n\\midrule\n{body}\n\\bottomrule\n"
            f"\\end{{tabular}}" + ("}" if fit else "") + "\n\\end{table}\n")


# ---------------------------------------------------------------- diagnostics
def diag_rows():
    rows = []
    for s in sorted(M.glob("*_summary.txt")):
        tag = s.name.replace("_summary.txt", "")
        if tag.startswith("h1_perimage"):
            continue
        txt = s.read_text()
        m = re.search(r"max Rhat ([\d.]+) \| min bulk ESS (\d+) \| min tail ESS (\d+) \| divergences (\d+)", txt)
        if not m:
            continue
        ppc = ""
        p = M / f"{tag}_ppc.csv"
        if p.exists():
            d = pd.read_csv(p)
            ppc = f"{int(d.inside.sum())}/{len(d)}"
        ok = "yes" if "criteria met: TRUE" in txt else "no"
        rows.append([tex(tag), m.group(1), m.group(2), m.group(3), m.group(4), ok, ppc])
    return rows


# ---------------------------------------------------------------- contrasts
def contrast_table(prefix, tags, caption, label, what):
    rows = []
    for fam in FAM_ORDER:
        r = [FAM[fam]]
        for t in tags:
            f = M / f"{prefix}{t}_contrasts.csv"
            if not f.exists():
                r.append("--"); continue
            d = pd.read_csv(f).set_index("family")
            r.append(ci(*d.loc[fam, ["delta_pp_med", "delta_pp_lo95", "delta_pp_hi95"]]) if fam in d.index else "--")
        rows.append(r)
    cols = ["Family"] + [AN[t.lstrip("_")] if t else "Primary" for t in tags]
    return table(caption + f" Entries are posterior medians and 95\\% credible intervals of {what} (percentage points) "
                 "for a typical hospital or generator and fitted detector. Dashes: family not fitted in that analysis.",
                 label, cols, rows, align="l" + "c" * len(tags), size=r"\scriptsize", fit=True)


# ---------------------------------------------------------------- figures
def montage():
    m = pd.read_csv(ROOT / "data/meta/images.csv", keep_default_na=False, dtype=str)
    m = m[(m.exclude == "") & (m.split == "test")]
    arm = m.extra.map(lambda e: json.loads(e).get("arm", "") if e.startswith("{") else "")
    groups = [("NIH (US)", m.source == "nih"), ("Open-I (US)", m.source == "openi"),
              ("Shenzhen (CN)", m.source == "shenzhen"), ("Montgomery (US)", m.source == "montgomery"),
              ("JSRT (JP)", m.site == "NODE21-jsrt"), ("PadChest (ES)", m.site == "NODE21-padchest"),
              ("G1 SynthCheX", m.generator == "G1"), ("G2 MONAI", m.generator == "G2"),
              ("G4 gpt-image-2, text", (m.generator == "G4") & (arm == "text")),
              ("G4 gpt-image-2, edit", (m.generator == "G4") & (arm == "image_conditioned")),
              ("G5 Gemini, text", (m.generator == "G5") & (arm == "text")),
              ("G5 Gemini, edit", (m.generator == "G5") & (arm == "image_conditioned"))]
    k, s, lab = 5, 160, 210
    W, H = lab + k * (s + 4), len(groups) * (s + 4)
    canvas = Image.new("L", (W, H), 255)
    dr = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 17)
    except OSError:
        font = ImageFont.load_default()
    rng = np.random.default_rng(20261109)
    for i, (name, sel) in enumerate(groups):
        g = m[sel]
        pick = g.iloc[rng.choice(len(g), size=min(k, len(g)), replace=False)]
        dr.text((4, i * (s + 4) + s // 2 - 10), name, fill=30, font=font)
        for j, (_, r) in enumerate(pick.iterrows()):
            im = Image.open(ROOT / "data" / r.out256).convert("L").resize((s, s), Image.LANCZOS)
            canvas.paste(im, (lab + j * (s + 4), i * (s + 4)))
    canvas.save(FIG / "si_montage.png", dpi=(300, 300))


def ppc_fig():
    import subprocess
    r = r'''
source("figures/theme_cxrf.R")
p <- read.csv("results/models/h1_cells_ppc.csv") |>
  mutate(nov = ifelse(novelty_site == 1, "Hospital unseen", "Hospital seen"),
         fam = c(mlp="Encoder probe (MLP)", mlp_lbr="MLP + latent blending", resnet="ResNet-18", knn="Encoder kNN",
                 format="Format logistic", ocsvm="One-class SVM", fft="Spectrum logistic")[family])
g <- ggplot(p, aes(100 * obs, 100 * pred_med, colour = nov)) +
  geom_abline(colour = cxrf_pal$ref, linetype = "22", linewidth = 0.3) +
  geom_linerange(aes(ymin = 100 * pred_lo, ymax = 100 * pred_hi), linewidth = 0.3) +
  geom_point(aes(shape = inside), size = 1.2, fill = "white") +
  scale_shape_manual(NULL, values = c(`TRUE` = 16, `FALSE` = 21), labels = c(`TRUE` = "inside 95% interval", `FALSE` = "outside")) +
  scale_colour_manual(NULL, values = c("Hospital seen" = cxrf_pal$seen, "Hospital unseen" = cxrf_pal$unseen)) +
  scale_x_log10("Observed false-positive rate (%)") + scale_y_log10("Predicted (%; median and 95% interval)") +
  facet_wrap(~fam, ncol = 4) + theme_cxrf(grid_y = TRUE) + theme(legend.position = "bottom")
save_cxrf(g, "manuscript/figures/si_ppc_h1", width_mm = 183, height_mm = 105)
'''
    subprocess.run(["Rscript", "-e", r], cwd=ROOT, check=True, capture_output=True)


# ---------------------------------------------------------------- LLM tables
def llm_tables():
    met = pd.read_csv(T / "llm_metrics.csv")
    rt = pd.read_csv(T / "llm_retest.csv").set_index("model")
    rows = []
    for cond, cl in [("zeroshot", "Zero-shot"), ("fourshot", "Four-shot")]:
        for mdl in MODEL_LAB:
            r = met[(met.model == mdl) & (met.condition == cond)].iloc[0]
            rows.append([cl, MODEL_LAB[mdl], f"{int(r.n):,}", f"{int(r.refusal + r.unparseable + r.api_error)}",
                         f"{r.auc:.2f} [{r.auc_lo95:.2f}, {r.auc_hi95:.2f}]", f"{r.fpr50:.1f}", f"{r.tpr50:.1f}",
                         f"{r.ece:.3f}", f"{r.brier:.3f}", f"{r.cost_per_1000:.2f}"])
    t1 = table("LLM discrimination, calibration and cost. FPR and TPR at the preregistered threshold of 50 (\\%). "
               "ECE: expected calibration error, 10 equal-mass bins. Cost: US\\$ per 1,000 images at batch prices.",
               "tab:si_llm", ["Condition", "Model", "Images", "Invalid", "AUC [95\\% CI]", "FPR", "TPR", "ECE",
                              "Brier", "Cost"], rows, size=r"\scriptsize")
    rows = [[MODEL_LAB[m], f"{int(r.n_images)}", f"{r.kappa_mean:.2f}", f"{r.kappa_min:.2f}",
             f"{r.agreement_pct:.1f}", f"{r.within_image_sd:.1f}"] for m, r in rt.loc[list(MODEL_LAB)].iterrows()]
    t2 = table("Test--retest repeatability over three zero-shot runs. $\\kappa$: Cohen's kappa of flags at threshold 50 "
               "(mean and minimum over run pairs); agreement: \\% of images with identical flags in all runs; "
               "SD: mean within-image standard deviation of the stated probability (points).",
               "tab:si_retest", ["Model", "Images", "$\\kappa$ mean", "$\\kappa$ min", "Agreement (\\%)", "SD"], rows)
    return t1, t2


def main():
    FIG.mkdir(exist_ok=True)
    montage()
    ppc_fig()
    shutil.copy(ROOT / "admin/prior_predictive.pdf", FIG / "si_prior_predictive.pdf")
    sys_prompt = ("You are taking part in an image-forensics research study on the provenance of chest radiographs. "
                  "This is not a diagnostic task, and no clinical decision depends on your answer.")
    from importlib.util import module_from_spec, spec_from_file_location  # read prompts from the frozen code
    src = (ROOT / "scripts/run_llm.py").read_text()
    user = re.search(r'USER = \((.*?)\)\n', src, re.S).group(1)
    user = "".join(re.findall(r'"(.*?)"', user)).replace("\\n", "\n\n")
    gsrc = (ROOT / "scripts/gen_api.py").read_text()
    ttxt = "".join(re.findall(r'"(.*?)"', re.search(r"TEXT_TEMPLATE = \((.*?)\)\n", gsrc, re.S).group(1)))
    ctxt = "".join(re.findall(r'"(.*?)"', re.search(r"COND_TEMPLATE = \((.*?)\)\n", gsrc, re.S).group(1)))
    t_llm, t_rt = llm_tables()
    pi = pd.read_csv(M / "h1_perimage_full_contrasts.csv").set_index("family")
    prim = pd.read_csv(M / "h1_cells_contrasts.csv").set_index("family")
    bs = pd.read_csv(T / "h1_bootstrap.csv").set_index("family")
    sf = pd.read_csv(M / "h1_sitefam_contrasts.csv").set_index("family")
    rob = [[FAM[f], ci(*prim.loc[f, ["delta_pp_med", "delta_pp_lo95", "delta_pp_hi95"]]),
            ci(*pi.loc[f, ["delta_pp_med", "delta_pp_lo95", "delta_pp_hi95"]]),
            ci(*sf.loc[f, ["delta_pp_med", "delta_pp_lo95", "delta_pp_hi95"]]),
            ci(*bs.loc[f, ["delta_pp_est", "delta_pp_lo95", "delta_pp_hi95"]])] for f in FAM_ORDER if f in pi.index]

    dev = [
        ["H1 likelihood", "Bernoulli per image with (1 | patient)",
         "Beta-binomial on 5,265 cells (custom brms family); patient effect replaced by within-cell overdispersion",
         "Per-image model needed $>$30 h and did not converge on a 5\\% sample; built-in beta-binomial failed to initialize"],
        ["H2/H4 likelihood", "Bernoulli with (1 | source report/image)", "Beta-binomial on cells, as for H1", "As above"],
        ["H4 model", "One model: family $\\times$ conditioning $\\times$ same-site",
         "Two models: H4a (conditioning) and H4b (source hospital seen vs unseen, edits only)",
         "The three-way model is not identified: text-prompted fakes have no source hospital"],
        ["Overdispersion prior", "Not specified", "$\\phi\\sim$ Gamma(0.01, 0.01) (brms default)", "Parameter added by the beta-binomial"],
        ["Draws", "4 chains $\\times$ 2,000", "4 $\\times$ 3,000 (H4b 4 $\\times$ 8,000)", "To exceed the ESS criterion"],
        ["Fit sequence", "Minimal $\\to$ maximal", "Maximal fitted directly", "All maximal models converged; no simplification needed"],
        ["Patient-level robustness", "--", "Per-image binomial with (1 | patient), 234,315 rows",
         "Added; did not converge (max $\\hat R$ 1.19); reported as supporting evidence only"],
        ["GLMM cross-check", "lme4 GLMM", "Not run; cluster bootstrap reported",
         "lme4 required a Matrix version that could not be compiled on the analysis computer"],
        ["LLM run window", "10--24 Nov 2026", "All calls on 9 Nov 2026 (12:45--14:24 EST), after registration (02:13 EST)",
         "Runs started once the registration was submitted; one day before the stated window"],
        ["LLM subset sizes", "Retest 400, four-shot 500", "403 and 503", "Rounding in stratified sampling"],
        ["LLM secondary threshold", "Threshold calibrated on validation reals", "Not run",
         "LLMs scored only the frozen test subset; no validation scores exist"],
        ["LLM multilevel rates", "Same multilevel models as detectors", "Per-stratum Wilson intervals",
         "Too few LLM images per hospital for the multilevel models"],
        ["S2, S3, S4 for ResNet-18", "All detectors", "Frozen-encoder and pixel detectors only",
         "Retraining ResNet-18 on each image variant was beyond the compute budget"],
        ["Exploratory model", "--", "Family-specific hospital effects (labelled exploratory)",
         "Added after the H1 posterior predictive check fitted 49/91 cells"],
        ["Reason coding", "Two coders on 10\\% of LLM reasons", "Not done", "Single-author study; deferred"],
    ]
    t_dev = table("Deviations from the preregistration (OSF 5vwkj).", "tab:si_dev",
                  ["Item", "Preregistered", "Done", "Reason"],
                  [[a, b, c, d] for a, b, c, d in dev], align="p{2.2cm}p{3.2cm}p{4.3cm}p{4.3cm}", size=r"\scriptsize")
    t_diag = table("Convergence diagnostics for every fitted model, after refits. Criteria: $\\hat R<1.01$, bulk and "
                   "tail ESS $>1{,}000$, no divergent transitions. PPC: hospital $\\times$ family $\\times$ novelty cells "
                   "whose observed rate falls inside the 95\\% posterior predictive interval (H1 models).",
                   "tab:si_diag", ["Model", "max $\\hat R$", "min bulk ESS", "min tail ESS", "Divergences", "Met",
                                   "PPC"], diag_rows(), size=r"\scriptsize")
    t_h1 = contrast_table("h1_", ["cells", "S1", "S2", "S3", "S4", "S5", "S6"],
                          "H1 across the primary and sensitivity analyses.", "tab:si_h1",
                          "$\\Delta$FPR, unseen $-$ seen hospital")
    t_h2 = contrast_table("h2", ["", "_S2", "_S3", "_S4", "_S5", "_S6", "_S7", "_S9"],
                          "H2 across the primary and sensitivity analyses.", "tab:si_h2",
                          "$\\Delta$TPR, unseen $-$ seen generator (text-prompted fakes)")
    t_h4 = contrast_table("h4a", ["", "_S2", "_S3", "_S6", "_S7"],
                          "H4a across the primary and sensitivity analyses.", "tab:si_h4a",
                          "$\\Delta$TPR, image-conditioned $-$ text-prompted commercial fakes")
    h4b = pd.read_csv(M / "h4b_contrasts.csv").set_index("family")
    t_h4b = table("H4b: among image-conditioned fakes, true-positive rate when the source radiograph's hospital was "
                  "unseen versus seen in training (\\%, posterior median; difference with 95\\% CrI). The preregistered "
                  "prediction was a negative difference.", "tab:si_h4b", ["Family", "Seen", "Unseen", "Difference"],
                  [[FAM[f], f"{h4b.loc[f, 'tpr0_med']:.1f}", f"{h4b.loc[f, 'tpr1_med']:.1f}",
                    ci(*h4b.loc[f, ['delta_pp_med', 'delta_pp_lo95', 'delta_pp_hi95']])] for f in FAM_ORDER])
    t_rob = table("H1 robustness: the primary cell-level model; the per-image model with a patient effect (did not meet "
                  "convergence criteria); the exploratory cell-level model that adds a hospital $\\times$ family "
                  "intercept and novelty slope, \\texttt{(1 + novelty | site:family)} (posterior predictive check "
                  "63/91 cells inside the 95\\% interval against 49/91 for the primary model; one divergent transition "
                  "in 12,000 draws); and the cluster bootstrap over patients (pooled over hospitals, not model-based). "
                  "$\\Delta$FPR in percentage points with 95\\% intervals.", "tab:si_rob",
                  ["Family", "Primary (cells)", "Per-image model", "Hospital $\\times$ family (exploratory)",
                   "Bootstrap (pooled)"], rob, fit=True)

    doc = rf"""%% supplementary.tex — generated by analysis/make_si.py; do not edit by hand (re-run the script).
\documentclass[11pt]{{article}}
\usepackage[margin=2cm]{{geometry}}
\usepackage[T1]{{fontenc}}
\usepackage{{booktabs,graphicx,amsmath,amssymb,array,helvet,caption}}
\renewcommand{{\familydefault}}{{\sfdefault}}
\renewcommand{{\thetable}}{{S\arabic{{table}}}}
\renewcommand{{\thefigure}}{{S\arabic{{figure}}}}
\captionsetup{{font=small,labelfont=bf}}
\title{{Supplementary Information\\[4pt]\large Detecting AI-generated chest radiographs across hospitals and generators: a multi-institution benchmark}}
\author{{Saugat Pandey}}
\date{{}}
\begin{{document}}
\maketitle
\tableofcontents

\section{{Preregistration deviations}}
The study was preregistered on the Open Science Framework (\texttt{{https://doi.org/10.17605/OSF.IO/5VWKJ}}). Table~\ref{{tab:si_dev}} lists every departure from the registered plan.
{t_dev}
\clearpage
\section{{Model diagnostics and checks}}
{t_diag}
\begin{{figure}}[!htbp]\centering
\includegraphics[width=0.75\textwidth]{{figures/si_prior_predictive.pdf}}
\caption{{Prior predictive check, run before registration: population false-positive rates implied by the preregistered priors for every family $\times$ novelty cell.}}
\end{{figure}}
\begin{{figure}}[!htbp]\centering
\includegraphics[width=\textwidth]{{figures/si_ppc_h1.pdf}}
\caption{{Posterior predictive check of the primary H1 model: observed false-positive rate on real test radiographs against the model's prediction for each hospital, family and novelty level. Open circles fall outside the 95\% predictive interval. The model over-predicts seen-hospital rates and cannot fit every hospital-by-family combination, because it assumes a hospital shifts all families equally on the log-odds scale.}}
\end{{figure}}
\clearpage
\section{{Sensitivity analyses}}
S1 PA views only; S2 upper 20\% and 8\% border masked; S3 JPEG (quality 90) before preprocessing; S4 512-pixel images, Open-I excluded; S5 priors twice as wide; S6 thresholds set on validation real images of all hospitals; S7 memorized fakes removed; S8 four-shot LLM prompts (Section~\ref{{sec:llm}}); S9 commercial text fakes restricted to prompts from test-split reports.
{t_h1}
{t_h2}
{t_h4}
{t_h4b}
{t_rob}
\clearpage
\section{{Multimodal LLM evaluation}}\label{{sec:llm}}
\subsection*{{Prompt (version detect-v3, frozen 9 November 2026)}}
\textbf{{System:}} {tex(sys_prompt)}

\noindent\textbf{{User}} (after the image): {tex(user)}

\noindent\textbf{{Output schema:}} JSON object with an integer \texttt{{probability}} and a string \texttt{{reason}}, enforced through each provider's native structured-output interface; the 0--100 range was checked on parsing. Reasoning effort was set to low; no sampling parameters were set; maximum 4,000 output tokens. Four-shot prompts preceded the image with: ``{tex("First, here are four labelled example radiographs from the same set, to show what authentic and AI-generated images in this set look like.")}'' followed by four labelled training-split examples (real NIH, G1 fake, real Open-I, G2 fake) and ``Now the radiograph to assess:''.

\subsection*{{Models}}
Model identifiers returned by the APIs: claude-opus-5-5, claude-haiku-5-5, gpt-6-astra, gpt-5.4-mini-2026-03-17, gemini-4-argon, gemini-3.1-flash-lite. All batches were submitted on 9 November 2026 between 12:45 and 13:42 EST.
{t_llm}
{t_rt}
\clearpage
\section{{Generation of commercial fakes}}
\textbf{{Text-prompted:}} ``{tex(ttxt.replace("{text}", "[cleaned report]"))}''

\noindent\textbf{{Image-conditioned}} (sent with the original source radiograph): ``{tex(ctxt.replace("{text}", "[finding]"))}''

First-attempt refusals: G4 text 2/600, G4 edit 6/283, G5 text 1/600, G5 edit 10/283. After retries, one edit per generator could not be obtained. Pre-scoring exclusions: 12 MONAI images (10 lateral projections, 2 unrecognisable) and 1 Gemini edit that returned two panels.

\begin{{figure}}[!htbp]\centering
\includegraphics[width=\textwidth,height=0.82\textheight,keepaspectratio]{{figures/si_montage.png}}
\caption{{Randomly drawn test images from every source after the shared preprocessing (256 px; seed 20261109). CheXpert images are not shown, as their licence does not permit redistribution. ``Edit'': image-conditioned fake made from a normal test radiograph.}}
\end{{figure}}

\section{{Reporting checklist}}
The completed CLAIM 2024 checklist is provided as a separate file.
\end{{document}}
"""
    (OUT / "supplementary.tex").write_text(doc)
    print("wrote manuscript/supplementary.tex")


if __name__ == "__main__":
    main()
