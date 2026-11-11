"""H5 (OSF 5vwkj): LLMs vs trained detectors on the identical 2,000-image subset.

LLM metrics per model x condition (run 1 unless stated):
  AUC (cluster bootstrap 2,000 over source groups), TPR/FPR at threshold 50, ECE (10 equal-mass bins),
  Brier score (probability / 100 vs label), refusal/unparseable counts, cost per 1,000 images,
  test-retest (runs 1-3 on the retest subset): Fleiss-style mean pairwise Cohen's kappa at 50, and the
  mean within-image SD of the probability.
Per-stratum rates (FPR per hospital, TPR per generator x arm) at threshold 50 with Wilson 95% CIs.
Trained detectors on the same images: per fitted detector, AUC and flags at its frozen threshold;
summarised per family (median and IQR over fits).
Outputs: results/tables/llm_metrics.csv, llm_strata.csv, llm_retest.csv, trained_on_llm_subset.csv
Usage: python analysis/llm_metrics.py
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/tables"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20261109)
B = 2000

sub = pd.read_csv(ROOT / "data/meta/llm_subset.csv")
meta = pd.read_csv(ROOT / "data/meta/images.csv", keep_default_na=False, dtype=str).set_index("image_id")
sub["group_id"] = meta.loc[sub.image_id, "group_id"].values
grp = dict(zip(sub.image_id, sub.group_id))
S = pd.read_csv(ROOT / "results/scores/llm/llm_scores.csv")
S["y"] = (S.cls == "fake").astype(int)


def ece_equal_mass(p, y, bins=10):
    o = np.argsort(p)
    return float(sum(abs(p[o][b].mean() - y[o][b].mean()) * len(b) for b in np.array_split(np.arange(len(p)), bins)) / len(p))


def boot_auc(df):
    g = df.image_id.map(grp).values
    ug, inv = np.unique(g, return_inverse=True)
    idx_by = [np.flatnonzero(inv == i) for i in range(len(ug))]
    out = []
    for _ in range(B):
        take = np.concatenate([idx_by[i] for i in rng.integers(len(ug), size=len(ug))])
        yy = df.y.values[take]
        if 0 < yy.sum() < len(yy):
            out.append(roc_auc_score(yy, df.probability.values[take]))
    return np.quantile(out, [.025, .975])


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (c - h), 100 * (c + h)


rows, strata = [], []
for (model, cond), g in S[S.run == 1].groupby(["model", "condition"]):
    ok = g[g.status == "ok"]
    p, y = ok.probability.values / 100, ok.y.values
    lo, hi = boot_auc(ok)
    rows.append({"model": model, "condition": cond, "n": len(g), "ok": len(ok),
                 "refusal": int((g.status == "refusal").sum()), "unparseable": int((g.status == "unparseable").sum()),
                 "api_error": int(g.status.str.startswith("api_").sum()),
                 "auc": roc_auc_score(y, p), "auc_lo95": lo, "auc_hi95": hi,
                 "fpr50": 100 * np.mean(p[y == 0] > .5), "tpr50": 100 * np.mean(p[y == 1] > .5),
                 "ece": ece_equal_mass(p, y), "brier": float(np.mean((p - y) ** 2)),
                 "mean_p_real": 100 * p[y == 0].mean(), "mean_p_fake": 100 * p[y == 1].mean(),
                 "cost_per_1000": 1000 * g.cost.sum() / len(g)})
    for st, h in ok.groupby("stratum"):
        k = int((h.probability > 50).sum())
        strata.append({"model": model, "condition": cond, "stratum": st, "cls": h.cls.iloc[0], "n": len(h),
                       "flag_pct": 100 * k / len(h), "lo95": wilson(k, len(h))[0], "hi95": wilson(k, len(h))[1],
                       "mean_p": h.probability.mean()})
pd.DataFrame(rows).round(4).to_csv(OUT / "llm_metrics.csv", index=False)
pd.DataFrame(strata).round(2).to_csv(OUT / "llm_strata.csv", index=False)

ret = []
for model, g in S[(S.condition == "zeroshot") & S.image_id.isin(sub.image_id[sub.retest])].groupby("model"):
    w = g[g.status == "ok"].pivot_table(index="image_id", columns="run", values="probability")
    if w.shape[1] < 3:
        continue
    w = w.dropna()
    kap = [cohen_kappa_score(w[a] > 50, w[b] > 50) for a, b in combinations(w.columns, 2)]
    agree = np.mean([(w[a] > 50).eq(w[b] > 50).mean() for a, b in combinations(w.columns, 2)])
    ret.append({"model": model, "n_images": len(w), "kappa_mean": np.mean(kap), "kappa_min": np.min(kap),
                "agreement_pct": 100 * agree, "within_image_sd": float(w.std(axis=1).mean()),
                "icc_like_r": float(np.mean([np.corrcoef(w[a], w[b])[0, 1] for a, b in combinations(w.columns, 2)]))})
pd.DataFrame(ret).round(3).to_csv(OUT / "llm_retest.csv", index=False)

# trained detectors on the same images
ids = set(sub.image_id)
tr = []
for kind in ["reals", "fakes"]:
    d = pd.read_parquet(ROOT / f"data/analysis/{kind}_long.parquet", columns=["image_id", "config", "seed", "family", "score", "flag", "cls"])
    tr.append(d[d.image_id.isin(ids)])
T = pd.concat(tr)
T["y"] = (T.cls == "fake").astype(int)
per_fit = T.groupby(["family", "config", "seed"]).apply(
    lambda h: pd.Series({"auc": roc_auc_score(h.y, h.score), "fpr": 100 * h.flag[h.y == 0].mean(),
                         "tpr": 100 * h.flag[h.y == 1].mean(), "n": len(h)}), include_groups=False).reset_index()
summ = per_fit.groupby("family").agg(fits=("auc", "size"), auc_med=("auc", "median"),
                                     auc_q25=("auc", lambda x: x.quantile(.25)), auc_q75=("auc", lambda x: x.quantile(.75)),
                                     auc_max=("auc", "max"), fpr_med=("fpr", "median"), tpr_med=("tpr", "median")).reset_index()
summ.round(3).to_csv(OUT / "trained_on_llm_subset.csv", index=False)
per_fit.round(4).to_csv(OUT / "trained_on_llm_subset_perfit.csv", index=False)
print(pd.DataFrame(rows).round(3)[["model", "condition", "ok", "auc", "auc_lo95", "auc_hi95", "fpr50", "tpr50", "ece", "brier", "cost_per_1000"]])
print(pd.DataFrame(ret).round(3))
print(summ.round(3))
