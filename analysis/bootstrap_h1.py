"""Frequentist cross-check for H1 and the headline TPRs (OSF 5vwkj: cluster bootstrap, 2,000 resamples).

H1   : FPR(unseen) - FPR(seen) per detector family, pooled over sites, configs and seeds; resampling
       unit = patient (reals). Also FPR per family x site x novelty.
TPR  : per family x generator x conditioning x novelty_gen; resampling unit = source report group (fakes).
Estimator: ratio of sums (total flags / total image-fits) within the resample.
Outputs: results/tables/h1_bootstrap.csv, fpr_by_site_bootstrap.csv, tpr_bootstrap.csv
Usage: python analysis/bootstrap_h1.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/tables"
OUT.mkdir(parents=True, exist_ok=True)
B = 2000
rng = np.random.default_rng(20261109)


def boot(df, cluster, cells):
    """df has k, n, cluster, cell columns. Returns point estimate and B bootstrap replicates per cell."""
    g = df.groupby([cluster] + cells, observed=True)[["k", "n"]].sum().reset_index()
    cl = pd.Index(g[cluster].unique())
    g["ci"] = cl.get_indexer(g[cluster])
    W = rng.multinomial(len(cl), np.ones(len(cl)) / len(cl), size=B).T  # clusters x B
    res = {}
    for key, sub in g.groupby(cells, observed=True):
        w = W[sub.ci.values]  # rows x B
        k = (sub.k.values[:, None] * w).sum(0)
        n = (sub.n.values[:, None] * w).sum(0)
        res[key if isinstance(key, tuple) else (key,)] = (sub.k.sum() / sub.n.sum(), np.where(n > 0, k / np.maximum(n, 1), np.nan))
    return res


def summ(est, reps):
    return {"est": 100 * est, "lo95": 100 * np.nanquantile(reps, .025), "hi95": 100 * np.nanquantile(reps, .975)}


R = pd.read_parquet(ROOT / "data/analysis/reals_agg.parquet")
r = boot(R, "patient_id", ["family", "novelty_site"])
rows = []
for fam in sorted({k[0] for k in r}):
    s, u = r[(fam, 0)], r[(fam, 1)]
    rows.append({"family": fam, **{f"seen_{k}": v for k, v in summ(*s).items()},
                 **{f"unseen_{k}": v for k, v in summ(*u).items()},
                 **{f"delta_pp_{k}": v for k, v in summ(u[0] - s[0], u[1] - s[1]).items()},
                 "boot_pr_pos": float(np.mean(u[1] - s[1] > 0))})
pd.DataFrame(rows).round(2).to_csv(OUT / "h1_bootstrap.csv", index=False)
print(pd.DataFrame(rows).round(1)[["family", "seen_est", "unseen_est", "delta_pp_est", "delta_pp_lo95", "delta_pp_hi95"]])

r = boot(R, "patient_id", ["family", "site", "novelty_site"])
pd.DataFrame([{"family": k[0], "site": k[1], "novelty_site": k[2], **summ(*v)} for k, v in r.items()]).round(2) \
    .to_csv(OUT / "fpr_by_site_bootstrap.csv", index=False)

F = pd.read_parquet(ROOT / "data/analysis/fakes_agg.parquet")
r = boot(F, "group_id", ["family", "generator", "conditioning", "novelty_gen"])
pd.DataFrame([{"family": k[0], "generator": k[1], "conditioning": k[2], "novelty_gen": k[3], **summ(*v)}
              for k, v in r.items()]).round(2).to_csv(OUT / "tpr_bootstrap.csv", index=False)
print("wrote", OUT)
