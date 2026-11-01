"""Registered analysis (OSF 8st47) of the per-subject results written by prereg_run.py.

    python deploy/prereg_analysis.py deploy/prereg_results/F1a_per_subject.csv \
        deploy/prereg_results/F1b_per_subject.csv deploy/prereg_results/F2_per_subject.csv --out deploy/prereg_analysis

Outcome: balanced accuracy per subject on the test trials, float32, averaged over seeds.
Families: F1 (F1a and F1b pooled, dataset as a fixed effect) and F2.  In each family:

  H1   LRT of the recipe main effect (3 df): A + R (+ dataset) vs A (+ dataset)
  H2   LRT of architecture x recipe (9 df): A x R (+ dataset) vs A + R (+ dataset)
       Both: linear mixed model, random intercept per subject, ML fit, recipes R1-R4.
  H2b  per subject c = (R2 - R1)_EEGNet - (R2 - R1)_CSOANet-v2; two-sided one-sample t-test;
       supported only if significant after Holm and the mean is positive.
       Holm over H1, H2, H2b at alpha 0.05.
  H3   6 architecture pairs x 4 recipes: Bonferroni-adjusted (24) CIs of the paired per-subject differences
       (99.79 %).  Category: equivalent if the CI lies strictly inside +-2 points (this takes precedence),
       otherwise A better / B better if it excludes 0, otherwise inconclusive.
  H4   the 6 pairs under condition T at the same level and with the same categories, and Kendall's tau
       between the four architectures' mean accuracies under R1 and under T.

Sensitivity analyses: int8 balanced accuracy in place of float32; aligned-rank-transform versions of H1 and
H2; F1a and F1b analysed separately (H1, H2, H2b, H3, H4, without the dataset term).
"""
import argparse
import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ARCHS = ("eegnet", "shallow", "mibminet", "csoanet")
RECIPES = ("R1", "R2", "R3", "R4")
ALPHA = 0.05
N_H3 = 6 * 4
CI_LEVEL = 1 - ALPHA / N_H3                     # 0.99792 -> "99.79 %"
EQUIV = 0.02                                    # +-2 percentage points on the accuracy scale
FAMILIES = {"F1": ("F1a", "F1b"), "F2": ("F2",)}


# ----------------------------------------------------------------------------- data
def seed_means(df, metric="bacc_fp32"):
    """One value per (cell, subject, arch, recipe): the mean over seeds.  Subject IDs are prefixed with the
    dataset so that F1a and F1b subjects stay distinct."""
    d = df.groupby(["cell", "dataset", "subject", "arch", "recipe"], as_index=False)[metric].mean()
    d = d.rename(columns={metric: "acc"})
    d["sid"] = d["dataset"].astype(str) + ":" + d["subject"].astype(str)
    return d


def _wide(d, recipe):
    return d[d.recipe == recipe].pivot(index="sid", columns="arch", values="acc")


# ----------------------------------------------------------------------------- mixed models
def _fit_ml(d, formula):
    import statsmodels.formula.api as smf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = None
        for method in ("lbfgs", "powell", "nm"):
            try:
                res = smf.mixedlm(formula, d, groups=d["sid"]).fit(reml=False, method=method)
                if np.isfinite(res.llf):
                    break
            except Exception:                    # noqa: BLE001  try the next optimiser
                continue
    if res is None:
        raise RuntimeError(f"mixed model did not fit: {formula}")
    return res


def lrt(d, full, reduced, y="acc"):
    """Likelihood-ratio test between two nested ML-fitted mixed models with a random intercept per subject."""
    f, r = _fit_ml(d, f"{y} ~ {full}"), _fit_ml(d, f"{y} ~ {reduced}")
    df = len(f.fe_params) - len(r.fe_params)
    stat = max(0.0, 2 * (f.llf - r.llf))
    return dict(chi2=float(stat), df=int(df), p=float(stats.chi2.sf(stat, df)))


def _terms(d):
    return " + C(dataset)" if d["dataset"].nunique() > 1 else ""


def h1(d, y="acc"):
    t = _terms(d)
    return lrt(d, f"C(arch) + C(recipe){t}", f"C(arch){t}", y)


def h2(d, y="acc"):
    t = _terms(d)
    return lrt(d, f"C(arch) * C(recipe){t}", f"C(arch) + C(recipe){t}", y)


def h2b(d):
    w1, w2 = _wide(d, "R1"), _wide(d, "R2")
    c = ((w2["eegnet"] - w1["eegnet"]) - (w2["csoanet"] - w1["csoanet"])).dropna()
    t = stats.ttest_1samp(c, 0.0)
    return dict(mean_c=float(c.mean()), sd_c=float(c.std(ddof=1)), n=int(len(c)), t=float(t.statistic),
                p=float(t.pvalue))


def holm(pvals, alpha=ALPHA):
    """Holm step-down: adjusted p-values and rejections, in input order."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj, adj <= alpha


# ----------------------------------------------------------------------------- intervals
def category(lo, hi, equiv=EQUIV):
    """Registered categories; 'equivalent' takes precedence over 'A better' / 'B better' (Yu, 2026-11-01)."""
    if -equiv < lo and hi < equiv:
        return "equivalent"
    if lo > 0:
        return "A better"
    if hi < 0:
        return "B better"
    return "inconclusive"


def paired_ci(diff, level=CI_LEVEL):
    diff = np.asarray(diff, float)
    diff = diff[np.isfinite(diff)]
    n, m = len(diff), float(np.mean(diff))
    se = float(np.std(diff, ddof=1) / np.sqrt(n))
    q = stats.t.ppf(1 - (1 - level) / 2, n - 1)
    return dict(n=n, mean_diff=m, lo=m - q * se, hi=m + q * se)


def pair_table(d, recipes, level=CI_LEVEL):
    rows = []
    for rec in recipes:
        w = _wide(d, rec)
        for a, b in itertools.combinations(ARCHS, 2):
            if a not in w or b not in w:
                continue
            ci = paired_ci((w[a] - w[b]).dropna(), level)
            rows.append(dict(recipe=rec, A=a, B=b, level=level, **ci, category=category(ci["lo"], ci["hi"])))
    return pd.DataFrame(rows)


def h4(d):
    table = pair_table(d, ["T"])
    m1 = d[d.recipe == "R1"].groupby("arch").acc.mean().reindex(ARCHS)
    mt = d[d.recipe == "T"].groupby("arch").acc.mean().reindex(ARCHS)
    tau = stats.kendalltau(m1.to_numpy(), mt.to_numpy())
    return table, dict(tau=float(tau.statistic), p=float(tau.pvalue),
                       mean_R1=m1.round(6).to_dict(), mean_T=mt.round(6).to_dict())


# ----------------------------------------------------------------------------- ART
def art_lrt(d, effect):
    """Aligned rank transform (Wobbrock et al. 2011) for 'recipe' (H1) or 'arch:recipe' (H2): residuals
    from the full factorial cell means plus the estimated effect, ranked, then the same LRT as H1 / H2."""
    d = d.copy()
    cells = ["arch", "recipe"] + (["dataset"] if d["dataset"].nunique() > 1 else [])
    resid = d.acc - d.groupby(cells).acc.transform("mean")
    grand = d.acc.mean()
    mr = d.groupby("recipe").acc.transform("mean")
    if effect == "recipe":
        est = mr - grand
    elif effect == "arch:recipe":
        est = d.groupby(["arch", "recipe"]).acc.transform("mean") - d.groupby("arch").acc.transform("mean") - mr + grand
    else:
        raise ValueError(effect)
    d["art"] = stats.rankdata(resid + est)
    return (h1 if effect == "recipe" else h2)(d, y="art")


# ----------------------------------------------------------------------------- family
def confirmatory(d, with_h4=True):
    d4 = d[d.recipe.isin(RECIPES)]
    r1, r2, r2b = h1(d4), h2(d4), h2b(d4)
    adj, rej = holm([r1["p"], r2["p"], r2b["p"]])
    for r, a, j in zip((r1, r2, r2b), adj, rej):
        r.update(p_holm=float(a), significant=bool(j))
    r2b["supported"] = bool(r2b["significant"] and r2b["mean_c"] > 0)
    out = dict(H1=r1, H2=r2, H2b=r2b, n_subjects=int(d4.sid.nunique()))
    tables = dict(H3=pair_table(d4, RECIPES))
    if with_h4 and "T" in set(d.recipe):
        tables["H4"], out["H4_tau"] = h4(d)
    return out, tables


def analyse(df, out_dir=None):
    """Run every registered analysis.  Returns a nested dict; writes JSON and CSV tables if out_dir is set."""
    report, all_tables = {}, {}
    for fam, cells in FAMILIES.items():
        sub = df[df.cell.isin(cells)]
        if sub.empty:
            continue
        d = seed_means(sub, "bacc_fp32")
        res, tables = confirmatory(d)
        res["sensitivity"] = {}
        d8 = seed_means(sub, "bacc_int8")
        res["sensitivity"]["int8"], t8 = confirmatory(d8)
        res["sensitivity"]["ART"] = dict(H1=art_lrt(d[d.recipe.isin(RECIPES)], "recipe"),
                                         H2=art_lrt(d[d.recipe.isin(RECIPES)], "arch:recipe"))
        for k, v in t8.items():
            tables[f"{k}_int8"] = v
        if len(cells) > 1:
            for c in cells:
                if (sub.cell == c).any():
                    res["sensitivity"][c], tc = confirmatory(seed_means(sub[sub.cell == c], "bacc_fp32"))
                    for k, v in tc.items():
                        tables[f"{k}_{c}"] = v
        report[fam] = res
        all_tables[fam] = tables
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "analysis.json").write_text(json.dumps(report, indent=1))
        for fam, tables in all_tables.items():
            for k, t in tables.items():
                t.to_csv(out_dir / f"{fam}_{k}.csv", index=False)
    return report, all_tables


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", help="per-subject CSVs from prereg_run.py")
    ap.add_argument("--out", default="deploy/prereg_analysis")
    a = ap.parse_args(argv)
    df = pd.concat([pd.read_csv(p) for p in a.results], ignore_index=True)
    report, tables = analyse(df, a.out)
    for fam, r in report.items():
        print(f"\n== {fam} ({r['n_subjects']} subjects)")
        for h in ("H1", "H2", "H2b"):
            print(f"  {h}: " + ", ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in r[h].items()))
        print(tables[fam]["H3"].to_string(index=False))
        if "H4" in tables[fam]:
            print(tables[fam]["H4"].to_string(index=False))
            print(f"  Kendall tau R1 vs T: {r['H4_tau']['tau']:.3f}")
    print(f"\nWritten to {a.out}")


if __name__ == "__main__":
    main()
