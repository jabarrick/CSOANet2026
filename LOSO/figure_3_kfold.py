#!/usr/bin/env python
"""Generate Fig 3: 5-fold CV accuracy across 7 methods (N=46), redesigned for clarity.

Inputs: 5-fold per-subject csvs (csoa, deep, shallow, eegnet, ml).
Output: fig3_kfold.png — single-panel box+strip plot, short labels.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

C_AXIS = "#5A5A5A"


def _load_5fold(csoa_csv, deep_csv, shallow_csv, eegnet_csv, ml_csv):
    """Return dict {method_name: {sub_id: acc}}."""
    dl_csvs = {
        "CSOA": (csoa_csv, "acc_mean"),
        "DeepConvNet": (deep_csv, "acc_mean"),
        "ShallowConvNet": (shallow_csv, "acc_mean"),
        "EEGNet": (eegnet_csv, "cv_eegnet"),
    }
    out = {}
    for name, (csv, col) in dl_csvs.items():
        df = pd.read_csv(csv, dtype={"subject": str})
        df["subject"] = df["subject"].str.zfill(2)
        out[name] = dict(zip(df["subject"], df[col]))

    # ML: single csv with three columns
    ml = pd.read_csv(ml_csv, dtype={"subject": str})
    ml["subject"] = ml["subject"].str.zfill(2)
    out["LDA"] = dict(zip(ml["subject"], ml["cv_lda"]))
    out["SVM"] = dict(zip(ml["subject"], ml["cv_svm"]))
    out["RF"] = dict(zip(ml["subject"], ml["cv_rf"]))
    return out


def generate_fig3(csoa_csv, deep_csv, shallow_csv, eegnet_csv, ml_csv,
                  out_path="fig3_kfold.png"):
    data = _load_5fold(csoa_csv, deep_csv, shallow_csv, eegnet_csv, ml_csv)

    # Display order (LDA → SVM → RF → EEGNet → CSOA → Shallow → Deep)
    methods_full = ["LDA", "SVM", "RF", "EEGNet", "CSOA", "ShallowConvNet", "DeepConvNet"]
    labels_short = ["LDA", "SVM", "RF", "EEGNet", "CSOA", "Shallow", "Deep"]

    # Color scheme matching Fig 8 (progressive blue gradient)
    colors_fill = ["#FAC775", "#9FE1CB", "#F5C4B3", "#CECBF6",
                   "#534AB7", "#3122A3", "#1B0E66"]
    colors_edge = ["#633806", "#085041", "#712B13", "#3C3489",
                   "#2A2470", "#1A1252", "#0C0633"]

    # Subjects sorted
    sids = sorted(data["CSOA"].keys())
    accs = [np.array([data[m][s] for s in sids]) for m in methods_full]
    means = [a.mean() for a in accs]
    stds = [a.std(ddof=1) for a in accs]

    fig, ax = plt.subplots(figsize=(10, 4.2))
    pos = np.arange(len(methods_full))

    bp = ax.boxplot(
        accs, positions=pos, widths=0.55, vert=True,
        patch_artist=True, showfliers=False,
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(color=C_AXIS, linewidth=0.6),
        capprops=dict(color=C_AXIS, linewidth=0.6),
    )
    for patch, fc, ec in zip(bp["boxes"], colors_fill, colors_edge):
        patch.set_facecolor(fc)
        patch.set_edgecolor(ec)
        patch.set_linewidth(0.8)

    # Strip overlay (jittered scatter)
    rng = np.random.default_rng(seed=0)
    for i, (a, ec) in enumerate(zip(accs, colors_edge)):
        jitter = rng.uniform(-0.18, 0.18, size=len(a))
        ax.scatter(pos[i] + jitter, a, s=12, c=ec, alpha=0.40,
                   edgecolor="none", zorder=3)

    # Mean ± SD label above each box
    for i, (mn, sd, ec) in enumerate(zip(means, stds, colors_edge)):
        ax.text(pos[i], 1.035, f"{mn:.3f}\n±{sd:.3f}",
                ha="center", va="bottom", fontsize=7.5,
                fontweight="bold", color=ec, linespacing=1.1)

    # Chance line
    ax.axhline(0.5, color="#D04040", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.text(len(methods_full) - 0.5, 0.51, "Chance (0.5)",
            ha="right", va="bottom", fontsize=7.5, color="#D04040")

    ax.set_xticks(pos)
    ax.set_xticklabels(labels_short, fontsize=9, color=C_AXIS)
    ax.set_xlim(-0.5, len(methods_full) - 0.5)
    ax.set_ylim(0.45, 1.10)
    ax.set_ylabel("5-fold CV accuracy", color=C_AXIS, fontsize=10)
    ax.tick_params(axis="y", colors=C_AXIS, labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_AXIS)

    fig.tight_layout()
    out = Path(out_path)
    fig.savefig(out, dpi=300, bbox_inches="tight",
                pad_inches=0.10, facecolor="white")
    plt.close(fig)
    return out


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--csoa-csv", required=True)
    pa.add_argument("--deep-csv", required=True)
    pa.add_argument("--shallow-csv", required=True)
    pa.add_argument("--eegnet-csv", required=True)
    pa.add_argument("--ml-csv", required=True)
    pa.add_argument("--out", default="fig3_kfold.png")
    a = pa.parse_args()
    p = generate_fig3(a.csoa_csv, a.deep_csv, a.shallow_csv,
                      a.eegnet_csv, a.ml_csv, a.out)
    print(f"  Wrote {p.resolve()}")


if __name__ == "__main__":
    main()
