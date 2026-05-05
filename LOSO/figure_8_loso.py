"""Generate Fig. 8: Cross-subject generalization (Leave-One-Subject-Out).

Three-panel composite:
  A. LOSO accuracy distribution across N folds (box + strip + reference lines)
  B. 5-fold vs LOSO paired comparison (one line per subject)
  C. CSOA gate weights under LOSO (mean ± SD across folds)

Inputs:
  --loso-csv     csoanet2026_loso.csv (output of batch_run_loso.py)
  --kfold-csv    csoanet2026_kfold.csv (per-subject 5-fold accuracy;
                 column 'subject' or 'held_out' + 'test_acc' or 'mean_acc')
  --kfold-mean   Reference 5-fold mean accuracy (default 0.9867).
                 Used as the dashed green reference line in Panel A.
  --kfold-gates  Optional 6-tuple of 5-fold mean gate weights to overlay
                 on Panel C as translucent bars (--kfold-gates 0.20 0.18
                 0.15 0.18 0.12 0.17 etc).

Output: fig8_loso.png at 300 DPI.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mp


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.linewidth": 0.6,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# Color palette (paired ramps consistent with Fig. 1B)
C_AXIS = "#444441"
C_BOX_FILL = "#CECBF6"
C_BOX_EDGE = "#3C3489"
C_STRIP    = "#534AB7"
C_HARD     = "#E24B4A"     # hard cases highlighted
C_HARD_E   = "#791F1F"
C_KFOLD_REF = "#1D9E75"    # green dashed line for 5-fold mean
C_LOSO_REF  = "#D85A30"    # orange dashed line for LOSO mean
C_LINE_LIGHT = "#9C9A93"   # paired-plot soft lines
C_BAR_LOSO   = "#F5C4B3"   # coral for LOSO gate bars (Panel C)
C_BAR_LOSO_E = "#712B13"
C_BAR_KFOLD  = "#9FE1CB"   # teal overlay for 5-fold gates
C_BAR_KFOLD_E = "#085041"


GATE_LABELS = ['BG', 'Alp', 'TD', 'BxA', 'AxT', 'BxT']
GATE_FULL   = ['Beta+Gamma', 'Alpha', 'Theta+Delta', 'Beta×Alpha',
               'Alpha×Theta', 'Beta×Theta']


def _fold_id(s):
    return f"{int(s):02d}"


def _hard_threshold(acc, pad=0.005):
    """Subjects with test_acc < 0.95 are flagged as hard cases."""
    return 0.95 - pad


# ===========================================================================
# Panel A: LOSO accuracy distribution
# ===========================================================================
def _panel_a(ax, loso_df, kfold_mean, multi_csv=None):
    """Panel A: multi-method LOSO accuracy distribution.
    If multi_csv is provided (5-way comparison csv), draw 5 boxes side-by-side.
    Otherwise fall back to single-box CSOA-only display.
    """
    if multi_csv is not None and Path(multi_csv).exists():
        return _panel_a_multi(ax, multi_csv, kfold_mean)
    return _panel_a_single(ax, loso_df, kfold_mean)


def _panel_a_multi(ax, multi_csv, kfold_mean):
    """5-way side-by-side box plots."""
    df = pd.read_csv(multi_csv)
    methods = ['LDA', 'SVM', 'RF', 'EEGNet', 'CSOA', 'ShallowConvNet', 'DeepConvNet']
    labels_short = ['LDA', 'SVM', 'RF', 'EEGNet', 'CSOA', 'Shallow', 'Deep']
    colors_fill = ["#FAC775", "#9FE1CB", "#F5C4B3", "#CECBF6", "#534AB7", "#3122A3", "#1B0E66"]
    colors_edge = ["#633806", "#085041", "#712B13", "#3C3489", "#2A2470", "#1A1252", "#0C0633"]

    pos = np.arange(len(methods))
    accs = [df[m].values for m in methods]
    means = [a.mean() for a in accs]

    bp = ax.boxplot(
        accs, positions=pos, widths=0.5, vert=True,
        patch_artist=True, showfliers=False,
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(color=C_AXIS, linewidth=0.6),
        capprops=dict(color=C_AXIS, linewidth=0.6),
    )
    for patch, fc, ec in zip(bp['boxes'], colors_fill, colors_edge):
        patch.set_facecolor(fc)
        patch.set_edgecolor(ec)
        patch.set_linewidth(0.8)

    # Strip overlay (jittered)
    rng = np.random.default_rng(seed=0)
    for i, (m, a, ec) in enumerate(zip(methods, accs, colors_edge)):
        jitter = rng.uniform(-0.20, 0.20, size=len(a))
        ax.scatter(pos[i] + 0.45 + jitter, a, s=10, c=ec, alpha=0.45,
                   edgecolor='none', zorder=3)

    # 5-fold reference line for CSOA — multi-line label at right edge
    ax.axhline(kfold_mean, color=C_KFOLD_REF, linestyle="--",
               linewidth=0.8, alpha=0.6)
    ax.text(len(methods) - 0.42, kfold_mean,
            f"CSOA\n5-fold\nmean\n{kfold_mean:.3f}",
            ha="left", va="center", fontsize=7, color=C_KFOLD_REF,
            style='italic', linespacing=1.1)

    # Mean labels above each box (large font, well above plot area)
    for i, mn in enumerate(means):
        ax.text(pos[i], 1.045, f"{mn:.3f}",
                ha='center', va='bottom', fontsize=8.5, fontweight='bold',
                color=colors_edge[i])

    ax.set_xticks(pos)
    ax.set_xticklabels(labels_short, fontsize=9, color=C_AXIS)
    ax.set_xlim(-0.55, len(methods) - 0.5 + 0.85)
    ax.set_ylim(0.45, 1.10)
    ax.set_ylabel("LOSO test accuracy", color=C_AXIS, fontsize=9)
    ax.tick_params(axis='y', colors=C_AXIS, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_AXIS)

    ax.set_title("(A) LOSO accuracy across methods (N = 46)",
                 loc="left", fontsize=10, fontweight="bold", pad=6)


def _panel_a_single(ax, loso_df, kfold_mean):
    """Original single-box CSOA-only Panel A (kept as fallback)."""
    acc = loso_df['test_acc'].values
    sids = [_fold_id(s) for s in loso_df['held_out'].values]
    N = len(acc)
    loso_mean = float(acc.mean())
    loso_med = float(np.median(acc))
    q1, q3 = np.percentile(acc, [25, 75])

    # Single-column box plot
    box_x = 0.6
    bp = ax.boxplot(
        [acc], positions=[box_x], widths=0.35, vert=True,
        patch_artist=True, showfliers=False,
        medianprops=dict(color=C_BOX_EDGE, linewidth=1.4),
        boxprops=dict(facecolor=C_BOX_FILL, edgecolor=C_BOX_EDGE,
                      linewidth=0.8),
        whiskerprops=dict(color=C_AXIS, linewidth=0.6),
        capprops=dict(color=C_AXIS, linewidth=0.6),
    )

    # Strip plot (jittered scatter), color hard cases
    rng = np.random.default_rng(seed=0)
    jitter = rng.uniform(-0.18, 0.18, size=N)
    hard_thr = _hard_threshold(acc)
    is_hard = acc < hard_thr
    strip_x = 1.4
    ax.scatter(strip_x + jitter[~is_hard], acc[~is_hard],
               s=22, c=C_STRIP, alpha=0.6, edgecolor='none', zorder=3)
    ax.scatter(strip_x + jitter[is_hard], acc[is_hard],
               s=32, c=C_HARD, edgecolor=C_HARD_E, linewidths=0.8, zorder=4)

    # Hard case labels (right side)
    label_x = 2.2
    label_data = sorted(
        [(sids[i], acc[i]) for i in range(N) if is_hard[i]],
        key=lambda t: t[1]
    )
    for sid, a_val in label_data:
        idx = sids.index(sid)
        ax.annotate(
            f"Sub-{sid} ({a_val:.3f})",
            xy=(strip_x + jitter[idx] + 0.1, a_val),
            xytext=(label_x, a_val),
            fontsize=7, color=C_HARD_E, va='center',
            arrowprops=dict(arrowstyle="-", color=C_HARD_E, lw=0.5,
                            connectionstyle="arc3,rad=0"),
        )

    # Reference lines
    xmin, xmax = 0.1, 3.0
    ax.axhline(kfold_mean, color=C_KFOLD_REF, linestyle="--",
               linewidth=0.8, xmin=0, xmax=1, zorder=1)
    ax.axhline(loso_mean, color=C_LOSO_REF, linestyle="--",
               linewidth=0.8, xmin=0, xmax=1, zorder=1)
    ax.text(xmax - 0.05, kfold_mean + 0.003,
            f"5-fold mean = {kfold_mean:.4f}",
            ha="right", va="bottom", fontsize=7, color=C_KFOLD_REF)
    ax.text(xmax - 0.05, loso_mean - 0.005,
            f"LOSO mean = {loso_mean:.4f}",
            ha="right", va="top", fontsize=7, color=C_LOSO_REF)

    # Stats annotation
    pct_above = 100 * (acc >= 0.95).mean()
    ax.text(xmin + 0.05, 0.685,
            f"N = {N} folds\n"
            f"Median = {loso_med:.4f}\n"
            f"IQR = [{q1:.3f}, {q3:.3f}]\n"
            f"≥ 0.95: {(acc >= 0.95).sum()}/{N} ({pct_above:.1f}%)",
            ha="left", va="bottom", fontsize=7.5, color=C_AXIS,
            bbox=dict(facecolor='white', edgecolor=C_AXIS,
                      linewidth=0.4, pad=4, alpha=0.95))

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0.65, 1.01)
    ax.set_xticks([box_x, strip_x])
    ax.set_xticklabels(['box', '46 folds'], fontsize=8, color=C_AXIS)
    ax.set_ylabel("LOSO test accuracy", color=C_AXIS, fontsize=9)
    ax.tick_params(axis='y', colors=C_AXIS, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_AXIS)
    ax.set_title("(A) LOSO accuracy distribution",
                 loc="left", fontsize=10, fontweight="bold", pad=6)


# ===========================================================================
# Panel B: 5-fold vs LOSO paired comparison
# ===========================================================================
def _panel_b(ax, loso_df, kfold_per_subj):
    """kfold_per_subj: dict {sid_str -> 5-fold test_acc}.
    Subjects without a 5-fold entry are skipped."""
    pairs = []
    for _, row in loso_df.iterrows():
        sid = _fold_id(row['held_out'])
        if sid in kfold_per_subj:
            pairs.append((sid, kfold_per_subj[sid], float(row['test_acc'])))
    if not pairs:
        ax.text(0.5, 0.5, "No 5-fold per-subject data\nprovided "
                "(--kfold-csv).",
                transform=ax.transAxes, ha='center', va='center',
                fontsize=10, color=C_AXIS)
        ax.set_title("(B) 5-fold vs LOSO (data missing)",
                     loc="left", fontsize=10, fontweight="bold", pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        return

    sids, k_acc, l_acc = zip(*pairs)
    k_acc = np.asarray(k_acc); l_acc = np.asarray(l_acc)
    diffs = l_acc - k_acc
    hard_thr = _hard_threshold(l_acc)
    is_hard = l_acc < hard_thr

    x_left, x_right = 0.0, 1.0
    for i, sid in enumerate(sids):
        col = C_HARD if is_hard[i] else C_LINE_LIGHT
        lw = 1.0 if is_hard[i] else 0.4
        alpha = 0.95 if is_hard[i] else 0.55
        zo = 4 if is_hard[i] else 2
        ax.plot([x_left, x_right], [k_acc[i], l_acc[i]],
                color=col, linewidth=lw, alpha=alpha, zorder=zo)
        ax.scatter([x_left, x_right], [k_acc[i], l_acc[i]],
                   s=14 if is_hard[i] else 8,
                   c=col, edgecolor='none', alpha=alpha, zorder=zo+1)
        if is_hard[i]:
            ax.annotate(f"Sub-{sid}",
                        xy=(x_right + 0.02, l_acc[i]),
                        fontsize=7, color=C_HARD_E, va='center')

    # Means (heavy lines)
    ax.plot([x_left, x_right], [k_acc.mean(), l_acc.mean()],
            color="black", linewidth=2.0, zorder=5,
            marker='o', markersize=6,
            label=f"Mean ({k_acc.mean():.3f} → {l_acc.mean():.3f})")

    # Group labels under each end
    ax.text(x_left, 0.66, "5-fold", ha='center', va='top',
            fontsize=9, fontweight='bold', color=C_AXIS)
    ax.text(x_right, 0.66, "LOSO", ha='center', va='top',
            fontsize=9, fontweight='bold', color=C_AXIS)

    # Δ annotation — placed ABOVE the plot area, two lines for compact width
    ax.text(x_left - 0.15, 1.060,
            f"Δ (LOSO − 5-fold) = {diffs.mean():+.3f} ± {diffs.std(ddof=1):.3f}\n"
            f"Hard cases: {is_hard.sum()} / {len(sids)} subjects",
            ha='left', va='top', fontsize=7.5, color=C_AXIS,
            bbox=dict(facecolor='white', edgecolor=C_AXIS,
                      linewidth=0.4, pad=3, alpha=1.0))

    ax.set_xlim(-0.25, 1.4)
    ax.set_ylim(0.65, 1.085)
    ax.set_xticks([])
    ax.set_ylabel("Test accuracy", color=C_AXIS, fontsize=9)
    ax.tick_params(axis='y', colors=C_AXIS, labelsize=8)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(C_AXIS)
    ax.legend(loc='lower left', fontsize=7, frameon=False)
    ax.set_title("(B) 5-fold vs LOSO (per-subject paired)",
                 loc="left", fontsize=10, fontweight="bold", pad=6)


# ===========================================================================
# Panel C: CSOA gate weights
# ===========================================================================
def _panel_c(ax, loso_df, kfold_gates=None):
    gate_cols = [f"gate_{g}" for g in GATE_LABELS]
    G = loso_df[gate_cols].values            # (N_folds, 6)
    means = G.mean(axis=0)
    stds = G.std(axis=0, ddof=1)

    x = np.arange(6)
    if kfold_gates is None:
        # LOSO bars only
        ax.bar(x, means, yerr=stds, width=0.65,
               color=C_BAR_LOSO, edgecolor=C_BAR_LOSO_E, linewidth=0.6,
               error_kw=dict(ecolor=C_AXIS, lw=0.6, capsize=3))
    else:
        # Paired bars: 5-fold + LOSO side by side
        w = 0.4
        ax.bar(x - w/2, kfold_gates, width=w,
               color=C_BAR_KFOLD, edgecolor=C_BAR_KFOLD_E, linewidth=0.6,
               label="5-fold (within-subject)")
        ax.bar(x + w/2, means, yerr=stds, width=w,
               color=C_BAR_LOSO, edgecolor=C_BAR_LOSO_E, linewidth=0.6,
               error_kw=dict(ecolor=C_AXIS, lw=0.6, capsize=3),
               label="LOSO (cross-subject)")
        ax.legend(loc='upper right', fontsize=7, frameon=False)

    ax.set_xticks(x)
    ax.set_xticklabels(GATE_LABELS, fontsize=9, color=C_AXIS)
    ax.set_ylabel("Gate weight (softmax)", color=C_AXIS, fontsize=9)
    ax.set_ylim(0, max(0.85, means.max() + stds.max() + 0.1))
    ax.tick_params(axis='y', colors=C_AXIS, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_AXIS)

    dom_pct = (means[2] + means[3]) * 100
    ax.text(0.98, 0.94,
            f"TD + BxA = {dom_pct:.1f}% of total weight\n"
            f"(N = {len(loso_df)} LOSO folds)",
            transform=ax.transAxes, ha='right', va='top', fontsize=7.5,
            color=C_AXIS,
            bbox=dict(facecolor='white', edgecolor=C_AXIS,
                      linewidth=0.4, pad=4, alpha=0.9))
    ax.set_title("(C) CSOA gate weights under LOSO (mean ± SD)",
                 loc="left", fontsize=10, fontweight="bold", pad=6)


# ===========================================================================
# Top-level
# ===========================================================================
def _resolve_kfold_per_subject(kfold_csv):
    """Try to parse a per-subject 5-fold CSV. Accept several common schemas."""
    if kfold_csv is None or not Path(kfold_csv).exists():
        return {}
    df = pd.read_csv(kfold_csv)
    # Find subject column
    sid_col = None
    for c in ['subject', 'sid', 'held_out', 'sub_id', 'sub']:
        if c in df.columns: sid_col = c; break
    if sid_col is None:
        print(f"  [warn] no subject column in {kfold_csv}; skipping Panel B.")
        return {}
    # Find accuracy column (mean or per-fold)
    acc_col = None
    for c in ['acc_mean', 'mean_acc', 'test_acc', 'acc', 'accuracy',
              'cv_eegnet', 'cv_basic_core_2d', 'cv_lda', 'cv_svm', 'cv_rf']:
        if c in df.columns: acc_col = c; break
    if acc_col is None:
        print(f"  [warn] no accuracy column in {kfold_csv}; skipping Panel B.")
        return {}
    out = {}
    for _, row in df.iterrows():
        out[_fold_id(row[sid_col])] = float(row[acc_col])
    return out


def generate_fig8(loso_csv, kfold_csv=None, kfold_mean=0.9867,
                  kfold_gates=None, multi_csv=None, out_path="fig8_loso.png"):
    loso_df = pd.read_csv(loso_csv)
    print(f"  Loaded {len(loso_df)} LOSO folds from {loso_csv}")

    kfold_per_subj = _resolve_kfold_per_subject(kfold_csv)
    if kfold_per_subj:
        print(f"  Loaded {len(kfold_per_subj)} per-subject 5-fold entries")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2),
                             gridspec_kw={"width_ratios": [1.7, 1.0, 1.2],
                                          "wspace": 0.32})
    _panel_a(axes[0], loso_df, kfold_mean, multi_csv=multi_csv)
    _panel_b(axes[1], loso_df, kfold_per_subj)
    _panel_c(axes[2], loso_df, kfold_gates)

    out = Path(out_path)
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.08,
                facecolor="white")
    plt.close(fig)
    return out


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--loso-csv", required=True)
    pa.add_argument("--kfold-csv", default=None)
    pa.add_argument("--multi-csv", default=None,
                    help="5-way per-subject LOSO csv with columns "
                         "[subject, LDA, SVM, RF, EEGNet, CSOA]. "
                         "If given, Panel A shows multi-method box comparison.")
    pa.add_argument("--kfold-mean", type=float, default=0.9867)
    pa.add_argument("--kfold-gates", type=float, nargs=6, default=None,
                    metavar=('BG','Alp','TD','BxA','AxT','BxT'))
    pa.add_argument("--out", default="fig8_loso.png")
    a = pa.parse_args()
    p = generate_fig8(a.loso_csv, a.kfold_csv, a.kfold_mean,
                      a.kfold_gates, a.multi_csv, a.out)
    print(f"  Wrote {p.resolve()}")


if __name__ == "__main__":
    main()
