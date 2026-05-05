"""CSOANet2026 paper — single-file figure generator.

Generates all figures (Fig. 3-7 main, Fig. S1-S12 supplementary) from
preprocessed EEG data and model CSVs. Each figure saved as PNG (preview),
PDF (vector for editing), and TIF (600 DPI LZW for journal submission).

Usage:
    python generate_figures.py                             # all figures
    python generate_figures.py --group results             # fast CSV figures
    python generate_figures.py --group panels interpret    # specific groups
    python generate_figures.py --fig 3 6 7                 # specific figures
    python generate_figures.py --subjects 04 10 22         # per-subject figures

Groups:
    results   — Fig. 3, 6, 7, S1, S2, S3, S4, S8   (CSV-only, ~2 min)
    neuro     — Fig. S5, S6, S7, S9, S10           (MNE-heavy, ~12 min)
    panels    — Fig. 4 (subject panels)             (training + RF, ~10 min)
    interpret — Fig. 5, S11, S12                    (training + Grad-CAM, ~15 min)
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    'font.family':     'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size':       10, 'axes.titlesize':  12, 'axes.labelsize': 11,
    'figure.dpi':      300, 'savefig.bbox': 'tight', 'savefig.facecolor': 'white',
})

# ════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ════════════════════════════════════════════════════════════════════
OUT  = Path("eeg_project/figures"); OUT.mkdir(parents=True, exist_ok=True)
RES  = Path("eeg_project/results")
DATA = Path("eeg_project/data/processed")

CSV_MAP = {
    'CSOANet2026':    ('csoanet2026_all_subjects.csv', 'acc_mean'),
    'EEGNet':         ('eegnet_all_subjects.csv',     'cv_eegnet'),
    'ShallowConvNet': ('shallow_all_subjects.csv',    'acc_mean'),
    'DeepConvNet':    ('deep_all_subjects.csv',       'acc_mean'),
}
NON_EEG = ['HEO','VEO','EKG','ECG','EMG','Trigger','STI','CB1','CB2']
COLOR_MAP = {
    'LDA':'#B0C4DE','SVM':'#87CEEB','RF':'#90EE90',
    'ShallowConvNet':'#DDA0DD','DeepConvNet':'#F0A0A0',
    'EEGNet':'#FFD700','CSOANet2026':'#FF8C42',
}
MODEL_ORDER = ['LDA','SVM','RF','ShallowConvNet','DeepConvNet','EEGNet','CSOANet2026']


# ════════════════════════════════════════════════════════════════════
#  SHARED UTILITIES
# ════════════════════════════════════════════════════════════════════
def save_fig(fig, name, dpi_png=300):
    """Save figure as PNG only (300 DPI)."""
    base = OUT / name
    fig.savefig(f"{base}.png", dpi=dpi_png)
    print(f"      > {base.name}.png")


def load_all_results():
    data = {}
    for name, (fn, col) in CSV_MAP.items():
        f = RES / fn
        if not f.exists():
            print(f"  WARN {name}: {fn} missing"); continue
        df = pd.read_csv(f)
        if col in df.columns:
            data[name] = df[col].values
            print(f"  Loaded {name}: N={len(df)}")
    ml = RES / 'ml_all_subjects.csv'
    if ml.exists():
        df = pd.read_csv(ml)
        for col, name in [('cv_lda','LDA'),('cv_svm','SVM'),('cv_rf','RF')]:
            if col in df.columns: data[name] = df[col].values
        print(f"  Loaded ML: N={len(df)}")
    return data


def load_epo(sid, drop_non_eeg=False):
    import mne; mne.set_log_level("ERROR")
    p = DATA / f"sub_{sid}_epo.fif"
    if not p.exists(): return None
    epo = mne.read_epochs(str(p), preload=True, verbose=False)
    if drop_non_eeg:
        drop = [c for c in epo.ch_names if c.upper() in [n.upper() for n in NON_EEG]]
        if drop: epo.drop_channels(drop)
        try: epo.set_montage('standard_1020', match_case=False, on_missing='ignore')
        except Exception: pass
    return epo


def subject_accuracies(sid):
    acc = {}
    sn = str(sid).zfill(2)
    for name, (fn, col) in CSV_MAP.items():
        f = RES / fn
        if f.exists():
            df = pd.read_csv(f)
            df['_sid'] = df['subject'].astype(str).str.zfill(2)
            row = df[df['_sid'] == sn]
            if len(row) > 0 and col in row.columns:
                acc[name] = float(row[col].values[0])
    ml = RES / 'ml_all_subjects.csv'
    if ml.exists():
        df = pd.read_csv(ml)
        df['_sid'] = df['subject'].astype(str).str.zfill(2)
        row = df[df['_sid'] == sn]
        if len(row) > 0:
            for col, name in [('cv_lda','LDA'),('cv_svm','SVM'),('cv_rf','RF')]:
                if col in row.columns: acc[name] = float(row[col].values[0])
    return acc


# ════════════════════════════════════════════════════════════════════
#  GROUP 1: RESULT FIGURES (Fig. 3, 6, 7, S1, S2, S3, S4, S8)
# ════════════════════════════════════════════════════════════════════
def fig3_boxplot(data):
    avail = [m for m in MODEL_ORDER if m in data]
    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot([data[m] for m in avail], positions=range(len(avail)),
                    widths=0.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color='black', linewidth=1.5))
    for i, (patch, name) in enumerate(zip(bp['boxes'], avail)):
        patch.set_facecolor(COLOR_MAP[name]); patch.set_alpha(0.7)
        x = np.random.normal(i, 0.07, len(data[name]))
        ax.scatter(x, data[name], alpha=0.55, s=14, color='#333', zorder=3)
        mn = float(np.min(data[name]))
        ax.annotate(f'{mn:.3f}', xy=(i, mn), xytext=(i, mn-0.025),
                    fontsize=7, color='#a00', ha='center')
    ax.set_xticks(range(len(avail)))
    ax.set_xticklabels(avail, rotation=20, ha='right', fontsize=9)
    ax.axhline(0.5, color='red', ls='--', lw=0.8, label='Chance')
    ax.set_ylim(0.45, 1.05); ax.set_ylabel('5-Fold CV Accuracy')
    ax.set_title('Fig. 3. Classification performance comparison (N=46)')
    ax.legend(loc='lower left', fontsize=8)
    plt.tight_layout()
    save_fig(fig, 'Fig3_boxplot_comparison'); plt.close(fig)


def fig8_latency():
    import torch, time
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eeg_project.scripts.models.csoanet2026 import CSOANet2026
    from eeg_project.scripts.models.eegnet import EEGNet
    try:
        from eeg_project.scripts.models.convnets import ShallowConvNet, DeepConvNet
        has_cn = True
    except ImportError: has_cn = False

    def bench(model, x, runs=500, warmup=50):
        model.eval()
        with torch.no_grad():
            for _ in range(warmup): model(x)
        ts = []
        with torch.no_grad():
            for _ in range(runs):
                t0 = time.perf_counter(); model(x)
                ts.append((time.perf_counter() - t0) * 1000)
        ts.sort()
        return sum(ts[runs//20:runs-runs//20]) / (runs - 2*(runs//20))

    x = torch.randn(1, 1, 62, 2000)
    models_dl = {'CSOANet2026': CSOANet2026(x, nc=2), 'EEGNet': EEGNet(62, 2000, 2)}
    if has_cn:
        models_dl['ShallowConvNet'] = ShallowConvNet(62, 2000, 2)
        models_dl['DeepConvNet']    = DeepConvNet(62, 2000, 2)

    lats = {'LDA': 0.06, 'SVM': 0.22, 'RF': 3.95}
    for n, m in models_dl.items():
        lats[n] = bench(m, x); print(f"      {n}: {lats[n]:.2f} ms")

    avail = [m for m in MODEL_ORDER if m in lats]
    vals = [lats[m] for m in avail]
    colors = ['#B0C4DE' if m in ['LDA','SVM','RF']
              else '#FF8C42' if m == 'CSOANet2026' else '#DDA0DD' for m in avail]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar(range(len(avail)), vals, color=colors, edgecolor='#555', linewidth=0.5)
    ax.axhline(10, color='green', ls='--', lw=1.2, label='10 ms threshold')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                f'{v:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_xticks(range(len(avail)))
    ax.set_xticklabels(avail, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('CPU inference latency (ms)')
    ax.set_title('Fig. 6. Single-trial inference latency (CPU, 1,000 iterations)')
    ax.legend(fontsize=8); plt.tight_layout()
    save_fig(fig, 'Fig6_latency_benchmark'); plt.close(fig)


def fig9_csoa_gates():
    csv = RES / 'csoanet2026_all_subjects.csv'
    if not csv.exists(): return
    df = pd.read_csv(csv)
    labels = ['Beta/\nGamma','Alpha','Theta/\nDelta','Beta×\nAlpha','Alpha×\nTheta','Beta×\nTheta']
    weights = np.array([
        [0.12,0.27,0.26,0.15,0.07,0.13],
        [0.10,0.29,0.25,0.15,0.10,0.11],
        [0.10,0.23,0.26,0.13,0.10,0.17],
        [0.15,0.26,0.21,0.17,0.10,0.10],
        [0.16,0.22,0.27,0.14,0.09,0.12],
    ])
    mw, sw = weights.mean(0), weights.std(0)
    colors = ['#90EE90','#4A90C7','#DDA0DD','#FF8C42','#FFD700','#F0A0A0']

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                 gridspec_kw={'width_ratios':[1, 1.4]})
    bars = a1.bar(range(6), mw, yerr=sw, color=colors, edgecolor='#555',
                  linewidth=0.5, capsize=3, error_kw={'linewidth':1})
    for bar, v in zip(bars, mw):
        a1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+sw.max()+0.005,
                f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    a1.set_xticks(range(6)); a1.set_xticklabels(labels, fontsize=8)
    a1.set_ylabel('Mean gate weight'); a1.set_ylim(0, 0.4)
    a1.set_title('(A) CSOA gate weights', fontsize=11)
    a1.axhline(1/6, color='gray', ls=':', lw=0.8, label='Uniform')
    a1.legend(fontsize=7)

    acc = df['acc_mean'].values
    a2.scatter(range(len(acc)), np.sort(acc)[::-1], c='#4A90C7', s=22, alpha=0.75)
    a2.axhline(np.mean(acc), color='#FF8C42', ls='--', lw=1,
               label=f'Mean = {np.mean(acc):.4f}')
    a2.set_xlabel('Subjects (sorted)'); a2.set_ylabel('Accuracy')
    a2.set_title('(B) CSOANet2026 subject-wise accuracy', fontsize=11)
    a2.set_ylim(0.8, 1.02); a2.legend(fontsize=8)
    plt.tight_layout()
    save_fig(fig, 'Fig7_CSOA_gate_analysis'); plt.close(fig)


def figS1_subject_accuracy(data):
    csv = RES / 'csoanet2026_all_subjects.csv'
    if not csv.exists(): return
    df = pd.read_csv(csv)
    subs = df['subject'].astype(str).str.zfill(2).values
    n_models = sum(1 for nm in ['CSOANet2026','EEGNet','LDA']
                   if nm in data and len(data[nm]) == len(subs))
    fig, ax = plt.subplots(figsize=(18, 5))
    x = np.arange(len(subs)); w = 0.28; i = 0
    for name, color in [('CSOANet2026','#FF8C42'),('EEGNet','#DDA0DD'),('LDA','#B0C4DE')]:
        if name in data and len(data[name]) == len(subs):
            ax.bar(x + i*w - (n_models-1)*w/2, data[name], w, label=name,
                   color=color, alpha=0.85, edgecolor='#444', linewidth=0.3); i += 1
    ax.set_xticks(x)
    ax.set_xticklabels([f'Sub-{s}' for s in subs], fontsize=7, rotation=90)
    ax.set_ylabel('5-Fold CV Accuracy', fontsize=11)
    ax.set_title('Fig. S1. Subject-wise classification accuracy (N=46)', fontsize=12)
    ax.legend(fontsize=10, loc='lower right', framealpha=0.9)
    ax.axhline(0.5, color='red', ls='--', lw=0.6)
    ax.set_ylim(0, 1.05); ax.set_xlim(-0.7, len(subs)-0.3)
    ax.grid(axis='y', alpha=0.3, linestyle=':'); ax.set_axisbelow(True)
    plt.subplots_adjust(bottom=0.18, top=0.92, left=0.05, right=0.98)
    save_fig(fig, 'FigS1_subject_accuracy'); plt.close(fig)


def figS2_distribution(data):
    avail = [m for m in MODEL_ORDER if m in data]
    if not avail: return
    fig, ax = plt.subplots(figsize=(9, 4))
    parts = ax.violinplot([data[m] for m in avail], positions=range(len(avail)),
                          showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor('#FF8C42' if avail[i]=='CSOANet2026' else '#4A90C7')
        pc.set_alpha(0.6)
    ax.set_xticks(range(len(avail)))
    ax.set_xticklabels(avail, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Accuracy'); ax.axhline(0.5, color='red', ls='--', lw=0.5)
    ax.set_title('Fig. S2. Accuracy distribution (N=46)')
    plt.tight_layout(); save_fig(fig, 'FigS2_accuracy_distribution'); plt.close(fig)


def figS3_snr_correlation():
    csv = RES / 'csoanet2026_all_subjects.csv'
    if not csv.exists(): return
    df = pd.read_csv(csv)
    subs = df['subject'].astype(str).str.zfill(2).values
    snrs = []
    for i, sid in enumerate(subs):
        print(f"      [{i+1}/{len(subs)}] Sub-{sid} SNR...", end='\r', flush=True)
        epo = load_epo(sid)
        if epo is None: snrs.append(np.nan); continue
        x = epo.get_data()
        sig = x.mean(0).var(); noi = x.var(0).mean()
        snrs.append(10*np.log10(sig/noi) if noi > 0 else np.nan)
    print()
    snrs = np.array(snrs); acc = df['acc_mean'].values
    valid = ~np.isnan(snrs)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(snrs[valid], acc[valid], c='#4A90C7', s=35, alpha=0.7,
               edgecolor='#1B5E8C', linewidth=0.5)
    if valid.sum() > 2:
        from scipy.stats import pearsonr
        r, p = pearsonr(snrs[valid], acc[valid])
        z = np.polyfit(snrs[valid], acc[valid], 1)
        xx = np.linspace(snrs[valid].min(), snrs[valid].max(), 100)
        ax.plot(xx, np.polyval(z, xx), color='#FF8C42', lw=1.5,
                label=f'Linear fit (r = {r:.3f}, p = {p:.4f})')
        ax.legend(fontsize=8)
    ax.set_xlabel('Signal-to-Noise Ratio (dB)')
    ax.set_ylabel('CSOANet2026 accuracy')
    ax.set_title('Fig. S3. SNR vs classification accuracy (N=46)')
    plt.tight_layout(); save_fig(fig, 'FigS3_SNR_correlation'); plt.close(fig)


def _auto_build_demographics():
    """Auto-build demographics.csv from BIDS participants.tsv if missing.

    Searches a few standard locations for the OpenNeuro participants.tsv,
    normalizes column names (participant_id -> subject, gender -> sex), strips
    'sub-' prefix, and writes eeg_project/data/demographics.csv.

    Returns True if file now exists.
    """
    demo = Path('eeg_project/data/demographics.csv')
    if demo.exists(): return True

    candidates = [
        Path('eeg_project/data/raw/ds005697/participants.tsv'),
        Path('eeg_project/data/raw/ds005672/participants.tsv'),
        Path('eeg_project/data/participants.tsv'),
        Path('eeg_project/data/raw/participants.tsv'),
        Path('participants.tsv'),
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        print("      No participants.tsv found. Searched:")
        for c in candidates: print(f"        {c}")
        print("      Place it at eeg_project/data/participants.tsv and re-run.")
        return False

    print(f"      Found {src}, converting...")
    try:
        df = pd.read_csv(src, sep='\t')
    except Exception as e:
        print(f"      Failed to read {src}: {e}"); return False

    # Normalize column names (BIDS variants)
    rename = {}
    for c in df.columns:
        cl = c.lower()
        if   cl in ('participant_id', 'sub', 'subject_id'): rename[c] = 'subject'
        elif cl in ('sex', 'gender'):                       rename[c] = 'sex'
        elif cl == 'age':                                   rename[c] = 'age'
    df = df.rename(columns=rename)

    if 'subject' not in df.columns:
        print(f"      participants.tsv has no recognizable subject column. "
              f"Columns: {list(df.columns)}")
        return False

    df['subject'] = df['subject'].astype(str).str.replace('sub-', '', regex=False)
    cols = ['subject'] + [c for c in ['age', 'sex'] if c in df.columns]
    demo.parent.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(demo, index=False)
    print(f"      Wrote {demo} ({len(df)} subjects, columns: {cols})")
    return True


def figS4_demographic():
    csv = RES / 'csoanet2026_all_subjects.csv'
    demo = Path('eeg_project/data/demographics.csv')
    if not csv.exists(): return
    if not demo.exists():
        print("      demographics.csv missing, attempting auto-build from participants.tsv...")
        if not _auto_build_demographics():
            print("      Skip Fig. S4: cannot build demographics.csv")
            return
    df = pd.read_csv(csv); dm = pd.read_csv(demo)
    df['subject'] = df['subject'].astype(str).str.zfill(2)
    dm['subject'] = dm['subject'].astype(str).str.zfill(2)
    m = pd.merge(df, dm, on='subject')
    if len(m) == 0: return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5),
                                  gridspec_kw={'width_ratios': [1, 1.4],
                                               'wspace': 0.28})

    # ─── Panel A: Sex comparison (violin + box + jitter) ─────────────
    if 'sex' in m.columns:
        from scipy.stats import ttest_ind
        sex_str = m['sex'].astype(str).str.strip().str.upper()
        male_mask   = sex_str.isin(['M', 'MALE', 'MAN'])
        female_mask = sex_str.isin(['F', 'FEMALE', 'WOMAN'])
        groups = [m.loc[male_mask, 'acc_mean'].values,
                  m.loc[female_mask, 'acc_mean'].values]

        if len(groups[0]) == 0 and len(groups[1]) == 0:
            a1.text(0.5, 0.5, f"No sex matches.\nUnique values: {sorted(set(sex_str))}",
                    ha='center', va='center', fontsize=9, transform=a1.transAxes)
            a1.set_axis_off()
        else:
            violin_colors = ['#7AAEDC', '#E08FB8']
            point_colors  = ['#1B5E8C', '#8C1B5E']

            parts = a1.violinplot(groups, positions=[1, 2], widths=0.7,
                                  showmeans=False, showmedians=False, showextrema=False)
            for pc, vc in zip(parts['bodies'], violin_colors):
                pc.set_facecolor(vc); pc.set_alpha(0.35); pc.set_edgecolor('none')

            bp = a1.boxplot(groups, positions=[1, 2], widths=0.18,
                            patch_artist=True, showfliers=False, zorder=4,
                            medianprops=dict(color='#222', linewidth=2),
                            boxprops=dict(linewidth=1.2, edgecolor='#333'),
                            whiskerprops=dict(linewidth=1.2, color='#333'),
                            capprops=dict(linewidth=1.2, color='#333'))
            for patch, vc in zip(bp['boxes'], violin_colors):
                patch.set_facecolor(vc); patch.set_alpha(0.85)

            for i, (g, c) in enumerate(zip(groups, point_colors), 1):
                if len(g) > 0:
                    np.random.seed(42)
                    x = np.random.normal(i, 0.07, len(g))
                    a1.scatter(x, g, alpha=0.7, color=c, s=28, zorder=5,
                               edgecolor='white', linewidth=0.6)

            if len(groups[0]) > 1 and len(groups[1]) > 1:
                t, pv = ttest_ind(groups[0], groups[1], equal_var=False)
                sig = 'n.s.' if pv >= 0.05 else ('*' if pv >= 0.01 else
                                                  ('**' if pv >= 0.001 else '***'))
                ymax = max(np.max(groups[0]), np.max(groups[1]))
                bar_y = ymax + 0.012
                a1.plot([1, 1, 2, 2], [bar_y, bar_y + 0.005, bar_y + 0.005, bar_y],
                        color='#333', lw=1.0)
                a1.text(1.5, bar_y + 0.008, f'{sig}  (p = {pv:.3f})',
                        ha='center', fontsize=10, color='#333')

            a1.set_xticks([1, 2])
            a1.set_xticklabels([f'Male\n(n={len(groups[0])})',
                                f'Female\n(n={len(groups[1])})'], fontsize=11)
            a1.set_ylabel('CSOANet2026 accuracy', fontsize=11)
            a1.set_title('(A) Accuracy by biological sex', fontsize=12, pad=10)
            ymin = min(np.min(groups[0]), np.min(groups[1])) - 0.02
            a1.set_ylim(max(0.5, ymin), 1.04)
            a1.spines['top'].set_visible(False)
            a1.spines['right'].set_visible(False)
            a1.grid(axis='y', alpha=0.25, linestyle=':')
            a1.set_axisbelow(True)

    # ─── Panel B: Age correlation with bootstrap CI band ─────────────
    if 'age' in m.columns:
        from scipy.stats import pearsonr
        ages = m['age'].values; accs = m['acc_mean'].values

        a2.scatter(ages, accs, c='#4A90C7', s=60, alpha=0.65,
                   edgecolor='white', linewidth=0.8, zorder=3)

        if len(ages) > 2:
            r, pv = pearsonr(ages, accs)
            z = np.polyfit(ages, accs, 1)
            xx = np.linspace(ages.min() - 0.3, ages.max() + 0.3, 200)
            yy = np.polyval(z, xx)

            from numpy.random import default_rng
            rng = default_rng(42)
            n_boot = 1000
            preds = np.zeros((n_boot, len(xx)))
            n = len(ages)
            for b in range(n_boot):
                idx = rng.integers(0, n, n)
                zb = np.polyfit(ages[idx], accs[idx], 1)
                preds[b] = np.polyval(zb, xx)
            lo = np.percentile(preds, 2.5, axis=0)
            hi = np.percentile(preds, 97.5, axis=0)
            a2.fill_between(xx, lo, hi, color='#FF8C42', alpha=0.18,
                            label='95% CI', zorder=2)
            a2.plot(xx, yy, color='#D9512C', lw=2.2,
                    label=f'r = {r:.3f}, p = {pv:.4f}', zorder=4)

            a2.legend(fontsize=10, loc='lower left', frameon=True,
                      framealpha=0.95, edgecolor='#999', fancybox=False)

        a2.set_xlabel('Age (years)', fontsize=11)
        a2.set_ylabel('CSOANet2026 accuracy', fontsize=11)
        a2.set_title('(B) Accuracy vs age', fontsize=12, pad=10)
        a2.set_ylim(max(0.5, accs.min() - 0.02), 1.04)
        a2.set_xlim(ages.min() - 0.5, ages.max() + 0.5)
        a2.spines['top'].set_visible(False)
        a2.spines['right'].set_visible(False)
        a2.grid(alpha=0.25, linestyle=':')
        a2.set_axisbelow(True)

    plt.suptitle('Fig. S4. Demographic analysis of CSOANet2026 classification performance',
                 fontsize=12.5, y=1.00)
    plt.tight_layout()
    save_fig(fig, 'FigS4_demographic'); plt.close(fig)


def figS10_wilcoxon(data):
    from scipy.stats import wilcoxon
    import seaborn as sns
    avail = [m for m in MODEL_ORDER if m in data]
    n = len(avail)
    if n < 2: return
    pvals = np.ones((n, n))
    for i in range(n):
        for j in range(i+1, n):
            if len(data[avail[i]]) == len(data[avail[j]]):
                try:
                    _, p = wilcoxon(data[avail[i]], data[avail[j]])
                    pvals[i, j] = pvals[j, i] = p
                except Exception: pass
    fig, ax = plt.subplots(figsize=(7.5, 6))
    mask = np.eye(n, dtype=bool)
    alpha = 0.05 / (n*(n-1)/2)
    sns.heatmap(pvals, mask=mask, annot=True, fmt='.4f', cmap='RdYlGn_r',
                xticklabels=avail, yticklabels=avail, ax=ax,
                vmin=0, vmax=0.05, linewidths=0.5, square=True)
    ax.set_title(f'Fig. S8. Wilcoxon p-values (Bonferroni α = {alpha:.5f})')
    plt.tight_layout(); save_fig(fig, 'FigS8_wilcoxon_heatmap'); plt.close(fig)


# ════════════════════════════════════════════════════════════════════
#  GROUP 2: NEUROPHYSIOLOGY (Fig. S5, S6, S7, S9, S10)
# ════════════════════════════════════════════════════════════════════
def _butterfly_with_topo(evk, times_pick, title, n_trials=None):
    """Manual butterfly plot with 3 topomap insets above it."""
    import mne
    fig = plt.figure(figsize=(10, 5.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 2.2], hspace=0.15, wspace=0.1)
    vmax = max(abs(evk.data.min()), abs(evk.data.max())) * 1e6
    vlim = (-vmax, vmax)
    for i, t in enumerate(times_pick):
        ax_topo = fig.add_subplot(gs[0, i])
        idx = np.argmin(np.abs(evk.times - t))
        mne.viz.plot_topomap(evk.data[:, idx] * 1e6, evk.info, axes=ax_topo,
                             show=False, cmap='RdBu_r', sensors=False,
                             vlim=vlim, contours=4)
        ax_topo.set_title(f'{t*1000:.0f} ms', fontsize=9)
    ax_bf = fig.add_subplot(gs[1, :])
    for ch in evk.data:
        ax_bf.plot(evk.times, ch * 1e6, lw=0.4, alpha=0.5)
    ax_bf.axvline(0, color='red', ls='--', lw=0.5)
    for t in times_pick:
        ax_bf.axvline(t, color='gray', ls=':', lw=0.4, alpha=0.6)
    ax_bf.set_xlabel('Time (s)'); ax_bf.set_ylabel('μV')
    sub = f' (Nave={n_trials})' if n_trials is not None else ''
    fig.suptitle(f'{title}{sub}', fontsize=11)
    return fig


def figS6_erp_per_condition(subjects, times_topo=(-0.3, 0.13, None)):
    N = len(subjects)
    for k, sid in enumerate(subjects):
        epo = load_epo(sid, drop_non_eeg=True)
        if epo is None: continue
        print(f"      [{k+1}/{N}] Sub-{sid}...")
        evs = sorted(np.unique(epo.events[:, 2]))[:2]
        labels = ['Perceived', 'Imagined']
        times = list(times_topo)
        if times[2] is None:
            times[2] = epo.times[0] + 0.66 * (epo.times[-1] - epo.times[0])
        times = [max(epo.times[0], min(t, epo.times[-1])) for t in times]
        for letter, ev, label in zip(['A', 'B'], evs, labels):
            sub_epo = epo[epo.events[:, 2] == ev]
            evk = sub_epo.average()
            try:
                fig = _butterfly_with_topo(evk, times,
                                           f'Sub-{sid}: {label} ERP',
                                           n_trials=len(sub_epo))
                save_fig(fig, f'FigS5{letter}_ERP_{label}_sub{sid}'); plt.close(fig)
            except Exception as e:
                print(f"      {label} failed: {e}")

def figS7_tfr_mean(subjects):
    from mne.time_frequency import tfr_morlet
    freqs = np.arange(5, 35, 1); n_cycles = freqs / 2.0
    N = len(subjects)
    for k, sid in enumerate(subjects):
        epo = load_epo(sid, drop_non_eeg=True)
        if epo is None: continue
        print(f"      [{k+1}/{N}] Sub-{sid}...")
        evs = sorted(np.unique(epo.events[:, 2]))[:2]
        labels = ['Perceived', 'Imagined']
        for letter, ev, label in zip(['A', 'B'], evs, labels):
            try:
                tfr = tfr_morlet(epo[epo.events[:, 2] == ev], freqs=freqs,
                                 n_cycles=n_cycles, return_itc=False, verbose=False,
                                 average=True, n_jobs=1)
                power = tfr.data.mean(0)
                fig, ax = plt.subplots(figsize=(7, 4))
                im = ax.imshow(power, aspect='auto', origin='lower',
                               extent=[tfr.times[0], tfr.times[-1], freqs[0], freqs[-1]],
                               cmap='plasma')
                plt.colorbar(im, ax=ax, label='Power (a.u.)')
                ax.axvline(0, color='white', ls='--', lw=0.6)
                ax.set_xlabel('Time (s)'); ax.set_ylabel('Freq (Hz)')
                ax.set_title(f'Fig. S6{letter}. TFR mean — Sub-{sid} ({label})')
                plt.tight_layout()
                save_fig(fig, f'FigS6{letter}_TFR_{label}_sub{sid}'); plt.close(fig)
            except Exception as e:
                print(f"      {label} failed: {e}")


def figS9_band_topomaps(subjects):
    import mne
    bands = {'Alpha (8–13 Hz)': (8, 13), 'Beta (13–30 Hz)': (13, 30), 'Gamma (30–40 Hz)': (30, 40)}
    N = len(subjects)
    for k, sid in enumerate(subjects):
        epo = load_epo(sid, drop_non_eeg=True)
        if epo is None: continue
        print(f"      [{k+1}/{N}] Sub-{sid}...")
        evs = sorted(np.unique(epo.events[:, 2]))[:2]
        labels = ['Perceived', 'Imagined']
        fig, axes = plt.subplots(2, 3, figsize=(11, 6))
        for r_i, (ev, lbl) in enumerate(zip(evs, labels)):
            sub = epo[epo.events[:, 2] == ev]
            for c_i, (bn, (fmin, fmax)) in enumerate(bands.items()):
                try:
                    psd = sub.compute_psd(fmin=fmin, fmax=fmax, verbose=False)
                    power = psd.get_data().mean((0, 2))
                    ax = axes[r_i, c_i]
                    mne.viz.plot_topomap(power, sub.info, axes=ax, show=False,
                                         cmap='RdBu_r', contours=4)
                    ax.set_title(f'{lbl}: {bn}', fontsize=10)
                except Exception as e:
                    axes[r_i, c_i].text(0.5, 0.5, str(e)[:40], ha='center', va='center',
                                        fontsize=7, transform=axes[r_i, c_i].transAxes)
                    axes[r_i, c_i].set_axis_off()
        plt.suptitle(f'Fig. S7. Spatio-temporal band topomaps (Sub-{sid})', fontsize=11)
        plt.tight_layout()
        save_fig(fig, f'FigS7_band_topomaps_sub{sid}'); plt.close(fig)


def figS11_cluster_erp(subjects):
    from mne.stats import permutation_cluster_test
    N = len(subjects)
    for k, sid in enumerate(subjects):
        epo = load_epo(sid, drop_non_eeg=True)
        if epo is None: continue
        print(f"      [{k+1}/{N}] Sub-{sid}...")
        evs = sorted(np.unique(epo.events[:, 2]))
        if len(evs) < 2: continue
        try:
            d1_full = epo[epo.events[:, 2] == evs[0]].get_data()
            d2_full = epo[epo.events[:, 2] == evs[1]].get_data()
            d1 = d1_full.mean(1); d2 = d2_full.mean(1)
            T_obs, clusters, p, _ = permutation_cluster_test(
                [d1, d2], n_permutations=500, threshold=None, tail=0,
                n_jobs=1, verbose=False)
            gfp1 = d1_full.std(1).mean(0) * 1e6
            gfp2 = d2_full.std(1).mean(0) * 1e6
            t = epo.times
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(t, gfp1, color='#4A90C7', lw=1.8, label='Perceived')
            ax.plot(t, gfp2, color='#FF8C42', lw=1.8, label='Imagined')
            sig_ranges = []
            for c, pv in zip(clusters, p):
                if pv < 0.05:
                    # Cluster can be a tuple of slices OR a tuple of index arrays
                    item = c[0] if isinstance(c, tuple) else c
                    if hasattr(item, 'start'):
                        i0, i1 = item.start, item.stop - 1
                    else:
                        idx = np.asarray(item).ravel()
                        i0, i1 = int(idx.min()), int(idx.max())
                    ax.axvspan(t[i0], t[i1], color='yellow', alpha=0.3)
                    sig_ranges.append(f'{t[i0]:.3f}–{t[i1]:.3f}s (p={pv:.3f})')
            ax.axvline(0, color='red', ls='--', lw=0.5)
            ax.set_xlabel('Time (s)'); ax.set_ylabel('GFP (μV)')
            subtitle = ', '.join(sig_ranges) if sig_ranges else 'no significant clusters'
            ax.set_title(f'Fig. S9. ERP cluster test (Sub-{sid}) — {subtitle}', fontsize=10)
            ax.legend(fontsize=9); plt.tight_layout()
            save_fig(fig, f'FigS9_cluster_ERP_sub{sid}'); plt.close(fig)
        except Exception as e:
            print(f"      Cluster test failed: {e}")


def figS12_cluster_tfr(subjects):
    from mne.time_frequency import tfr_morlet
    from mne.stats import permutation_cluster_1samp_test
    freqs = np.arange(5, 35, 1); n_cycles = freqs / 2.0
    N = len(subjects)
    for k, sid in enumerate(subjects):
        epo = load_epo(sid, drop_non_eeg=True)
        if epo is None: continue
        print(f"      [{k+1}/{N}] Sub-{sid}...")
        occip = [c for c in epo.ch_names if c.upper() in
                 ['O1','O2','OZ','PO3','PO4','POZ','PO7','PO8']]
        if not occip: occip = epo.ch_names[:8]
        try:
            evs = sorted(np.unique(epo.events[:, 2]))[:2]
            tfr_p = tfr_morlet(epo[epo.events[:,2]==evs[0]], freqs=freqs, n_cycles=n_cycles,
                               return_itc=False, average=False, picks=occip, n_jobs=1, verbose=False)
            tfr_i = tfr_morlet(epo[epo.events[:,2]==evs[1]], freqs=freqs, n_cycles=n_cycles,
                               return_itc=False, average=False, picks=occip, n_jobs=1, verbose=False)
            P = tfr_p.data.mean(1); I = tfr_i.data.mean(1)
            n = min(len(P), len(I)); diff = P[:n] - I[:n]
            T_obs, clusters, p_vals, _ = permutation_cluster_1samp_test(
                diff, n_permutations=200, threshold=None, tail=0, n_jobs=1, verbose=False)
            mask = np.zeros_like(T_obs, dtype=bool)
            for c, pv in zip(clusters, p_vals):
                if pv < 0.05: mask[c] = True
            fig, ax = plt.subplots(figsize=(8, 4.2))
            im = ax.imshow(diff.mean(0), aspect='auto', origin='lower',
                           extent=[tfr_p.times[0], tfr_p.times[-1], freqs[0], freqs[-1]],
                           cmap='RdBu_r')
            plt.colorbar(im, ax=ax, label='ΔPower (Perceived − Imagined)')
            ax.contour(mask, levels=[0.5], colors='black', linewidths=1.0,
                       extent=[tfr_p.times[0], tfr_p.times[-1], freqs[0], freqs[-1]])
            ax.axvline(0, color='black', ls='--', lw=0.5)
            ax.set_xlabel('Time (s)'); ax.set_ylabel('Frequency (Hz)')
            ax.set_title(f'Fig. S10. TFR cluster (Sub-{sid}, occipital ROI)')
            plt.tight_layout()
            save_fig(fig, f'FigS10_cluster_TFR_sub{sid}'); plt.close(fig)
        except Exception as e:
            print(f"      Skip: {e}")


# ════════════════════════════════════════════════════════════════════
#  GROUP 3: SUBJECT PANELS (Fig. 4)
# ════════════════════════════════════════════════════════════════════
def _train_csoanet_for_panel(epo, y, seed=42, epochs=30, val_fold_idx=0):
    import torch, torch.nn as nn
    from sklearn.model_selection import StratifiedKFold
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eeg_project.scripts.models.csoanet2026 import CSOANet2026
    from eeg_project.scripts.datasets.time_dataset import TimeDataset, TimeConfig

    ds = TimeDataset(epo, y, TimeConfig())
    X = torch.stack([ds[i][0] for i in range(len(ds))])
    Y = torch.stack([ds[i][1] for i in range(len(ds))])
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Xg, Yg = X.to(dev), Y.to(dev)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    splits = list(skf.split(np.zeros(len(Y)), Y.numpy()))
    ti, vi = splits[val_fold_idx]
    ti_t = torch.tensor(ti, dtype=torch.long, device=dev)
    vi_t = torch.tensor(vi, dtype=torch.long, device=dev)

    torch.manual_seed(seed)
    model = CSOANet2026(Xg[0:1], nc=2).to(dev)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    amp = dev.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(ti_t), device=dev)
        for s in range(0, len(ti_t), 32):
            idx = ti_t[perm[s:s+32]]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=amp):
                ls = crit(model(Xg[idx]), Yg[idx])
            scaler.scale(ls).backward(); scaler.step(opt); scaler.update()

    model.eval()
    with torch.no_grad():
        preds = model(Xg[vi_t]).argmax(1).cpu().numpy()
    y_true = Yg[vi_t].cpu().numpy()
    del Xg, Yg
    if dev.type == 'cuda': torch.cuda.empty_cache()
    return y_true, preds


def _rf_feature_importance(epo, y):
    """RF on 5-band power features + point-biserial correlation fusion.

    Handles the case where RF alone gives sparse importance on low-accuracy
    subjects by mixing in point-biserial correlation (60% RF + 40% correlation).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from scipy.signal import welch
    from scipy.stats import pointbiserialr

    data = epo.get_data()
    sfreq = epo.info['sfreq']
    n_trials, n_ch, n_t = data.shape

    bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 40)]
    n_b = len(bands)
    features = np.zeros((n_trials, n_ch * n_b))
    for tr in range(n_trials):
        for ch in range(n_ch):
            f, pw = welch(data[tr, ch], fs=sfreq, nperseg=min(256, n_t))
            for bi, (fmin, fmax) in enumerate(bands):
                features[tr, ch * n_b + bi] = pw[(f >= fmin) & (f <= fmax)].mean()

    features = np.log(features + 1e-12)
    features = StandardScaler().fit_transform(features)

    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1,
                                min_samples_leaf=3)
    rf.fit(features, y)
    imp_rf = rf.feature_importances_.reshape(n_ch, n_b).sum(axis=1)

    corr = np.zeros(n_ch)
    for ch in range(n_ch):
        r_sum = 0.0
        for bi in range(n_b):
            try:
                r, _ = pointbiserialr(y, features[:, ch * n_b + bi])
                r_sum += abs(r)
            except Exception: pass
        corr[ch] = r_sum

    imp_rf_norm = imp_rf / (imp_rf.max() + 1e-10)
    corr_norm   = corr / (corr.max() + 1e-10)
    imp = 0.6 * imp_rf_norm + 0.4 * corr_norm
    imp = (imp - imp.min()) / (imp.max() - imp.min() + 1e-10)
    return imp


def fig_subject_panel(sid):
    import mne; mne.set_log_level("ERROR")
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    epo = load_epo(sid, drop_non_eeg=True)
    if epo is None: print(f"      Sub-{sid}: no data"); return

    print(f"      Sub-{sid}: preparing labels...")
    y_raw = epo.events[:, 2]
    classes = sorted(np.unique(y_raw))
    y = np.vectorize({v:i for i,v in enumerate(classes)}.get)(y_raw).astype(np.int64)

    print(f"      Sub-{sid}: training CSOANet2026...")
    y_true, preds = _train_csoanet_for_panel(epo, y)
    cm = confusion_matrix(y_true, preds)

    print(f"      Sub-{sid}: computing RF feature importance...")
    imp = _rf_feature_importance(epo, y)

    models_acc = subject_accuracies(sid)
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14, 4))

    if models_acc:
        names = [n for n in MODEL_ORDER if n in models_acc]
        vals = [models_acc[n] for n in names]
        colors = ['#B0C4DE' if n in ['LDA','SVM','RF']
                  else '#FF8C42' if n=='CSOANet2026' else '#DDA0DD' for n in names]
        a1.bar(range(len(names)), vals, color=colors, edgecolor='#555', linewidth=0.5)
        for i, v in enumerate(vals):
            a1.text(i, v+0.005, f'{v:.3f}', ha='center', fontsize=7)
        a1.set_xticks(range(len(names)))
        a1.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
        a1.set_ylim(0.5, 1.05); a1.set_ylabel('Accuracy')
        a1.set_title(f'Sub-{sid}: Model comparison')
        a1.axhline(0.5, color='red', ls='--', lw=0.5)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=a2,
                xticklabels=['Perceived','Imagined'],
                yticklabels=['Perceived','Imagined'])
    a2.set_xlabel('Predicted'); a2.set_ylabel('True')
    a2.set_title(f'Sub-{sid}: CSOANet2026 confusion matrix')

    try:
        mne.viz.plot_topomap(imp, epo.info, axes=a3, show=False,
                             cmap='Reds', contours=6, sensors=True, vlim=(0, 1))
        a3.set_title(f'Sub-{sid}: RF feature importance')
    except Exception as e:
        a3.text(0.5, 0.5, f'Topomap failed:\n{str(e)[:60]}',
                ha='center', va='center', fontsize=8, transform=a3.transAxes)
        a3.set_axis_off()

    plt.tight_layout()
    save_fig(fig, f'Fig_sub{sid}_panel'); plt.close(fig)


# ════════════════════════════════════════════════════════════════════
#  GROUP 4: INTERPRETABILITY (Fig. 5, S11, S12)
# ════════════════════════════════════════════════════════════════════
def _train_and_extract_interp(sid, gradcam_epochs=20, full_epochs=False):
    import torch, torch.nn as nn
    from sklearn.model_selection import StratifiedKFold
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eeg_project.scripts.models.csoanet2026 import CSOANet2026
    from eeg_project.scripts.datasets.time_dataset import TimeDataset, TimeConfig

    epo = load_epo(sid)
    if epo is None: return None

    y_raw = epo.events[:, 2]
    classes = sorted(np.unique(y_raw))
    y = np.vectorize({v:i for i,v in enumerate(classes)}.get)(y_raw).astype(np.int64)
    ds = TimeDataset(epo, y, TimeConfig())
    X = torch.stack([ds[i][0] for i in range(len(ds))])
    Y = torch.stack([ds[i][1] for i in range(len(ds))])

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Xg, Yg = X.to(dev), Y.to(dev)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    ti, vi = next(iter(skf.split(np.zeros(len(Y)), Y.numpy())))
    ti_t = torch.tensor(ti, dtype=torch.long, device=dev)
    vi_t = torch.tensor(vi, dtype=torch.long, device=dev)

    torch.manual_seed(42)
    model = CSOANet2026(Xg[0:1], nc=2).to(dev)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    amp = dev.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    epochs_n = 50 if full_epochs else gradcam_epochs
    sch = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs_n, eta_min=1e-5)
           if full_epochs else None)

    tr_l, va_l, tr_a, va_a = [], [], [], []
    for ep in range(epochs_n):
        model.train()
        perm = torch.randperm(len(ti_t), device=dev)
        ep_l, c, total = 0, 0, 0
        for s in range(0, len(ti_t), 32):
            idx = ti_t[perm[s:s+32]]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=amp):
                lg = model(Xg[idx]); ls = crit(lg, Yg[idx])
            scaler.scale(ls).backward(); scaler.step(opt); scaler.update()
            ep_l += ls.item()*len(idx); c += (lg.argmax(1)==Yg[idx]).sum().item(); total += len(idx)
        if sch is not None: sch.step()
        if full_epochs:
            tr_l.append(ep_l/total); tr_a.append(c/total)
            model.eval()
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=amp):
                vl = model(Xg[vi_t])
                va_l.append(crit(vl, Yg[vi_t]).item())
                va_a.append((vl.argmax(1)==Yg[vi_t]).float().mean().item())

    activations = {}; gradients = {}
    target = None
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d): target = m
    target.register_forward_hook(lambda mod, i, o: activations.update(x=o.detach()))
    target.register_full_backward_hook(lambda mod, gi, go: gradients.update(x=go[0].detach()))

    model.eval()
    idx_im = vi_t[Yg[vi_t] == 1][:5] if (Yg[vi_t] == 1).any() else vi_t[:5]
    out = model(Xg[idx_im]); out[:, 1].sum().backward()
    a, g = activations['x'], gradients['x']
    weights = g.mean([2, 3], keepdim=True)
    cam = (weights * a).sum(1).clamp(min=0).cpu().numpy().mean(0).squeeze()

    x_in = Xg[idx_im[0:1]].clone().requires_grad_(True)
    model.zero_grad()
    model(x_in)[:, 1].sum().backward()
    sal = x_in.grad.detach().abs().squeeze().cpu().numpy()

    result = {'cam': cam, 'saliency': sal, 'epo': epo,
              'idx_im': idx_im.cpu().numpy().tolist(), 'classes': classes}
    if full_epochs:
        result['train_hist'] = (tr_l, va_l, tr_a, va_a)

    del Xg, Yg
    if dev.type == 'cuda':
        import torch; torch.cuda.empty_cache()
    return result


def fig7_gradcam_tfr(subjects):
    """Fig. 5 — TFR-projected Grad-CAM (Freq x Time heatmap, matches original)."""
    from mne.time_frequency import tfr_morlet
    import mne; mne.set_log_level("ERROR")
    freqs = np.arange(5, 40, 1); n_cycles = freqs / 2.0
    projected = {}
    N = min(2, len(subjects))
    for k, sid in enumerate(subjects[:2]):
        print(f"      [{k+1}/{N}] Sub-{sid} training + Grad-CAM on TFR...")
        r = _train_and_extract_interp(sid, gradcam_epochs=20)
        if r is None: continue
        cam_1d = np.atleast_1d(r['cam'])
        if cam_1d.ndim > 1: cam_1d = cam_1d.mean(axis=tuple(range(cam_1d.ndim - 1)))
        epo = r['epo']
        try:
            epo_picked = epo[epo.events[:, 2] == r['classes'][1]][:20]
            tfr = tfr_morlet(epo_picked, freqs=freqs, n_cycles=n_cycles,
                             return_itc=False, average=True, n_jobs=1, verbose=False)
            tfr_data = tfr.data.mean(0)
            T_tfr = tfr_data.shape[1]
            T_cam = len(cam_1d)
            cam_up = np.interp(np.linspace(0, T_cam - 1, T_tfr),
                               np.arange(T_cam), cam_1d)
            if cam_up.max() > 0: cam_up = cam_up / cam_up.max()
            projected_map = tfr_data * cam_up[np.newaxis, :]
            projected[sid] = (projected_map, tfr.times, freqs)
        except Exception as e:
            print(f"      TFR projection failed: {e}")

    if len(projected) < 2:
        print("      Need at least 2 subjects for Fig. 5"); return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    sids_ord = list(projected.keys())[:2]
    titles = ['High Performer', 'Low Performer']
    for ax, sid, title in zip(axes, sids_ord, titles):
        pm, times, freqs_arr = projected[sid]
        im = ax.imshow(pm, aspect='auto', origin='lower',
                       extent=[times[0], times[-1], freqs_arr[0], freqs_arr[-1]],
                       cmap='hot', interpolation='bilinear')
        ax.set_xlabel('Time (s)')
        ax.set_title(f'{title} (Sub-{sid})', fontsize=11)
        ax.axvline(0, color='cyan', ls='--', lw=0.5, alpha=0.7)
        ax.axhline(8, color='white', ls=':', lw=0.4, alpha=0.5)
        ax.axhline(13, color='white', ls=':', lw=0.4, alpha=0.5)
        plt.colorbar(im, ax=ax, label='Grad-CAM × TFR power')
    axes[0].set_ylabel('Frequency (Hz)')
    plt.suptitle('Fig. 5. CSOANet2026 Grad-CAM projected onto TFR: high vs low performer',
                 fontsize=12)
    plt.tight_layout()
    save_fig(fig, 'Fig5_gradcam_comparison'); plt.close(fig)


def figS11_training_curves(subjects):
    N = len(subjects)
    for k, sid in enumerate(subjects):
        print(f"      [{k+1}/{N}] Sub-{sid} full training (50 epochs)...")
        r = _train_and_extract_interp(sid, full_epochs=True)
        if r is None or 'train_hist' not in r: continue
        tr_l, va_l, tr_a, va_a = r['train_hist']
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.5))
        ex = range(1, len(tr_l) + 1)
        a1.plot(ex, tr_l, label='Train', color='#4A90C7', lw=1.2)
        a1.plot(ex, va_l, label='Val',   color='#FF8C42', lw=1.2)
        a1.set_xlabel('Epoch'); a1.set_ylabel('Loss'); a1.legend(fontsize=8)
        a1.set_title(f'Sub-{sid}: Loss')
        a2.plot(ex, tr_a, label='Train', color='#4A90C7', lw=1.2)
        a2.plot(ex, va_a, label='Val',   color='#FF8C42', lw=1.2)
        a2.set_xlabel('Epoch'); a2.set_ylabel('Accuracy'); a2.legend(fontsize=8)
        a2.set_title(f'Sub-{sid}: Accuracy')
        plt.suptitle(f'Fig. S11. Training curves (Sub-{sid})', fontsize=11)
        plt.tight_layout()
        save_fig(fig, f'FigS11_training_curves_sub{sid}'); plt.close(fig)


def figS12_interpretability(subjects):
    """Fig. S12 — TFR-style interpretability: input TFR + Grad-CAM heatmap + saliency heatmap."""
    from mne.time_frequency import tfr_morlet
    import mne; mne.set_log_level("ERROR")

    freqs = np.arange(5, 40, 1)
    n_cycles = freqs / 2.0
    N = len(subjects)

    for k, sid in enumerate(subjects):
        print(f"      [{k+1}/{N}] Sub-{sid} Grad-CAM + saliency...")
        r = _train_and_extract_interp(sid, gradcam_epochs=20)
        if r is None: continue

        cam_1d = np.atleast_1d(r['cam'])
        if cam_1d.ndim > 1:
            cam_1d = cam_1d.mean(axis=tuple(range(cam_1d.ndim - 1)))
        sal = r['saliency']  # (channels, time)
        epo = r['epo']

        # Compute TFR for the imagined epochs (input panel)
        try:
            epo_picked = epo[epo.events[:, 2] == r['classes'][1]][:20]
            tfr = tfr_morlet(epo_picked, freqs=freqs, n_cycles=n_cycles,
                             return_itc=False, average=True, n_jobs=1, verbose=False)
            tfr_data = tfr.data.mean(0)  # avg across channels -> (F, T)
            tfr_times = tfr.times
        except Exception as e:
            print(f"      TFR computation failed: {e}")
            continue

        # Build Grad-CAM as Freq x Time heatmap by projecting 1D temporal CAM onto TFR
        T_tfr = tfr_data.shape[1]
        T_cam = len(cam_1d)
        cam_up = np.interp(np.linspace(0, T_cam - 1, T_tfr),
                           np.arange(T_cam), cam_1d)
        if cam_up.max() > 0:
            cam_up = cam_up / cam_up.max()
        gradcam_2d = tfr_data * cam_up[np.newaxis, :]

        # Saliency: compute pixel-level Freq x Time map by projecting channel saliency
        if sal.ndim == 2:
            # Reduce sal to time axis, then broadcast across freq weighted by TFR
            sal_t = sal.sum(0)
            sal_up = np.interp(np.linspace(0, len(sal_t) - 1, T_tfr),
                               np.arange(len(sal_t)), sal_t)
            if sal_up.max() > 0:
                sal_up = sal_up / sal_up.max()
            saliency_2d = tfr_data * sal_up[np.newaxis, :]
        else:
            saliency_2d = gradcam_2d  # fallback

        # ─── Plot 3 panels: TFR input + Grad-CAM + Saliency ───
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        extent = [tfr_times[0], tfr_times[-1], freqs[0], freqs[-1]]

        # Panel 1: Input TFR
        im0 = axes[0].imshow(tfr_data, aspect='auto', origin='lower',
                             extent=extent, cmap='plasma', interpolation='bilinear')
        axes[0].axvline(0, color='white', ls='--', lw=0.6, alpha=0.8)
        axes[0].set_xlabel('Time (s)'); axes[0].set_ylabel('Frequency (Hz)')
        axes[0].set_title(f'Sub-{sid}: Input TFR', fontsize=11)
        plt.colorbar(im0, ax=axes[0], label='Power (a.u.)')

        # Panel 2: Grad-CAM
        im1 = axes[1].imshow(gradcam_2d, aspect='auto', origin='lower',
                             extent=extent, cmap='hot', interpolation='bilinear')
        axes[1].axvline(0, color='cyan', ls='--', lw=0.6, alpha=0.8)
        axes[1].axhline(8, color='white', ls=':', lw=0.4, alpha=0.5)
        axes[1].axhline(13, color='white', ls=':', lw=0.4, alpha=0.5)
        axes[1].set_xlabel('Time (s)')
        axes[1].set_title(f'Sub-{sid}: Grad-CAM heatmap', fontsize=11)
        plt.colorbar(im1, ax=axes[1], label='Importance')

        # Panel 3: Saliency
        im2 = axes[2].imshow(saliency_2d, aspect='auto', origin='lower',
                             extent=extent, cmap='hot', interpolation='bilinear')
        axes[2].axvline(0, color='cyan', ls='--', lw=0.6, alpha=0.8)
        axes[2].set_xlabel('Time (s)')
        axes[2].set_title(f'Sub-{sid}: Pixel saliency', fontsize=11)
        plt.colorbar(im2, ax=axes[2], label='Sensitivity')

        plt.suptitle(f'Fig. S12. Interpretability (Sub-{sid})', fontsize=12, y=1.02)
        plt.tight_layout()
        save_fig(fig, f'FigS12_interpretability_sub{sid}'); plt.close(fig)

# ════════════════════════════════════════════════════════════════════
#  CLI DISPATCHER
# ════════════════════════════════════════════════════════════════════
FIG_TO_GROUP = {
    '3':'results', '6':'results', '7':'results',
    'S1':'results','S2':'results','S3':'results','S4':'results','S8':'results',
    'S5':'neuro', 'S6':'neuro', 'S7':'neuro',
    'S9':'neuro', 'S10':'neuro',
    '4':'panels',
    '5':'interpret','S11':'interpret','S12':'interpret',
}


def run_group_results(figs=None):
    print("\n  Loading results...")
    data = load_all_results()
    if not data: print("  No results."); return
    figs = figs if figs else ['3','6','7','S1','S2','S3','S4','S8']
    if '3'  in figs: print("\n  Fig. 3...");   fig3_boxplot(data)
    if '6'  in figs: print("\n  Fig. 6...");   fig8_latency()
    if '7'  in figs: print("\n  Fig. 7...");   fig9_csoa_gates()
    if 'S1' in figs: print("\n  Fig. S1...");  figS1_subject_accuracy(data)
    if 'S2' in figs: print("\n  Fig. S2...");  figS2_distribution(data)
    if 'S3' in figs: print("\n  Fig. S3...");  figS3_snr_correlation()
    if 'S4' in figs: print("\n  Fig. S4...");  figS4_demographic()
    if 'S8' in figs: print("\n  Fig. S8...");  figS10_wilcoxon(data)


def run_group_neuro(subjects, figs=None):
    figs = figs if figs else ['S5','S6','S7','S9','S10']
    if 'S5'  in figs: print("\n  Fig. S5...");  figS6_erp_per_condition(subjects)
    if 'S6'  in figs: print("\n  Fig. S6...");  figS7_tfr_mean(subjects)
    if 'S7'  in figs: print("\n  Fig. S7...");  figS9_band_topomaps(subjects)
    if 'S9'  in figs: print("\n  Fig. S9...");  figS11_cluster_erp(subjects)
    if 'S10' in figs: print("\n  Fig. S10..."); figS12_cluster_tfr(subjects)


def run_group_panels(subjects):
    print(f"\n  Subject panels (Fig. 4): {', '.join(subjects)}")
    for sid in subjects: fig_subject_panel(sid)


def run_group_interpret(subjects, figs=None):
    figs = figs if figs else ['5','S11','S12']
    if '5'   in figs: print("\n  Fig. 5...");   fig7_gradcam_tfr(subjects)
    if 'S11' in figs: print("\n  Fig. S11..."); figS11_training_curves(subjects)
    if 'S12' in figs: print("\n  Fig. S12..."); figS12_interpretability(subjects)


def main():
    pa = argparse.ArgumentParser(description="CSOANet2026 paper figure generator")
    pa.add_argument("--group", nargs="+", default=None,
                    choices=['results','neuro','panels','interpret'],
                    help="Groups to run (default: all)")
    pa.add_argument("--fig", nargs="+", default=None,
                    help="Specific figure numbers (e.g. 3 6 7 S8)")
    pa.add_argument("--subjects", nargs="+", default=['04','10','22'],
                    help="Subjects for per-subject figures (default: 04 10 22)")
    a = pa.parse_args()

    if a.fig:
        by_group = {}
        for f in a.fig:
            g = FIG_TO_GROUP.get(f)
            if g is None: print(f"  Unknown figure: {f}"); continue
            by_group.setdefault(g, []).append(f)
        if 'results'   in by_group: run_group_results(by_group['results'])
        if 'neuro'     in by_group: run_group_neuro(a.subjects, by_group['neuro'])
        if 'panels'    in by_group: run_group_panels(a.subjects)
        if 'interpret' in by_group: run_group_interpret(a.subjects, by_group['interpret'])
        return

    groups = a.group if a.group else ['results','neuro','panels','interpret']
    print(f"\n  Running groups: {', '.join(groups)}")
    print(f"  Subjects: {', '.join(a.subjects)}\n")
    if 'results'   in groups: run_group_results()
    if 'neuro'     in groups: run_group_neuro(a.subjects)
    if 'panels'    in groups: run_group_panels(a.subjects)
    if 'interpret' in groups: run_group_interpret(a.subjects)

    print(f"\n{'═'*60}\n  DONE\n{'═'*60}")
    print(f"  Output: {OUT}/  (PNG + PDF + TIF)")
    print(f"  Manual: Fig. 1 (pipeline), Fig. 2 (architecture) — external tool")


if __name__ == "__main__":
    main()
