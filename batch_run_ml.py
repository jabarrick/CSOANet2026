"""Train classical ML classifiers (LDA, SVM, RF) on all subjects.

Extracts ERP + spectral features, runs 5-fold CV, saves per-subject results.

Usage:
    python batch_run_ml.py                     # all 46 subjects
    python batch_run_ml.py --subjects 04 10 22 # quick test
"""
import warnings; warnings.filterwarnings("ignore")
import mne; mne.set_log_level("ERROR")

import os, sys, argparse, time
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

os.environ.setdefault("MPLBACKEND", "Agg")
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from eeg_project.scripts.features import prepare_features

ALL_SUBJECTS = [
    "01","02","03","04","05","06","08","09","10","11","14","15","16","17","18","19",
    "20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35",
    "36","37","38","41","42","43","44","45","46","47","48","49","50","54",
]

def load_epo(sid):
    d = Path("eeg_project/data/processed")
    for n in [f"sub_{sid}_epo.fif", f"sub{sid}-epo.fif"]:
        if (d / n).exists():
            return mne.read_epochs(str(d / n), preload=True, verbose=False)
    raise FileNotFoundError(f"No epoch file for sub-{sid}")

def make_pipeline(name):
    clfs = {
        "LDA": LinearDiscriminantAnalysis(),
        "SVM": SVC(kernel="rbf", C=1.0),
        "RF":  RandomForestClassifier(n_estimators=200, random_state=42),
    }
    return Pipeline([("scaler", StandardScaler()), ("clf", clfs[name])])

def fmt(s):
    if s < 60: return f"{s:.0f}s"
    return f"{s//60:.0f}m{s%60:.0f}s"

def run_sub(sid, nf):
    epo = load_epo(sid)
    X, y = prepare_features(epo, ["erp", "spectral"], scale=False)
    cv = StratifiedKFold(n_splits=nf, shuffle=True, random_state=42)
    row = {"subject": sid}
    for name in ["LDA", "SVM", "RF"]:
        pipe = make_pipeline(name)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
        row[f"cv_{name.lower()}"] = round(scores.mean(), 4)
        row[f"cv_{name.lower()}_std"] = round(scores.std(), 4)
        print(f"    {name}: {scores.mean():.4f} +/- {scores.std():.4f}")
    return row

def main():
    pa = argparse.ArgumentParser(description="Train LDA/SVM/RF on all subjects")
    pa.add_argument("--subjects", nargs="+", default=None)
    pa.add_argument("--folds", type=int, default=5)
    a = pa.parse_args()
    subs = a.subjects or ALL_SUBJECTS; N = len(subs)

    print(f"\n  [Classical ML] LDA / SVM / RF | {N} subjects | {a.folds}-fold CV\n")

    csv = Path("eeg_project/results/ml_all_subjects.csv")
    csv.parent.mkdir(parents=True, exist_ok=True)
    done, res = set(), []
    if csv.exists():
        df_old = pd.read_csv(csv)
        done = set(df_old["subject"].astype(str).str.zfill(2))
        res = df_old.to_dict("records")
        if done: print(f"  Loaded {len(done)} completed (skipping).\n")

    t0 = time.time()
    for i, sid in enumerate(subs):
        ts = time.time()
        if sid in done:
            print(f"  [{i+1}/{N}] Sub-{sid} SKIP"); continue
        print(f"  [{i+1}/{N}] Sub-{sid}")
        try:
            row = run_sub(sid, a.folds)
            res.append(row)
            pd.DataFrame(res).to_csv(csv, index=False)
            print(f"  >> Done ({fmt(time.time()-ts)}) | Saved ({len(res)})\n")
        except Exception as e:
            print(f"  >> FAIL: {e}\n")
            import traceback; traceback.print_exc()

    if not res: return
    df = pd.DataFrame(res)
    for m in ["lda", "svm", "rf"]:
        v = df[f"cv_{m}"].values
        print(f"  {m.upper()}: {np.mean(v):.4f} +/- {np.std(v):.4f}")
    print(f"\n  DONE in {fmt(time.time()-t0)} | Saved: {csv}")

if __name__ == "__main__":
    main()
