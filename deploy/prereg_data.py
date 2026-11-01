"""Confirmatory datasets, preprocessing, splits and exclusions for the registered study (OSF 8st47).

    F1a  Lee2019-MI  54 subjects, 62 ch, 1000 Hz, cross-subject, 5 subject folds, 0-4 s
    F1b  Cho2017     52 subjects, 64 ch,  512 Hz, cross-subject, 5 subject folds, 0-3 s
    F2   Lee2019-MI  within-subject: session 1 trains, session 2 tests, 0-4 s

Preprocessing (MOABB 1.7.2 LeftRightImagery): 1-40 Hz band-pass, resample to 250 Hz, window after the
cue, all EEG channels, no artefact rejection beyond MOABB defaults, then average reference.  The
average reference is applied after filtering and resampling; all three are linear and act on every
channel alike, so the order does not change the result.

LOCK.  Lee2019-MI and Cho2017 are the confirmatory datasets.  Every path that would fetch or read them
calls `require_unlock`, which raises `ConfirmatoryLocked` unless the environment variable
CSOANET_CONFIRMATORY_UNLOCK is exactly "1".  Only Yu sets it, after the code is frozen at a git tag.
Tests use MOABB FakeDataset stand-ins and synthetic arrays.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

UNLOCK_VAR = "CSOANET_CONFIRMATORY_UNLOCK"
CONFIRMATORY_CLASSES = ("Lee2019_MI", "Cho2017")
FS = 250.0
L_FREQ, H_FREQ = 1.0, 40.0
EVENTS = ("left_hand", "right_hand")
MIN_FRACTION = 0.8

CELLS = {
    "F1a": dict(dataset="Lee2019_MI", protocol="cross-subject", tmax=4.0, n_ch=62, sfreq=1000.0, n_subjects=54),
    "F1b": dict(dataset="Cho2017", protocol="cross-subject", tmax=3.0, n_ch=64, sfreq=512.0, n_subjects=52),
    "F2": dict(dataset="Lee2019_MI", protocol="within-subject", tmax=4.0, n_ch=62, sfreq=1000.0, n_subjects=54),
}
N_FOLDS, VAL_FRAC_CROSS, VAL_FRAC_WITHIN, SPLIT_SEED = 5, 0.1, 0.2, 0
LEE_TRIALS_PER_SESSION = 200            # 2 runs (offline + online) x 100 trials


class ConfirmatoryLocked(RuntimeError):
    pass


class ExclusionError(RuntimeError):
    pass


def require_unlock(what):
    """Refuse to touch a confirmatory dataset unless Yu has unlocked it after the freeze."""
    if os.environ.get(UNLOCK_VAR) != "1":
        raise ConfirmatoryLocked(
            f"{what} is a confirmatory dataset of the preregistered study (OSF 8st47) and is locked until "
            f"the code is frozen at a git tag.  Only Yu may set {UNLOCK_VAR}=1, after the freeze.  "
            f"Use the synthetic stand-ins (--synthetic) for development and tests.")


def is_confirmatory(dataset):
    """True for MOABB Lee2019_MI / Cho2017 instances, including subclasses."""
    names = {c.__name__ for c in type(dataset).__mro__}
    code = str(getattr(dataset, "code", "")).lower().replace("_", "-")
    return bool(names & set(CONFIRMATORY_CLASSES)) or code in ("lee2019-mi", "cho2017")


def confirmatory_dataset(name):
    """The MOABB dataset object for a confirmatory dataset.  Locked."""
    if name not in CONFIRMATORY_CLASSES:
        raise ValueError(name)
    require_unlock(name)
    if name == "Lee2019_MI":
        from moabb.datasets import Lee2019_MI
        return Lee2019_MI(train_run=True, test_run=True)   # both runs of each session (MOABB's default loads one)
    from moabb.datasets import Cho2017
    return Cho2017()


# ----------------------------------------------------------------------------- loading
def load_cell(cell, dataset=None, subjects=None):
    """Load and preprocess one cell.

    `dataset=None` loads the registered dataset (locked).  Tests pass a MOABB FakeDataset stand-in.
    Returns a dict: X (trials, ch, samples) float32, y (0 left, 1 right), subject, session (0 = first),
    run, fs, failed {subject: reason} for subjects MOABB could not load.
    """
    spec = CELLS[cell]
    if dataset is None:
        dataset = confirmatory_dataset(spec["dataset"])
    elif is_confirmatory(dataset):
        require_unlock(type(dataset).__name__)
    import mne
    from moabb.paradigms import LeftRightImagery
    mne.set_log_level("ERROR")
    tmax = spec["tmax"]
    paradigm = LeftRightImagery(fmin=L_FREQ, fmax=H_FREQ, tmin=0.0, tmax=tmax, resample=FS)
    n_t = int(round(tmax * FS))
    subjects = list(dataset.subject_list) if subjects is None else list(subjects)
    Xs, ys, gs, ss, rs, failed = [], [], [], [], [], {}
    for s in subjects:
        try:
            X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[s])
            if X.shape[2] < n_t:
                raise ValueError(f"{X.shape[2]} samples per trial, expected {n_t}")
        except Exception as e:                      # MOABB cannot load this subject
            failed[int(s)] = f"{type(e).__name__}: {e}"
            continue
        X = X[:, :, :n_t].astype(np.float32)
        X -= X.mean(axis=1, keepdims=True)         # average reference
        Xs.append(X)
        ys.append(np.array([EVENTS.index(lab) for lab in labels], dtype=np.int64))
        gs.append(meta["subject"].to_numpy().astype(np.int64))
        names = sorted(meta["session"].astype(str).unique())
        ss.append(meta["session"].astype(str).map({n: i for i, n in enumerate(names)}).to_numpy().astype(np.int64))
        rs.append(meta["run"].astype(str).to_numpy())
    if not Xs:
        raise RuntimeError(f"{cell}: no subject could be loaded: {failed}")
    return dict(X=np.concatenate(Xs), y=np.concatenate(ys), subject=np.concatenate(gs),
                session=np.concatenate(ss), run=np.concatenate(rs), fs=FS, failed=failed)


def synthetic_cell(cell, n_subjects=8, trials_per_session=40, seed=0):
    """Preprocessed-shape stand-in for a cell (62 x 1000 or 64 x 750 at 250 Hz, two sessions for Lee2019).
    Noise plus a class-dependent 8-13 Hz spatial pattern.  For smoke tests only."""
    spec = CELLS[cell]
    ch, t = spec["n_ch"], int(round(spec["tmax"] * FS))
    n_sess = 2 if spec["dataset"] == "Lee2019_MI" else 1
    rng = np.random.default_rng(seed)
    patterns = rng.standard_normal((2, ch)).astype(np.float32)
    tt = np.arange(t) / FS
    X, y, g, s = [], [], [], []
    for sub in range(1, n_subjects + 1):
        gain = rng.uniform(0.5, 2.0)
        for sess in range(n_sess):
            for i in range(trials_per_session):
                c = i % 2
                src = np.sin(2 * np.pi * rng.uniform(8, 13) * tt + rng.uniform(0, 6.28))
                X.append(gain * (rng.standard_normal((ch, t)) * 5 + 2 * np.outer(patterns[c], src)))
                y.append(c)
                g.append(sub)
                s.append(sess)
    X = np.asarray(X, dtype=np.float32)
    X -= X.mean(axis=1, keepdims=True)
    return dict(X=X, y=np.asarray(y, np.int64), subject=np.asarray(g, np.int64), session=np.asarray(s, np.int64),
                run=np.zeros(len(y), dtype=str), fs=FS, failed={})


# ----------------------------------------------------------------------------- exclusions
def read_cho2017_documented(mat_path):
    """Documented trials of one Cho2017 subject: 2 x n_imagery_trials from the authors' original .mat file."""
    from scipy.io import loadmat
    eeg = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False,
                  verify_compressed_data_integrity=False)["eeg"]
    n = getattr(eeg, "n_imagery_trials", None)
    if n is None or np.size(n) != 1:
        raise ExclusionError(f"{mat_path}: field eeg.n_imagery_trials missing or not a scalar; "
                             f"stop and report, do not guess the documented trial count")
    return 2 * int(np.asarray(n).item())


def documented_trials(cell, subjects, dataset=None):
    """{subject: documented count} (F1a, F1b) or {subject: {session: count}} (F2).

    Cho2017's counts come from the original .mat files (locked; `dataset.data_path` downloads them).
    """
    spec = CELLS[cell]
    if spec["dataset"] == "Lee2019_MI":
        if spec["protocol"] == "within-subject":
            return {int(s): {0: LEE_TRIALS_PER_SESSION, 1: LEE_TRIALS_PER_SESSION} for s in subjects}
        return {int(s): 2 * LEE_TRIALS_PER_SESSION for s in subjects}
    if dataset is None:
        dataset = confirmatory_dataset("Cho2017")
    elif is_confirmatory(dataset):
        require_unlock(type(dataset).__name__)
    out = {}
    for s in subjects:
        path = dataset.data_path(int(s))
        path = path[0] if isinstance(path, (list, tuple)) else path
        out[int(s)] = read_cho2017_documented(path)
    return out


def exclusion_table(cell, data, documented, all_subjects):
    """One row per registered subject: loaded and documented trials, excluded or not, and why.

    A subject is excluded only if MOABB cannot load it or fewer than 80 % of its documented trials remain
    (F2: in either session).  Decided before any training."""
    rows = []
    within = CELLS[cell]["protocol"] == "within-subject"
    for s in all_subjects:
        s = int(s)
        if s in data["failed"]:
            rows.append(dict(subject=s, n_loaded="", n_documented="", excluded=True,
                             reason=f"MOABB could not load it ({data['failed'][s]})"))
            continue
        k = data["subject"] == s
        if within:
            doc = documented[s]
            got = {sess: int((k & (data["session"] == sess)).sum()) for sess in doc}
            short = [sess for sess in doc if got[sess] < MIN_FRACTION * doc[sess]]
            rows.append(dict(subject=s, n_loaded=";".join(str(got[x]) for x in sorted(doc)),
                             n_documented=";".join(str(doc[x]) for x in sorted(doc)), excluded=bool(short),
                             reason="; ".join(f"session {x + 1}: {got[x]} of {doc[x]} trials (< 80 %)" for x in short)))
        else:
            n, doc = int(k.sum()), documented[s]
            rows.append(dict(subject=s, n_loaded=n, n_documented=doc, excluded=n < MIN_FRACTION * doc,
                             reason=f"{n} of {doc} trials (< 80 %)" if n < MIN_FRACTION * doc else ""))
    return pd.DataFrame(rows)


def apply_exclusions(data, table):
    keep = np.isin(data["subject"], table.loc[~table.excluded, "subject"].to_numpy())
    out = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(keep) else v) for k, v in data.items()}
    return out


# ----------------------------------------------------------------------------- splits
def cross_subject_folds(subjects, n_folds=N_FOLDS, val_frac=VAL_FRAC_CROSS, seed=SPLIT_SEED):
    """[(train_subjects, val_subjects, test_subjects)]: subjects shuffled with `seed` into folds, then
    10 % of each fold's training subjects held out for validation (same procedure as the pilot)."""
    from deploy.train_physionet_mi import subject_folds
    return list(subject_folds(np.asarray(subjects), n_folds, val_frac, seed))


def within_subject_split(first_session_idx, val_frac=VAL_FRAC_WITHIN, seed=SPLIT_SEED):
    """(train_idx, val_idx): a random 20 % of one subject's session-1 trials, fresh rng(seed) per subject."""
    idx = np.asarray(first_session_idx)
    perm = idx[np.random.default_rng(seed).permutation(len(idx))]
    n_val = max(1, int(round(len(idx) * val_frac)))
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


def make_folds(cell, data):
    """[dict(fold, train, val, test, test_subjects)] of trial indices into `data`."""
    g, sess = data["subject"], data["session"]
    subjects = np.unique(g)
    if CELLS[cell]["protocol"] == "cross-subject":
        return [dict(fold=i, train=np.flatnonzero(np.isin(g, tr)), val=np.flatnonzero(np.isin(g, va)),
                     test=np.flatnonzero(np.isin(g, te)), test_subjects=np.sort(te))
                for i, (tr, va, te) in enumerate(cross_subject_folds(subjects))]
    out = []
    for s in subjects:
        tr, va = within_subject_split(np.flatnonzero((g == s) & (sess == 0)))
        out.append(dict(fold=int(s), train=tr, val=va, test=np.flatnonzero((g == s) & (sess == 1)),
                        test_subjects=np.array([s])))
    return out


def normalise(X, train_idx):
    """One gain per channel, from the fold's training trials."""
    gain = X[train_idx].std(axis=(0, 2), keepdims=True) + 1e-8
    return (X / gain).astype(np.float32)
