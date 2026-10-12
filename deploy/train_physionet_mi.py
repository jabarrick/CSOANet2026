"""EEGNet vs CSOANet2026 v1 / v2 on motor-imagery EEG, in float32 and in int8.

For every variant and fold: train, pick the epoch on held-out validation data,
test in float32, then export, quantize to int8 (calibrated on training trials)
and test again.

    python deploy/train_physionet_mi.py                           # PhysioNet-MI, 4-class, 5 subject folds
    python deploy/train_physionet_mi.py --task lr                 # left vs right hand imagery
    python deploy/train_physionet_mi.py --dataset bci2a           # BCI Competition IV 2a, leave-one-subject-out
    python deploy/train_physionet_mi.py --dataset bci2a --protocol within-subject   # session 1 -> session 2
    python deploy/train_physionet_mi.py --variants v2-dynamic v2-unbounded v2-elu   # add ablations
    python deploy/train_physionet_mi.py --synthetic --epochs 5    # smoke test, no download

Runs are resumable: each finished (variant, fold) is written to the per-subject
CSV, and a later call with the same settings skips what is already there.  So
variants can be added one at a time, and an interrupted run continues where it
stopped.  --fresh discards earlier rows.

All variants share the data, split, augmentation, optimiser, schedule and early
stopping.  Weight constraints are each model's own: max-norm 0.25 on every
filter for CSOANet (v1's recipe), max-norm 1 on the spatial filters and 0.25 on
the classifier for EEGNet (Lawhern et al. 2018).  Kernel lengths follow each
model's own rule at the dataset's sampling rate.

The shared recipe is v1's (AdamW, cosine schedule, mixup, crop, noise, channel
drop, label smoothing, early stopping on validation accuracy).  `eegnet-orig`
trains EEGNet with its own published recipe instead: Adam at 1e-3, plain cross-
entropy, no augmentation, constant learning rate, all epochs, keeping the
checkpoint with the lowest validation loss.  It is exported spatial-first.
`v2-longkt-lawhern` trains v2-longkt with that same recipe (keeping CSOANet's own
weight constraints), so recipe and architecture can be separated.

Datasets (--dataset).  Both: average reference, 1-40 Hz band-pass, 0-4 s after
the cue, no ICA (it cannot run on the target).
  physionet  EEGMMIDB via mne.datasets.eegbci.  Imagery runs 4/8/12 (left vs right
             fist) and, for --task 4class, 6/10/14 (both fists vs both feet).
             64 channels, 160 Hz -> 64 x 640.
  bci2a      BCI Competition IV 2a via MOABB (BNCI2014_001, `pip install moabb`).
             9 subjects, 2 sessions, left hand / right hand / feet / tongue.
             22 EEG channels, 250 Hz -> 22 x 1000.

Protocols (--protocol).
  cross-subject   test subjects are never seen in training.  5 folds on PhysioNet,
                  leave-one-subject-out on 2a.  100 epochs, patience 15, batch 64.
  within-subject  2a only: per subject, train on the first session (20 % of it held
                  out for early stopping), test on the second.  This is the usual 2a
                  benchmark.  500 epochs, patience 80, batch 32; EEGNet dropout 0.5.

Normalisation (--norm):
  fixed     one gain per channel, estimated on the training trials.  On the
            device this folds into the spatial filter weights and costs nothing.
  epoch_z   v1's per-epoch, per-channel z-score.  Needs the whole raw window in
            RAM before inference can start.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deploy.baselines import EEGNetSpatialFirst, eegnet_max_norm, make_eegnet  # noqa: E402
from deploy.profile_model import profile_model  # noqa: E402
from deploy.quantize import int8_report  # noqa: E402
from eeg_project.scripts.models.csoanet2026 import CSOANet2026, auto_cfg  # noqa: E402
from eeg_project.scripts.models.csoanet2026_v2 import CSOANetV2  # noqa: E402
from eeg_project.scripts.models.eegnet import EEGNet  # noqa: E402

def _spatial_first(model):
    """Export this EEGNet with the spatial conv first (EEGNetSpatialFirst): identical outputs, far
    fewer MACs.  Training is untouched; if the fold's trained `eegnet` file exists it is reused."""
    model.export_spatial_first = True
    return model


def deploy_form(model):
    """The graph that is exported, quantized and profiled for a trained model."""
    if hasattr(model, "fuse"):
        return model.fuse()
    if getattr(model, "export_spatial_first", False):
        return EEGNetSpatialFirst.from_module(model)
    return model


def _lawhern(model):
    """Train with EEGNet's own recipe (Lawhern et al. 2018) instead of the shared v1 recipe."""
    model.recipe = "lawhern"
    return model


def _relaxed(model):
    """Train with EEGNet's weight constraints (max-norm 1 on the spatial filters, 0.25 on the
    classifier) instead of v1's max-norm 0.25 on every filter.  The network is unchanged."""
    model.relaxed_constraints = True
    return model


BAD_SUBJECTS = (88, 92, 100, 104)   # PhysioNet: wrong sampling rate or irregular trial timing
SFREQ = 160.0                       # PhysioNet
BCI2A_CLASSES = ("left_hand", "right_hand", "feet", "tongue")


def _long(fs):
    return int(fs / 2) | 1          # 0.5 s temporal kernel, as EEGNet


# name -> factory(ch, t, nc, fs, drop); `drop` is EEGNet's dropout, CSOANet takes its own from auto_cfg
VARIANTS = {
    "eegnet": lambda ch, t, nc, fs, drop: make_eegnet(ch, t, nc, fs=fs, drop=drop),
    "eegnet-sf": lambda ch, t, nc, fs, drop: _spatial_first(make_eegnet(ch, t, nc, fs=fs, drop=drop)),
    "eegnet-orig": lambda ch, t, nc, fs, drop: _lawhern(_spatial_first(make_eegnet(ch, t, nc, fs=fs, drop=drop))),
    "v1": lambda ch, t, nc, fs, drop: CSOANet2026(auto_cfg(ch, t, nc), nc=nc),
    "v2": lambda ch, t, nc, fs, drop: CSOANetV2(ch, t, nc),
    "v2-dynamic": lambda ch, t, nc, fs, drop: CSOANetV2(ch, t, nc, gate="dynamic"),      # v1's per-trial gate
    "v2-unbounded": lambda ch, t, nc, fs, drop: CSOANetV2(ch, t, nc, bound=False),       # v1's raw products
    "v2-elu": lambda ch, t, nc, fs, drop: CSOANetV2(ch, t, nc, act="elu"),
    "v2-decim2": lambda ch, t, nc, fs, drop: CSOANetV2(ch, t, nc, decim=2),              # needs --h-freq <= fs/4.5
    "v2-longkt": lambda ch, t, nc, fs, drop: CSOANetV2(ch, t, nc, kt=_long(fs)),
    "v2-longkt-relaxed": lambda ch, t, nc, fs, drop: _relaxed(CSOANetV2(ch, t, nc, kt=_long(fs))),
    "v2-longkt-lawhern": lambda ch, t, nc, fs, drop: _lawhern(CSOANetV2(ch, t, nc, kt=_long(fs))),  # EEGNet's recipe
}
DEFAULT_VARIANTS = ["eegnet", "eegnet-sf", "v1", "v2", "v2-longkt"]


# ----------------------------------------------------------------------------- data
def load_physionet(root, task, l_freq, h_freq, tmin, tmax, max_subjects=None):
    import mne
    from mne.datasets import eegbci
    mne.set_log_level("ERROR")
    run_sets = [((4, 8, 12), 0)] + ([((6, 10, 14), 2)] if task == "4class" else [])
    n_t = int(round((tmax - tmin) * SFREQ))
    subjects = [s for s in range(1, 110) if s not in BAD_SUBJECTS][:max_subjects]
    Xs, ys, gs = [], [], []
    for s in subjects:
        try:
            for runs, offset in run_sets:
                for f in eegbci.load_data(s, list(runs), path=str(root), update_path=False):
                    raw = mne.io.read_raw_edf(f, preload=True)
                    if raw.info["sfreq"] != SFREQ:
                        raise ValueError(f"sfreq {raw.info['sfreq']}")
                    eegbci.standardize(raw)
                    raw.set_eeg_reference("average")
                    raw.filter(l_freq, h_freq)
                    events, _ = mne.events_from_annotations(raw, event_id=dict(T1=1, T2=2))
                    ep = mne.Epochs(raw, events, tmin=tmin, tmax=tmax, baseline=None, preload=True)
                    Xs.append((ep.get_data()[:, :, :n_t] * 1e6).astype(np.float32))
                    ys.append(ep.events[:, 2] - 1 + offset)
                    gs.append(np.full(len(ep), s))
        except Exception as e:
            print(f"  subject {s}: skipped ({e})")
    return np.concatenate(Xs), np.concatenate(ys).astype(np.int64), np.concatenate(gs)


def load_bci2a(l_freq, h_freq, tmin, tmax, max_subjects=None, dataset=None):
    """BCI Competition IV 2a through MOABB.  Returns X (trials, 22, samples), y, subject,
    session (0 = first / training session, 1 = second / evaluation session) and the sampling rate.
    `dataset` lets the tests pass a MOABB FakeDataset."""
    import mne
    from moabb.paradigms import MotorImagery
    mne.set_log_level("ERROR")
    if dataset is None:
        try:
            from moabb.datasets import BNCI2014_001 as BCI2A
        except ImportError:                               # MOABB < 1.0
            from moabb.datasets import BNCI2014001 as BCI2A
        dataset, events = BCI2A(), list(BCI2A_CLASSES)
    else:
        events = list(dataset.event_id)
    paradigm = MotorImagery(n_classes=len(events), events=events, fmin=l_freq, fmax=h_freq, tmin=tmin, tmax=tmax)
    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=list(dataset.subject_list)[:max_subjects])
    fs = float(round((X.shape[2] - 1) / (tmax - tmin)))
    X = X[:, :, :int(round((tmax - tmin) * fs))].astype(np.float32)
    X -= X.mean(axis=1, keepdims=True)                    # average reference (commutes with the band-pass)
    y = np.array([events.index(lab) for lab in labels], dtype=np.int64)
    names = sorted(meta["session"].unique())
    first = [n for n in names if "train" in n.lower() or n in ("session_T", "T")] or names[:1]
    sess = (~meta["session"].isin(first)).to_numpy().astype(np.int64)
    return X, y, meta["subject"].to_numpy().astype(np.int64), sess, fs


def load_synthetic(nc, n_subjects=20, n_trials=40, ch=64, t=640, seed=0):
    """Pink-ish noise plus a class-specific 8-13 Hz source.  For smoke tests only."""
    rng = np.random.default_rng(seed)
    patterns = rng.standard_normal((nc, ch)).astype(np.float32)
    tt = np.arange(t) / SFREQ
    X, y, g, sess = [], [], [], []
    for s in range(n_subjects):
        gain = rng.uniform(0.5, 2.0)
        for i in range(n_trials):
            c = int(rng.integers(nc))
            noise = np.cumsum(rng.standard_normal((ch, t)), axis=1)
            noise -= noise.mean(1, keepdims=True)
            src = np.sin(2 * np.pi * rng.uniform(8, 13) * tt + rng.uniform(0, 6.28))
            X.append(gain * (noise / noise.std() * 10 + 6 * np.outer(patterns[c], src)))
            y.append(c)
            g.append(s + 1)
            sess.append(int(i >= n_trials // 2))
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64), np.asarray(g), np.asarray(sess)


def normalise(X, train_idx, mode):
    if mode == "epoch_z":
        return (X - X.mean(-1, keepdims=True)) / (X.std(-1, keepdims=True) + 1e-8)
    gain = X[train_idx].std(axis=(0, 2), keepdims=True) + 1e-8      # (1, C, 1), training trials only
    return X / gain


def subject_folds(groups, n_folds, val_frac, seed):
    subs = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(subs)
    for test in np.array_split(subs, n_folds):
        rest = np.setdiff1d(subs, test)
        rest = rest[rng.permutation(len(rest))]
        n_val = max(1, int(round(len(rest) * val_frac)))
        yield rest[n_val:], rest[:n_val], test


def make_folds(g, sess, protocol, n_folds, val_frac, seed):
    """Yield (train_idx, val_idx, test_idx, test_subjects) for each fold."""
    if protocol == "cross-subject":
        for tr_s, va_s, te_s in subject_folds(g, n_folds, val_frac, seed):
            yield tuple(np.flatnonzero(np.isin(g, x)) for x in (tr_s, va_s, te_s)) + (te_s,)
    else:                                   # within-subject: first session -> second session
        rng = np.random.default_rng(seed)
        for s in np.unique(g):
            first = np.flatnonzero((g == s) & (sess == 0))
            first = first[rng.permutation(len(first))]
            n_val = max(1, int(round(len(first) * val_frac)))
            yield first[n_val:], first[:n_val], np.flatnonzero((g == s) & (sess == 1)), np.array([s])


# ----------------------------------------------------------------------------- training (v1 recipe)
def augment(X, Y, nc, noise=0.02, shift=5, crop=0.9, ch_drop=0.1, mixup_alpha=0.2):
    B, _, C, T = X.shape
    X = X + torch.randn_like(X) * noise
    s = int(torch.randint(-shift, shift + 1, (1,)))
    if s:
        X = torch.roll(X, s, dims=-1)
    cl = int(T * crop)
    st = int(torch.randint(0, T - cl + 1, (1,)))
    out = torch.zeros_like(X)
    out[..., :cl] = X[..., st:st + cl]
    X = out * (torch.rand(B, 1, C, 1, device=X.device) > ch_drop).float()
    lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().to(X.device)
    perm = torch.randperm(B, device=X.device)
    Y1 = F.one_hot(Y, nc).float() * 0.9 + 0.1 / nc                  # label smoothing 0.1
    return lam * X + (1 - lam) * X[perm], lam * Y1 + (1 - lam) * Y1[perm]


@torch.no_grad()
def predict(model, X, dev, bs=256):
    model.eval()
    return torch.cat([model(X[i:i + bs].to(dev)).argmax(1).cpu() for i in range(0, len(X), bs)]).numpy()


def constrain(model):
    if isinstance(model, EEGNet):
        eegnet_max_norm(model)
    elif getattr(model, "relaxed_constraints", False):
        with torch.no_grad():
            model.spatial.weight.copy_(torch.renorm(model.spatial.weight, 2, 0, 1.0))
            model.fc.weight.copy_(torch.renorm(model.fc.weight, 2, 0, 0.25))
    else:
        CSOANetV2.max_norm(model)                                    # identical to v1's max_norm


@torch.no_grad()
def val_loss(model, X, Y, dev, bs=256):
    model.eval()
    logits = torch.cat([model(X[i:i + bs].to(dev)).float().cpu() for i in range(0, len(X), bs)])
    return float(F.cross_entropy(logits, Y)), float((logits.argmax(1) == Y).float().mean())


def train(model, Xtr, Ytr, Xva, Yva, nc, a, dev):
    """Shared v1 recipe, or EEGNet's own recipe when model.recipe == "lawhern"."""
    lawhern = getattr(model, "recipe", None) == "lawhern"
    if lawhern:
        opt, sch = torch.optim.Adam(model.parameters(), lr=a.lr), None
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=a.lr * 0.01)
    amp = dev.type == "cuda" and not a.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best, best_loss, best_state, wait, best_ep = -1.0, float("inf"), None, 0, 0
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(Ytr))
        for i in range(0, len(perm), a.batch):
            idx = perm[i:i + a.batch]
            if len(idx) < 2:
                continue
            if lawhern:
                xb, yb = Xtr[idx].to(dev), F.one_hot(Ytr[idx].to(dev), nc).float()
            else:
                xb, yb = augment(Xtr[idx].to(dev), Ytr[idx].to(dev), nc)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                loss = -(yb * F.log_softmax(model(xb).float(), 1)).sum(1).mean()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            constrain(model)
        if sch is not None:
            sch.step()
        if lawhern:                              # all epochs, keep the lowest validation loss
            vl, acc = val_loss(model, Xva, Yva, dev)
            if vl < best_loss:
                best_loss, best, best_ep = vl, acc, ep + 1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            continue
        acc = float((predict(model, Xva, dev) == Yva.numpy()).mean())
        if acc > best:
            best, wait, best_ep = acc, 0, ep + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= a.patience:
                break
    model.load_state_dict(best_state)
    step = max(1, len(Ytr) // 2000)                                  # un-augmented training accuracy, ~2000 trials
    train_acc = float((predict(model, Xtr[::step], dev) == Ytr[::step].numpy()).mean())
    return best, ep + 1, train_acc


# ----------------------------------------------------------------------------- resume
def previous_rows(csv, settings_path, settings, fresh):
    """Rows of earlier runs in the same --out, if they were produced with the same settings."""
    rows = []
    if csv.exists() and not fresh:
        if settings_path.exists():
            old = json.loads(settings_path.read_text())
            old.setdefault("dataset", "physionet")
            old.setdefault("protocol", "cross-subject")
            if old != settings:
                sys.exit(f"{csv} was produced with different settings.\n"
                         f"Use another --out, or --fresh to discard it.")
            rows = pd.read_csv(csv).to_dict("records")
            print(f"Resuming: {len({(r['variant'], r['fold']) for r in rows})} (variant, fold) runs already in {csv.name}")
        else:                                   # written by an older version of this script: settings unknown
            backup = csv.with_name(csv.stem + "_previous.csv")
            os.replace(csv, backup)
            print(f"Found results without recorded settings; kept them as {backup.name} and starting over.")
    settings_path.write_text(json.dumps(settings, indent=1))
    return rows


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS, choices=list(VARIANTS))
    ap.add_argument("--dataset", choices=["physionet", "bci2a"], default="physionet")
    ap.add_argument("--protocol", choices=["cross-subject", "within-subject"], default="cross-subject")
    ap.add_argument("--task", choices=["lr", "4class"], default="4class", help="PhysioNet only")
    ap.add_argument("--data-root", default="~/mne_data", help="PhysioNet download folder")
    ap.add_argument("--out", default=str(ROOT / "deploy" / "results"))
    ap.add_argument("--norm", choices=["fixed", "epoch_z"], default="fixed")
    ap.add_argument("--l-freq", type=float, default=1.0)
    ap.add_argument("--h-freq", type=float, default=40.0)
    ap.add_argument("--tmin", type=float, default=0.0)
    ap.add_argument("--tmax", type=float, default=4.0)
    ap.add_argument("--folds", type=int, default=None, help="cross-subject folds: 5 on PhysioNet, one per subject on 2a")
    ap.add_argument("--val-frac", type=float, default=None, help="0.1 of the training subjects, or 0.2 of session 1")
    ap.add_argument("--epochs", type=int, default=None, help="100 cross-subject, 500 within-subject")
    ap.add_argument("--patience", type=int, default=None, help="15 cross-subject, 80 within-subject")
    ap.add_argument("--batch", type=int, default=None, help="64 cross-subject, 32 within-subject")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calib", choices=["minmax", "percentile", "entropy"], default="minmax")
    ap.add_argument("--n-calib", type=int, default=256)
    ap.add_argument("--max-subjects", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--synthetic", action="store_true", help="random stand-in data, for a smoke test")
    ap.add_argument("--fresh", action="store_true", help="discard rows from earlier runs in --out")
    a = ap.parse_args()

    within = a.protocol == "within-subject"
    if within and a.dataset != "bci2a" and not a.synthetic:
        ap.error("--protocol within-subject needs two sessions per subject: use --dataset bci2a")
    dev = torch.device("cuda" if a.device == "cuda" and torch.cuda.is_available() else "cpu")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    span = f"{a.l_freq:g}-{a.h_freq:g}Hz_{a.tmin:g}-{a.tmax:g}s_n{a.max_subjects or 'all'}"

    if a.synthetic:
        nc, fs, tag = (2 if a.task == "lr" else 4), SFREQ, "synthetic" + ("_within" if within else "")
        X, y, g, sess = load_synthetic(nc, n_subjects=a.max_subjects or 20)
    elif a.dataset == "bci2a":
        nc, tag, cache = 4, "bci2a_" + ("within" if within else "loso"), out / f"cache_bci2a_{span}.npz"
        if cache.exists():
            d = np.load(cache)
            X, y, g, sess, fs = d["X"], d["y"], d["g"], d["sess"], float(d["fs"])
        else:
            print("Loading BCI Competition IV 2a through MOABB (first run downloads; later runs use the cache) ...")
            X, y, g, sess, fs = load_bci2a(a.l_freq, a.h_freq, a.tmin, a.tmax, a.max_subjects)
            np.savez(cache, X=X, y=y, g=g, sess=sess, fs=fs)
    else:
        nc, fs, tag, cache = (2 if a.task == "lr" else 4), SFREQ, f"physionet_{a.task}", out / f"cache_{a.task}_{span}.npz"
        if cache.exists():
            d = np.load(cache)
            X, y, g = d["X"], d["y"], d["g"]
        else:
            print("Loading PhysioNet-MI (first run downloads and filters; later runs use the cache) ...")
            X, y, g = load_physionet(Path(a.data_root).expanduser(), a.task, a.l_freq, a.h_freq,
                                     a.tmin, a.tmax, a.max_subjects)
            np.savez(cache, X=X, y=y, g=g)
        sess = np.zeros(len(y), dtype=np.int64)
    tag += f"_n{a.max_subjects}" if a.max_subjects else ""
    ch, t, n_sub = X.shape[1], X.shape[2], len(np.unique(g))

    # protocol defaults
    if a.folds is None:
        a.folds = n_sub if (within or a.dataset == "bci2a") and not a.synthetic else 5
    if within:
        a.folds = n_sub
    a.val_frac = a.val_frac if a.val_frac is not None else (0.2 if within else 0.1)
    a.epochs = a.epochs or (500 if within else 100)
    a.patience = a.patience or (80 if within else 15)
    a.batch = a.batch or (32 if within else 64)
    drop = 0.5 if within else 0.25                       # EEGNet: Lawhern et al.'s within / cross-subject values
    print(f"{tag}: {len(y)} trials, {n_sub} subjects, {ch} ch x {t} samples @ {fs:g} Hz, {nc} classes | "
          f"{a.protocol}, {a.folds} folds | {dev}")

    csv = out / f"{tag}_per_subject.csv"
    keys = ("task", "norm", "l_freq", "h_freq", "tmin", "tmax", "folds", "val_frac", "epochs", "patience",
            "lr", "batch", "seed", "calib", "n_calib", "max_subjects", "synthetic", "dataset", "protocol")
    rows = previous_rows(csv, out / f"{tag}_settings.json", {k: getattr(a, k) for k in keys}, a.fresh)
    done = {(r["variant"], r["fold"]) for r in rows}
    Y = torch.from_numpy(y)
    for fold, (tr, va, te, te_s) in enumerate(make_folds(g, sess, a.protocol, a.folds, a.val_frac, a.seed)):
        if all((v, fold) in done for v in a.variants):
            continue
        Xn = torch.from_numpy(normalise(X, tr, a.norm)).unsqueeze(1)
        rng = np.random.default_rng(a.seed + fold)
        calib = Xn[rng.choice(tr, min(a.n_calib, len(tr)), replace=False)].numpy()
        for v in a.variants:
            if (v, fold) in done:
                print(f"  fold {fold + 1}/{a.folds} {v:<13} already done, skipped")
                continue
            t0 = time.time()
            trained_eegnet = out / "onnx" / f"{tag}_eegnet_fold{fold}_fp32.onnx"
            if v == "eegnet-sf" and trained_eegnet.exists():        # reorder the already trained EEGNet
                export = EEGNetSpatialFirst.from_onnx(trained_eegnet)
                val_acc = train_acc = float("nan")
                n_ep = 0
            else:
                torch.manual_seed(a.seed * 1000 + fold)
                model = VARIANTS[v](ch, t, nc, fs, drop).to(dev)
                val_acc, n_ep, train_acc = train(model, Xn[tr], Y[tr], Xn[va], Y[va], nc, a, dev)
                export = deploy_form(model.cpu().eval())
            fp, q, _ = int8_report(export, calib, Xn[te].numpy(), out / "onnx", f"{tag}_{v}_fold{fold}", a.calib)
            p_fp, p_q, yt, gt = fp.argmax(1), q.argmax(1), y[te], g[te]
            for s in te_s:
                k = gt == s
                rows.append(dict(variant=v, fold=fold, subject=int(s), n_trials=int(k.sum()),
                                 acc_fp32=float((p_fp[k] == yt[k]).mean()),
                                 acc_int8=float((p_q[k] == yt[k]).mean()),
                                 int8_agreement=float((p_fp[k] == p_q[k]).mean()),
                                 train_acc=train_acc, val_acc=val_acc, epochs=n_ep))
            print(f"  fold {fold + 1}/{a.folds} {v:<13} train {train_acc:.3f} val {val_acc:.3f} | test fp32 {(p_fp == yt).mean():.3f}"
                  f" int8 {(p_q == yt).mean():.3f} | "
                  + (f"{n_ep} epochs" if n_ep else "reordered from the trained eegnet") + f", {time.time() - t0:.0f}s")
            pd.DataFrame(rows).to_csv(csv, index=False)

    df = pd.DataFrame(rows)
    order = [v for v in VARIANTS if v in set(df.variant)]
    df["variant"] = pd.Categorical(df.variant, order)
    df = df.sort_values(["variant", "fold", "subject"])
    profiles = {}
    for v in order:
        profiles[v] = profile_model(deploy_form(VARIANTS[v](ch, t, nc, fs, drop)), torch.randn(1, 1, ch, t))
    summ = df.groupby("variant", observed=True).agg(
        subjects=("subject", "nunique"), acc_fp32=("acc_fp32", "mean"), sd_fp32=("acc_fp32", "std"),
        acc_int8=("acc_int8", "mean"), sd_int8=("acc_int8", "std"), int8_agreement=("int8_agreement", "mean"))
    summ["int8_drop"] = summ["acc_fp32"] - summ["acc_int8"]
    for v in summ.index:
        p = profiles[v]
        summ.loc[v, ["params", "MACs_M", "peak_RAM_KB"]] = [p["params"], p["macs"] / 1e6, p["peak_ram_bytes"] / 1024]
    summ.to_csv(out / f"{tag}_summary.csv")
    print(f"\nPer-subject means (chance = {1 / nc:.2f}); per-subject rows in {csv}")
    print(summ.round(4).to_string())
    n = df.subject.nunique()
    complete = [v for v in order if df[df.variant == v].subject.nunique() == n]
    if len(complete) > 1 and n >= 6:
        from scipy.stats import wilcoxon
        ref = next((v for v in ("eegnet-sf", "eegnet") if v in complete), complete[0])
        note = f"paired Wilcoxon over {n} subjects" + ("; few subjects, low power" if n < 20 else "")
        for metric in ("acc_fp32", "acc_int8"):
            wide = df.pivot(index="subject", columns="variant", values=metric)
            for v in complete:
                if v == ref:
                    continue
                d = wide[v] - wide[ref]
                pval = wilcoxon(d).pvalue if (d != 0).any() else 1.0
                print(f"{metric}: {v} minus {ref} = {d.mean():+.4f} ({note}, p = {pval:.3g})")


if __name__ == "__main__":
    main()
