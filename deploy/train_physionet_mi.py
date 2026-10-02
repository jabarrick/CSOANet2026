"""Cross-subject comparison of CSOANet2026 v1 and v2 on PhysioNet-MI, in float and in int8.

For every variant and fold: train on the training subjects, pick the epoch on
held-out validation subjects, test on unseen subjects in float32, then export,
quantize to int8 (calibrated on training trials) and test again.  The question
the output answers: what does each deployment change cost in accuracy, and how
much accuracy survives int8.

    python deploy/train_physionet_mi.py                           # v1 vs v2, 4-class, 5 folds
    python deploy/train_physionet_mi.py --task lr                 # left vs right hand imagery
    python deploy/train_physionet_mi.py --variants v1 v2 v2-dynamic v2-unbounded v2-elu
    python deploy/train_physionet_mi.py --synthetic --epochs 5    # smoke test, no download

Data: EEGMMIDB via mne.datasets.eegbci (downloaded to --data-root on first use).
Imagery runs 4/8/12 (left vs right fist) and, for --task 4class, 6/10/14 (both
fists vs both feet).  Average reference, 1-40 Hz band-pass, 0-4 s after cue,
160 Hz -> 64 x 640.  No ICA: it cannot run on the target.

Normalisation (--norm):
  fixed     one gain per channel, estimated on the training subjects.  On the
            device this folds into the spatial filter weights and costs nothing.
  epoch_z   v1's per-epoch, per-channel z-score.  Needs the whole raw window in
            RAM before inference can start.
"""
import argparse
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

from deploy.profile_model import profile_model  # noqa: E402
from deploy.quantize import int8_report  # noqa: E402
from eeg_project.scripts.models.csoanet2026 import CSOANet2026, auto_cfg  # noqa: E402
from eeg_project.scripts.models.csoanet2026_v2 import CSOANetV2  # noqa: E402

VARIANTS = {
    "v1": lambda ch, t, nc: CSOANet2026(auto_cfg(ch, t, nc), nc=nc),
    "v2": lambda ch, t, nc: CSOANetV2(ch, t, nc),
    "v2-dynamic": lambda ch, t, nc: CSOANetV2(ch, t, nc, gate="dynamic"),      # v1's per-trial gate
    "v2-unbounded": lambda ch, t, nc: CSOANetV2(ch, t, nc, bound=False),       # v1's raw products
    "v2-elu": lambda ch, t, nc: CSOANetV2(ch, t, nc, act="elu"),
    "v2-decim2": lambda ch, t, nc: CSOANetV2(ch, t, nc, decim=2),              # needs --h-freq <= 35
}
BAD_SUBJECTS = (88, 92, 100, 104)   # wrong sampling rate or irregular trial timing
SFREQ = 160.0


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


def load_synthetic(nc, n_subjects=20, n_trials=40, ch=64, t=640, seed=0):
    """Pink-ish noise plus a class-specific 8-13 Hz source.  For smoke tests only."""
    rng = np.random.default_rng(seed)
    patterns = rng.standard_normal((nc, ch)).astype(np.float32)
    tt = np.arange(t) / SFREQ
    X, y, g = [], [], []
    for s in range(n_subjects):
        gain = rng.uniform(0.5, 2.0)
        for _ in range(n_trials):
            c = int(rng.integers(nc))
            noise = np.cumsum(rng.standard_normal((ch, t)), axis=1)
            noise -= noise.mean(1, keepdims=True)
            src = np.sin(2 * np.pi * rng.uniform(8, 13) * tt + rng.uniform(0, 6.28))
            X.append(gain * (noise / noise.std() * 10 + 6 * np.outer(patterns[c], src)))
            y.append(c)
            g.append(s + 1)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64), np.asarray(g)


def normalise(X, train_idx, mode):
    if mode == "epoch_z":
        return (X - X.mean(-1, keepdims=True)) / (X.std(-1, keepdims=True) + 1e-8)
    gain = X[train_idx].std(axis=(0, 2), keepdims=True) + 1e-8      # (1, C, 1), training subjects only
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


def train(model, Xtr, Ytr, Xva, Yva, nc, a, dev):
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=a.lr * 0.01)
    amp = dev.type == "cuda" and not a.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best, best_state, wait = -1.0, None, 0
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(Ytr))
        for i in range(0, len(perm), a.batch):
            idx = perm[i:i + a.batch]
            if len(idx) < 2:
                continue
            xb, yb = augment(Xtr[idx].to(dev), Ytr[idx].to(dev), nc)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                loss = -(yb * F.log_softmax(model(xb).float(), 1)).sum(1).mean()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            CSOANetV2.max_norm(model)                                # identical to v1's max_norm
        sch.step()
        acc = float((predict(model, Xva, dev) == Yva.numpy()).mean())
        if acc > best:
            best, wait = acc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= a.patience:
                break
    model.load_state_dict(best_state)
    return best, ep + 1


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", default=["v1", "v2"], choices=list(VARIANTS))
    ap.add_argument("--task", choices=["lr", "4class"], default="4class")
    ap.add_argument("--data-root", default="~/mne_data")
    ap.add_argument("--out", default=str(ROOT / "deploy" / "results"))
    ap.add_argument("--norm", choices=["fixed", "epoch_z"], default="fixed")
    ap.add_argument("--l-freq", type=float, default=1.0)
    ap.add_argument("--h-freq", type=float, default=40.0)
    ap.add_argument("--tmin", type=float, default=0.0)
    ap.add_argument("--tmax", type=float, default=4.0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calib", choices=["minmax", "percentile", "entropy"], default="minmax")
    ap.add_argument("--n-calib", type=int, default=256)
    ap.add_argument("--max-subjects", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--synthetic", action="store_true", help="random stand-in data, for a smoke test")
    a = ap.parse_args()

    dev = torch.device("cuda" if a.device == "cuda" and torch.cuda.is_available() else "cpu")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    nc = 2 if a.task == "lr" else 4
    tag = "synthetic" if a.synthetic else f"physionet_{a.task}"

    if a.synthetic:
        X, y, g = load_synthetic(nc, n_subjects=a.max_subjects or 20)
    else:
        cache = out / f"cache_{a.task}_{a.l_freq:g}-{a.h_freq:g}Hz_{a.tmin:g}-{a.tmax:g}s_n{a.max_subjects or 'all'}.npz"
        if cache.exists():
            d = np.load(cache)
            X, y, g = d["X"], d["y"], d["g"]
        else:
            print("Loading PhysioNet-MI (first run downloads and filters; later runs use the cache) ...")
            X, y, g = load_physionet(Path(a.data_root).expanduser(), a.task, a.l_freq, a.h_freq,
                                     a.tmin, a.tmax, a.max_subjects)
            np.savez(cache, X=X, y=y, g=g)
    ch, t = X.shape[1], X.shape[2]
    print(f"{tag}: {len(y)} trials, {len(np.unique(g))} subjects, {ch} ch x {t} samples, {nc} classes | {dev}")

    profiles = {}
    for v in a.variants:
        m = VARIANTS[v](ch, t, nc)
        profiles[v] = profile_model(m.fuse() if hasattr(m, "fuse") else m, torch.randn(1, 1, ch, t))

    rows = []
    csv = out / f"{tag}_per_subject.csv"
    for fold, (tr_s, va_s, te_s) in enumerate(subject_folds(g, a.folds, a.val_frac, a.seed)):
        tr, va, te = (np.flatnonzero(np.isin(g, s)) for s in (tr_s, va_s, te_s))
        Xn = torch.from_numpy(normalise(X, tr, a.norm)).unsqueeze(1)
        Y = torch.from_numpy(y)
        rng = np.random.default_rng(a.seed + fold)
        calib = Xn[rng.choice(tr, min(a.n_calib, len(tr)), replace=False)].numpy()
        for v in a.variants:
            t0 = time.time()
            torch.manual_seed(a.seed * 1000 + fold)
            model = VARIANTS[v](ch, t, nc).to(dev)
            val_acc, n_ep = train(model, Xn[tr], Y[tr], Xn[va], Y[va], nc, a, dev)
            model = model.cpu().eval()
            export = model.fuse() if hasattr(model, "fuse") else model
            fp, q, _ = int8_report(export, calib, Xn[te].numpy(), out / "onnx", f"{tag}_{v}_fold{fold}", a.calib)
            p_fp, p_q, yt, gt = fp.argmax(1), q.argmax(1), y[te], g[te]
            for s in te_s:
                k = gt == s
                rows.append(dict(variant=v, fold=fold, subject=int(s), n_trials=int(k.sum()),
                                 acc_fp32=float((p_fp[k] == yt[k]).mean()),
                                 acc_int8=float((p_q[k] == yt[k]).mean()),
                                 int8_agreement=float((p_fp[k] == p_q[k]).mean())))
            print(f"  fold {fold + 1}/{a.folds} {v:<13} val {val_acc:.3f} | test fp32 {(p_fp == yt).mean():.3f}"
                  f" int8 {(p_q == yt).mean():.3f} | {n_ep} epochs, {time.time() - t0:.0f}s")
            pd.DataFrame(rows).to_csv(csv, index=False)

    df = pd.DataFrame(rows)
    summ = df.groupby("variant", sort=False).agg(
        subjects=("subject", "nunique"), acc_fp32=("acc_fp32", "mean"), sd_fp32=("acc_fp32", "std"),
        acc_int8=("acc_int8", "mean"), sd_int8=("acc_int8", "std"), int8_agreement=("int8_agreement", "mean"))
    summ["int8_drop"] = summ["acc_fp32"] - summ["acc_int8"]
    for v in summ.index:
        p = profiles[v]
        summ.loc[v, ["params", "MACs_M", "peak_RAM_KB"]] = [p["params"], p["macs"] / 1e6, p["peak_ram_bytes"] / 1024]
    summ.to_csv(out / f"{tag}_summary.csv")
    print(f"\nPer-subject means (chance = {1 / nc:.2f}); per-subject rows in {csv}")
    print(summ.round(4).to_string())
    if len(a.variants) > 1 and df.subject.nunique() >= 10:
        from scipy.stats import wilcoxon
        base = a.variants[0]
        wide = df.pivot(index="subject", columns="variant", values="acc_int8")
        for v in a.variants[1:]:
            d = wide[v] - wide[base]
            pval = wilcoxon(d).pvalue if (d != 0).any() else 1.0
            print(f"int8 accuracy, {v} minus {base}: {d.mean():+.4f} (paired Wilcoxon over subjects, p = {pval:.3g})")


if __name__ == "__main__":
    main()
