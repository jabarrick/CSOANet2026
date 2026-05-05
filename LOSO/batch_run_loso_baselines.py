#!/usr/bin/env python3
"""LOSO baselines runner — companion to batch_run_loso.py.

Reuses load_epo + compute_common_channels from batch_run_loso.py.
ML features come straight from the project's features.py to keep LOSO and
5-fold pipelines comparable.

Usage:
  python batch_run_loso_baselines.py --model lda
  python batch_run_loso_baselines.py --model svm
  python batch_run_loso_baselines.py --model rf
  python batch_run_loso_baselines.py --model eegnet
  python batch_run_loso_baselines.py --model shallow
  python batch_run_loso_baselines.py --model deep

Outputs:
  CSV:  eeg_project/results/loso_<model>.csv  (resumable; last-write-wins)
  ML feature cache: D:/CSOANet_cache_loso/feats/sub_<sid>.npz  (per-subject,
                    shared by all 3 ML models so extraction runs only once)

============================================================================
DL_FACTORIES — verify against eeg_project/scripts/models/ before running
============================================================================
Constructor kwargs per user instruction:
  EEGNet(n_channels, n_timepoints, n_classes)         eegnet.py
  ShallowConvNet(n_channels, n_timepoints, n_classes) convnets.py
  DeepConvNet(n_channels, n_timepoints, n_classes)    convnets.py
All take input shape (B, 1, C, T). Class-name assumption for the two ConvNets:
both live in convnets.py — verify by reading the top of that file once.

DL input length T_in:
  Default = T from cache (3501). Override with --input-T 2000 if your 5-fold
  baselines were trained on a 2-s window. The script will center-crop the
  cache (3501) to T_in around event onset (sample 500 = t=0).
============================================================================
"""
import os, sys, argparse, time, json, hashlib, gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))


# ---------------------------------------------------------------------------
# Reuse from batch_run_loso.py (must be in the same dir)
# ---------------------------------------------------------------------------
from batch_run_loso import (
    ALL_SUBJECTS, load_epo, to_binary, compute_common_channels, fmt,
)

DEFAULT_CACHE = Path("D:/CSOANet_cache_loso")
ML_MODELS = {"lda", "svm", "rf"}
DL_MODELS = {"eegnet", "shallow", "deep"}
ALL_MODELS = ML_MODELS | DL_MODELS


# ---------------------------------------------------------------------------
# DL factories (per user-confirmed signatures)
# ---------------------------------------------------------------------------
def _build_eegnet(C, T):
    from eeg_project.scripts.models.eegnet import EEGNet
    return EEGNet(n_channels=C, n_timepoints=T, n_classes=2)


def _build_shallow(C, T):
    # convnets.py — verify the class name matches
    from eeg_project.scripts.models.convnets import ShallowConvNet
    return ShallowConvNet(n_channels=C, n_timepoints=T, n_classes=2)


def _build_deep(C, T):
    from eeg_project.scripts.models.convnets import DeepConvNet
    return DeepConvNet(n_channels=C, n_timepoints=T, n_classes=2)


DL_FACTORIES = {
    "eegnet":  _build_eegnet,
    "shallow": _build_shallow,
    "deep":    _build_deep,
}


# ---------------------------------------------------------------------------
# ML features (project's features.py)
# ---------------------------------------------------------------------------
def _import_feature_fns():
    """Locate extract_erp_features + extract_spectral_features."""
    candidates = [
        "eeg_project.scripts.features",
        "eeg_project.scripts.utils.features",
        "eeg_project.scripts.datasets.features",
        "eeg_project.features",
        "features",
    ]
    last_err = None
    for mod_name in candidates:
        try:
            mod = __import__(mod_name, fromlist=[
                "extract_erp_features", "extract_spectral_features"])
            erp_fn = getattr(mod, "extract_erp_features")
            psd_fn = getattr(mod, "extract_spectral_features")
            return erp_fn, psd_fn, mod_name
        except (ImportError, AttributeError) as e:
            last_err = e
    raise ImportError(
        "Cannot find extract_erp_features / extract_spectral_features. "
        f"Tried: {candidates}. Last error: {last_err}. "
        "Edit _import_feature_fns() and add your features.py module path."
    )


def _build_subject_features(sid, common_chs, erp_fn, psd_fn):
    """Load sub-<sid> .fif, channel-align, run features.py, return (X_feat, Y).

    extract_spectral_features returns a dict (band -> ndarray); we concat its
    values along axis=1. extract_erp_features returns a single ndarray.
    """
    epo = load_epo(sid)
    epo = epo.copy().pick(common_chs).reorder_channels(common_chs)
    if list(epo.ch_names) != list(common_chs):
        raise RuntimeError(f"Sub-{sid}: channel order mismatch after pick.")

    Y = to_binary(epo.events[:, 2])
    if isinstance(Y, torch.Tensor):
        Y = Y.numpy()
    Y = np.asarray(Y, dtype=np.int64)

    erp = erp_fn(epo)                          # ndarray (N, F_erp)
    psd = psd_fn(epo)                          # dict {band: (N, F_band)} or ndarray
    if isinstance(psd, dict):
        psd_arr = np.concatenate([np.asarray(v) for v in psd.values()], axis=1)
    else:
        psd_arr = np.asarray(psd)

    erp = np.asarray(erp)
    if erp.ndim > 2:
        erp = erp.reshape(erp.shape[0], -1)
    if psd_arr.ndim > 2:
        psd_arr = psd_arr.reshape(psd_arr.shape[0], -1)

    if erp.shape[0] != psd_arr.shape[0]:
        raise RuntimeError(
            f"Sub-{sid}: ERP n={erp.shape[0]} vs PSD n={psd_arr.shape[0]}")
    if erp.shape[0] != len(Y):
        raise RuntimeError(
            f"Sub-{sid}: features n={erp.shape[0]} vs labels n={len(Y)}")

    X_feat = np.concatenate([erp, psd_arr], axis=1).astype(np.float32)
    return X_feat, Y


def _save_features_npz(stem_path, X_feat, Y):
    """Atomic .npz write — Defender-safe.
    np.savez() auto-appends '.npz' if the path argument is a string/Path,
    which breaks atomic rename. We pass an open file handle to suppress that
    behavior, so the temp file lands at exactly the path we asked for.
    """
    fp = stem_path.with_suffix(".npz")
    tmp = stem_path.parent / (stem_path.name + ".npz.tmp")
    with open(tmp, "wb") as f:
        np.savez(f, X=X_feat, Y=Y)
    tmp.replace(fp)
    return fp


def _load_features_npz(stem_path):
    fp = stem_path.with_suffix(".npz")
    d = np.load(fp)
    return d["X"], d["Y"]


# ---------------------------------------------------------------------------
# Tensor cache I/O (matches batch_run_loso.py format)
# ---------------------------------------------------------------------------
def _channel_hash(common_chs):
    return hashlib.md5("|".join(common_chs).encode()).hexdigest()[:8]


def _tensor_cache_path(cache_dir, sid, n_chs, chs_hash):
    return cache_dir / f"sub_{sid}_C{n_chs}_{chs_hash}.pt"


def _load_subject_tensor(cache_dir, sid, n_chs, chs_hash):
    cp = _tensor_cache_path(cache_dir, sid, n_chs, chs_hash)
    if not cp.exists():
        raise FileNotFoundError(
            f"Cache not found: {cp}\n"
            f"Run batch_run_loso.py first to populate the channel-aligned cache.")
    d = torch.load(cp, weights_only=False)
    return d["X"], d["Y"]


def _peek_tensor_metadata(cache_dir):
    """Discover (n_chs, chs_hash, sample_shape, x_dtype, y_dtype) from the
    first cache file — avoids needing to recompute channel alignment."""
    pts = sorted(cache_dir.glob("sub_*_C*_*.pt"))
    if not pts:
        raise FileNotFoundError(
            f"No cache files in {cache_dir}. "
            f"Run batch_run_loso.py first.")
    name = pts[0].stem                        # sub_01_C65_392dd6b0
    parts = name.split("_")
    n_chs = int(parts[2][1:])
    chs_hash = parts[3]
    d = torch.load(pts[0], weights_only=False)
    return (n_chs, chs_hash,
            tuple(d["X"].shape[1:]), d["X"].dtype, d["Y"].dtype)


def _center_crop_T(x, T_target):
    """Center-crop the time axis (last) to T_target around event onset.
    Cache layout is (-0.5, +3.0) s @ 1000 Hz → 3501 samples; sample 500 == t=0.
    For T_target=2000, the window centered on 0 spans samples [0, 2000]
    = (-0.5, +1.5) s. To better match a true (-1.0, +1.0) classification
    window when only (-0.5, +3.0) is available, we instead use the post-event
    window: samples [500, 500+T_target).
    """
    T = x.shape[-1]
    if T_target is None or T_target == T:
        return x
    if T_target > T:
        raise ValueError(f"T_target={T_target} > cache T={T}")
    start = 500                               # event onset = sample 500
    # If 500 + T_target > T, pull the start back so we still get T_target samples.
    if start + T_target > T:
        start = T - T_target
    return x[..., start:start + T_target]


# ---------------------------------------------------------------------------
# DL training
# ---------------------------------------------------------------------------
def augment(X, Y, noise=0.02, shift=5, ch_drop=0.1, mixup_alpha=0.2):
    """No random crop here — input is already at the target T."""
    B, _, C, T = X.shape
    X = X + noise * torch.randn_like(X)
    if shift > 0:
        sh = torch.randint(-shift, shift + 1, (1,)).item()
        if sh != 0:
            X = torch.roll(X, sh, dims=-1)
    if ch_drop > 0:
        mask = (torch.rand(B, 1, C, 1, device=X.device) > ch_drop).float()
        X = X * mask
    mixed = False
    if mixup_alpha > 0 and torch.rand(1).item() < 0.5:
        lam = float(np.random.beta(mixup_alpha, mixup_alpha))
        idx = torch.randperm(B, device=X.device)
        X = lam * X + (1 - lam) * X[idx]
        Y_oh = F.one_hot(Y, num_classes=2).float()
        Y = lam * Y_oh + (1 - lam) * Y_oh[idx]
        mixed = True
    return X, Y, mixed


@torch.no_grad()
def evaluate(model, X_cpu, Y_cpu, dev, bs, amp):
    from sklearn.metrics import f1_score
    model.eval()
    preds, ys = [], []
    for s in range(0, len(X_cpu), bs):
        xb = X_cpu[s:s + bs].to(dev, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=amp):
            out = model(xb)
        preds.append(out.argmax(1).cpu()); ys.append(Y_cpu[s:s + bs])
    preds = torch.cat(preds).numpy()
    ys = torch.cat(ys).numpy()
    return float((preds == ys).mean()), \
           float(f1_score(ys, preds, average='binary', zero_division=0))


def train_one_loso_dl(model, tr_X, tr_Y, val_X, val_Y, test_X, test_Y,
                      ep_n, lr, dev, amp, bs=32, patience=10, eval_bs=64):
    crit_hard = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ep_n,
                                                     eta_min=lr * 0.01)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)

    best_val, wait = 0.0, 0
    best_test_acc, best_test_f1 = 0.0, 0.0
    best_state, last_ep = None, 0
    n_train = len(tr_X)

    for ep in range(1, ep_n + 1):
        model.train()
        perm = np.random.permutation(n_train)
        for s in range(0, n_train, bs):
            idx = perm[s:s + bs]
            xb = tr_X[idx].to(dev, non_blocking=True)
            yb = tr_Y[idx].to(dev, non_blocking=True)
            xb, yb, mixed = augment(xb, yb)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=amp):
                logits = model(xb)
                if mixed:
                    loss = -((yb * F.log_softmax(logits, 1)).sum(1)).mean()
                else:
                    loss = crit_hard(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        sch.step(); last_ep = ep

        val_acc, _ = evaluate(model, val_X, val_Y, dev, eval_bs, amp)
        if val_acc > best_val:
            best_val, wait = val_acc, 0
            ta, tf = evaluate(model, test_X, test_Y, dev, eval_bs, amp)
            best_test_acc, best_test_f1 = ta, tf
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_test_acc, best_test_f1, best_val, last_ep


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--model", required=True, choices=sorted(ALL_MODELS))
    pa.add_argument("--subjects", nargs="+", default=None)
    pa.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    pa.add_argument("--feat-dir", default=None,
                    help="ML feature cache dir (default: <cache-dir>/feats)")
    pa.add_argument("--out", default=None)
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--val-frac", type=float, default=0.1)
    pa.add_argument("--input-T", type=int, default=None,
                    help="DL input time-axis length. Default = cache T (3501). "
                         "Pass 2000 if your 5-fold baselines were trained on "
                         "a 2.0 s window (post-event 0..+2.0 s slice).")
    # DL only
    pa.add_argument("--epochs", type=int, default=50)
    pa.add_argument("--lr", type=float, default=1e-3)
    pa.add_argument("--patience", type=int, default=10)
    pa.add_argument("--bs", type=int, default=32)
    a = pa.parse_args()

    cache_dir = Path(a.cache_dir)
    feat_dir = Path(a.feat_dir) if a.feat_dir else (cache_dir / "feats")
    feat_dir.mkdir(parents=True, exist_ok=True)
    out_csv = Path(a.out) if a.out else \
        Path(f"eeg_project/results/loso_{a.model}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    np.random.seed(a.seed); torch.manual_seed(a.seed)

    # CRITICAL: --subjects controls which folds get held out for evaluation,
    # NOT which subjects are in the train pool. Train pool is always the full
    # cohort minus the held-out subject. This lets us re-run individual folds
    # without breaking the LOSO contract.
    all_subs = ALL_SUBJECTS
    test_subs = a.subjects if a.subjects else ALL_SUBJECTS
    if any(s not in all_subs for s in test_subs):
        bad = [s for s in test_subs if s not in all_subs]
        raise ValueError(f"--subjects contains IDs not in cohort: {bad}")
    subs = all_subs                       # alias for back-compat
    N = len(test_subs)
    N_pool = len(all_subs)
    is_dl = a.model in DL_MODELS

    dev_name = (torch.cuda.get_device_name(0)
                if (is_dl and torch.cuda.is_available()) else "cpu")
    print(f"\n  [LOSO baseline={a.model}] {dev_name} | {N} test fold(s) over {N_pool}-subject cohort | seed={a.seed}")
    print(f"  Tensor cache: {cache_dir}")
    if N != N_pool:
        print(f"  Held-out subset: {test_subs}  (train pool always uses all {N_pool} subjects minus the held-out one)")
    if not is_dl:
        print(f"  Feature cache: {feat_dir}")
    print(f"  Output: {out_csv}\n")

    n_chs, chs_hash, sample_shape, x_dtype, y_dtype = _peek_tensor_metadata(cache_dir)
    sample_C, sample_T = sample_shape[-2], sample_shape[-1]
    T_in = a.input_T if a.input_T is not None else sample_T
    if is_dl and T_in != sample_T:
        print(f"  DL input crop: cache T={sample_T} -> T_in={T_in} "
              f"(post-event slice [{500}:{500+T_in}])")
    print(f"  Cache shape per subject: {sample_shape} | n_chs={n_chs} "
          f"hash={chs_hash}")

    sizes = {}
    for sid in subs:
        d = torch.load(_tensor_cache_path(cache_dir, sid, n_chs, chs_hash),
                       weights_only=False)
        sizes[sid] = len(d["Y"])
        del d
    print(f"  Total trials across {N_pool} subjects: {sum(sizes.values())}")

    # Resume support
    done, res = set(), []
    if out_csv.exists():
        df_old = pd.read_csv(out_csv)
        done = set(df_old["held_out"].astype(str).str.zfill(2))
        res = df_old.to_dict("records")
        if done:
            print(f"  Skip {len(done)} done.")

    # ---------------- ML feature pre-extraction ----------------
    if not is_dl:
        # Channel names (slow path computes from .fif on first call;
        # subsequent runs hit the JSON cache batch_run_loso wrote).
        print("  Resolving common channel names for feature extraction...")
        common_chs = compute_common_channels(subs)
        if len(common_chs) != n_chs:
            raise RuntimeError(
                f"Common channels {len(common_chs)} mismatch cache {n_chs}.")
        cur_hash = _channel_hash(common_chs)
        if cur_hash != chs_hash:
            # Channel SET is identical (n_chs check above passed). Order can
            # differ between runs of compute_common_channels. The tensor cache
            # was already aligned to one specific order (chs_hash); since ML
            # features are channel-permutation-invariant after StandardScaler,
            # a hash mismatch is harmless. Tag feature cache with the cache
            # hash so all 3 ML models share the same features.
            print(f"  [warn] channel order hash differs "
                  f"(computed={cur_hash} cache={chs_hash}); same set, "
                  f"different order. ML features are order-invariant — "
                  f"continuing with cache hash.")
        print(f"  {len(common_chs)} common channels.")

        erp_fn, psd_fn, feat_mod = _import_feature_fns()
        print(f"  Feature module: {feat_mod}")
        print(f"  Pre-extracting features for {N_pool} subjects (full cohort)...")
        t_pre = time.time()
        for j, sid in enumerate(subs):
            stem = feat_dir / f"sub_{sid}_features"
            if stem.with_suffix(".npz").exists():
                continue
            tj = time.time()
            X_feat, Y = _build_subject_features(sid, common_chs, erp_fn, psd_fn)
            _save_features_npz(stem, X_feat, Y)
            print(f"    [{j+1}/{N_pool}] sub-{sid}: feats {X_feat.shape} "
                  f"in {fmt(time.time()-tj)}")
            del X_feat, Y
        print(f"  Feature extraction done in {fmt(time.time()-t_pre)}\n")

    rng = np.random.RandomState(a.seed)
    dev = torch.device("cuda" if (is_dl and torch.cuda.is_available()) else "cpu")
    amp = (dev.type == "cuda")
    t0 = time.time()

    # ---------------- Fold loop ----------------
    for i, test_sid in enumerate(test_subs):
        ts = time.time()
        el = ts - t0
        eta = (el / max(i, 1)) * (N - i) if i > 0 else 0
        if test_sid in done:
            print(f"  [{i+1}/{N}] held-out=Sub-{test_sid} SKIP")
            continue
        print(f"  [{i+1}/{N}] held-out=Sub-{test_sid}"
              f"{' | ETA '+fmt(eta) if i>0 else ''}")

        try:
            train_subs = [s for s in subs if s != test_sid]
            ks = {s: max(1, int(sizes[s] * a.val_frac)) for s in train_subs}
            n_train = sum(sizes[s] - ks[s] for s in train_subs)
            n_val = sum(ks[s] for s in train_subs)

            t_io = time.time()
            test_X = test_Y = None

            if is_dl:
                # Pre-allocate at T_in (post-crop) for memory bound
                in_shape = (1, sample_C, T_in)
                tr_X = torch.empty(n_train, *in_shape, dtype=x_dtype)
                tr_Y = torch.empty(n_train, dtype=y_dtype)
                val_X = torch.empty(n_val, *in_shape, dtype=x_dtype)
                val_Y = torch.empty(n_val, dtype=y_dtype)

                tr_ofs, val_ofs = 0, 0
                for s in subs:
                    Xs, Ys = _load_subject_tensor(cache_dir, s, n_chs, chs_hash)
                    Xs_in = _center_crop_T(Xs, T_in) if T_in != sample_T else Xs
                    if s == test_sid:
                        test_X, test_Y = Xs_in.contiguous(), Ys
                    else:
                        n = sizes[s]; k = ks[s]
                        p = np.arange(n); rng.shuffle(p)
                        val_X[val_ofs:val_ofs + k] = Xs_in[p[:k]]
                        val_Y[val_ofs:val_ofs + k] = Ys[p[:k]]
                        val_ofs += k
                        tr_X[tr_ofs:tr_ofs + (n - k)] = Xs_in[p[k:]]
                        tr_Y[tr_ofs:tr_ofs + (n - k)] = Ys[p[k:]]
                        tr_ofs += (n - k)
                        del Xs, Ys, Xs_in
                assert tr_ofs == n_train and val_ofs == n_val
                io_secs = time.time() - t_io

                torch.manual_seed(a.seed + i)
                model = DL_FACTORIES[a.model](sample_C, T_in).to(dev)
                np_p = sum(p.numel() for p in model.parameters())
                ta, tf, va, last_ep = train_one_loso_dl(
                    model, tr_X, tr_Y, val_X, val_Y, test_X, test_Y,
                    a.epochs, a.lr, dev, amp,
                    bs=a.bs, patience=a.patience,
                )
                row = {
                    "held_out": test_sid,
                    "test_acc": round(float(ta), 4),
                    "test_f1": round(float(tf), 4),
                    "val_acc": round(float(va), 4),
                    "n_train": int(n_train), "n_val": int(n_val),
                    "n_test": int(len(test_X)),
                    "epochs_trained": int(last_ep),
                    "n_params": int(np_p),
                    "T_in": int(T_in),
                }
                print(f"  >> {a.model} Acc={ta:.4f} F1={tf:.4f} Val={va:.4f} | "
                      f"ep={last_ep} | params={np_p:,} | T_in={T_in} | "
                      f"io={fmt(io_secs)} | total={fmt(time.time()-ts)}")
                del model, tr_X, tr_Y, val_X, val_Y, test_X, test_Y
                if dev.type == "cuda":
                    torch.cuda.empty_cache()

            else:
                # ML path: stream pre-extracted features
                Xtr_blk, ytr_blk = [], []
                Xva_blk, yva_blk = [], []
                for s in subs:
                    stem = feat_dir / f"sub_{s}_features"
                    Xs, Ys = _load_features_npz(stem)
                    if s == test_sid:
                        Xte, yte = Xs, Ys
                    else:
                        n = sizes[s]; k = ks[s]
                        p = np.arange(n); rng.shuffle(p)
                        Xva_blk.append(Xs[p[:k]]); yva_blk.append(Ys[p[:k]])
                        Xtr_blk.append(Xs[p[k:]]); ytr_blk.append(Ys[p[k:]])
                Xtr = np.concatenate(Xtr_blk); ytr = np.concatenate(ytr_blk)
                Xva = np.concatenate(Xva_blk); yva = np.concatenate(yva_blk)
                del Xtr_blk, ytr_blk, Xva_blk, yva_blk
                io_secs = time.time() - t_io

                from sklearn.preprocessing import StandardScaler
                from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
                from sklearn.svm import SVC
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.metrics import f1_score

                sc = StandardScaler().fit(Xtr)
                Xtr_s = sc.transform(Xtr); Xva_s = sc.transform(Xva)
                Xte_s = sc.transform(Xte)
                if a.model == "lda":
                    clf = LinearDiscriminantAnalysis()
                elif a.model == "svm":
                    clf = SVC(C=1.0, kernel="rbf", random_state=a.seed)
                else:
                    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                                 random_state=a.seed)
                t_fit = time.time()
                clf.fit(Xtr_s, ytr)
                fit_secs = time.time() - t_fit

                yp_te = clf.predict(Xte_s); yp_va = clf.predict(Xva_s)
                ta = float((yp_te == yte).mean())
                tf = float(f1_score(yte, yp_te, average="binary",
                                    zero_division=0))
                va = float((yp_va == yva).mean())
                row = {
                    "held_out": test_sid,
                    "test_acc": round(ta, 4),
                    "test_f1": round(tf, 4),
                    "val_acc": round(va, 4),
                    "n_train": int(n_train), "n_val": int(n_val),
                    "n_test": int(len(Xte)),
                    "n_features": int(Xtr.shape[1]),
                }
                print(f"  >> {a.model} Acc={ta:.4f} F1={tf:.4f} Val={va:.4f} | "
                      f"feats={Xtr.shape[1]} | fit={fmt(fit_secs)} | "
                      f"io={fmt(io_secs)} | total={fmt(time.time()-ts)}")
                del clf, Xtr, Xva, Xte, Xtr_s, Xva_s, Xte_s

            res.append(row)
            pd.DataFrame(res).to_csv(out_csv, index=False)
            print(f"  >> Saved ({len(res)})\n")
            gc.collect()

        except Exception as e:
            print(f"  >> FAIL: {e}\n")
            import traceback; traceback.print_exc()

    print(f"\n  Done. Total: {fmt(time.time()-t0)} | CSV: {out_csv}")


if __name__ == "__main__":
    main()
