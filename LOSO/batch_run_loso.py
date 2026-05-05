"""LOSO (Leave-One-Subject-Out) cross-validation for CSOANet2026.

Test = held-out subject (full); train = 90% of remaining 45 subjects (per-subject
stratified); val = 10% (used for early stopping). Same augmentation / optimizer
/ schedule / max_norm as batch_run_csoanet2026.py.

CHANNEL ALIGNMENT (v2 fix):
  Different subjects in ds005697 have different auxiliary channels (HEO/VEO/EMG/
  Trigger/CB1/CB2 etc.). 5-fold within-subject works fine, but LOSO concatenation
  across subjects fails. We compute the intersection of channel names across all
  participating subjects (preserving the first subject's order for spatial
  consistency), pick those on every subject, and cache the aligned tensors in a
  separate directory keyed by channel-set hash.

Output: eeg_project/results/csoanet2026_loso.csv
"""
import warnings
warnings.filterwarnings("ignore")
import mne
mne.set_log_level("ERROR")

import os, sys, argparse, time, json, hashlib
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import f1_score

os.environ.setdefault("MPLBACKEND", "Agg")
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from eeg_project.scripts.datasets.time_dataset import TimeDataset, TimeConfig
from eeg_project.scripts.models.csoanet2026 import CSOANet2026, auto_cfg

ALL_SUBJECTS = [
    "01","02","03","04","05","06","08","09","10","11","14","15","16","17","18","19",
    "20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35",
    "36","37","38","41","42","43","44","45","46","47","48","49","50","54",
]
GATE_LABELS = ['BG','Alp','TD','BxA','AxT','BxT']
# Default cache lives off the OneDrive-synced Desktop tree to dodge sync collisions
# during torch.save (~340 MB per file × 46 subjects). Override with --cache-dir.
LOSO_CACHE_DIR = Path("D:/CSOANet_cache_loso")

def to_binary(y):
    c = sorted(np.unique(y))
    return np.vectorize({v:i for i,v in enumerate(c)}.get)(y).astype(np.int64)

def load_epo(sid):
    d = Path("eeg_project/data/processed")
    for n in [f"sub_{sid}_epo.fif", f"sub{sid}-epo.fif"]:
        if (d/n).exists(): return mne.read_epochs(str(d/n), preload=True, verbose=False)
    raise FileNotFoundError(f"sub_{sid}_epo.fif")

def get_channel_names(sid):
    """Lightweight channel-name read using a small JSON cache; avoids re-loading
    the full .fif file once we've seen this subject before."""
    LOSO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    meta = LOSO_CACHE_DIR / f"sub_{sid}_chs.json"
    if meta.exists():
        with open(meta) as f: return json.load(f)["channels"]
    epo = load_epo(sid)
    chs = list(epo.ch_names)
    with open(meta, "w") as f: json.dump({"channels": chs}, f)
    return chs

def compute_common_channels(subs):
    """Return ordered list of channels common to all participating subjects.
    Order follows the first subject (preserves typical 10-20 spatial layout)."""
    print(f"  Scanning channels across {len(subs)} subjects...")
    ch_per_sub = {sid: get_channel_names(sid) for sid in subs}
    sizes = sorted({len(v) for v in ch_per_sub.values()})
    print(f"  Per-subject channel counts seen: {sizes}")
    first = ch_per_sub[subs[0]]
    common = set(first)
    for sid in subs[1:]:
        common &= set(ch_per_sub[sid])
    common_ordered = [c for c in first if c in common]
    union = set()
    for chs in ch_per_sub.values(): union |= set(chs)
    dropped = sorted(union - common)
    print(f"  Common channels: {len(common_ordered)} | dropped (non-common): {len(dropped)}")
    if dropped:
        print(f"  Dropped: {dropped}")
    if not common_ordered:
        raise RuntimeError("No common channels across subjects.")
    return common_ordered

def preload_time_aligned(sid, common_chs, chs_hash):
    """Load subject's epochs, pick + reorder to common_chs, return (X, Y).
    Cache keyed by (n_chs, hash) so different subject subsets don't collide.
    Atomic write (.pt.tmp -> rename) so half-written files never get loaded."""
    cp = LOSO_CACHE_DIR / f"sub_{sid}_C{len(common_chs)}_{chs_hash}.pt"
    if cp.exists():
        d = torch.load(cp, weights_only=False); return d["X"], d["Y"]
    epo = load_epo(sid)
    epo = epo.copy().pick(common_chs).reorder_channels(common_chs)
    if list(epo.ch_names) != list(common_chs):
        raise RuntimeError(f"Sub-{sid}: channel order mismatch after pick.")
    y = to_binary(epo.events[:,2])
    ds = TimeDataset(epo, y, TimeConfig())
    xs, ys = zip(*[ds[i] for i in range(len(ds))])
    X, Y = torch.stack(xs), torch.stack(ys)
    tmp = cp.with_suffix(".pt.tmp")
    torch.save({"X": X, "Y": Y}, tmp)
    tmp.replace(cp)
    return X, Y

def augment(X, Y, noise=0.02, shift=5, crop=0.9, ch_drop=0.1, mixup_alpha=0.2):
    B,_,C,T = X.shape
    if noise > 0: X = X + torch.randn_like(X) * noise
    if shift > 0:
        s = torch.randint(-shift, shift+1, (1,)).item()
        if s: X = torch.roll(X, s, dims=-1)
    if crop < 1.0:
        cl = int(T*crop); st = torch.randint(0, T-cl+1, (1,)).item()
        out = torch.zeros_like(X); out[:,:,:,:cl] = X[:,:,:,st:st+cl]; X = out
    if ch_drop > 0:
        X = X * (torch.rand(B,1,C,1, device=X.device) > ch_drop).float()
    if mixup_alpha > 0 and B > 1:
        lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().to(X.device)
        perm = torch.randperm(B, device=X.device)
        X = lam * X + (1 - lam) * X[perm]
        Y_onehot = F.one_hot(Y, 2).float()
        Y_mix = lam * Y_onehot + (1 - lam) * Y_onehot[perm]
        return X, Y_mix, True
    return X, Y, False

def fmt(s):
    if s<60: return f"{s:.0f}s"
    if s<3600: return f"{s//60:.0f}m{s%60:.0f}s"
    return f"{s//3600:.0f}h{(s%3600)//60:.0f}m"

def evaluate(model, X_cpu, Y_cpu, dev, bs, amp):
    model.eval()
    preds = []
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=amp):
        for s in range(0, len(Y_cpu), bs):
            xb = X_cpu[s:s+bs].to(dev, non_blocking=True)
            preds.append(model(xb).argmax(1).cpu())
    p = torch.cat(preds).numpy()
    y = Y_cpu.numpy()
    return (p == y).mean(), f1_score(y, p, average="macro", zero_division=0), p

def probe_csoa(model, X_cpu, dev, bs=64):
    model.eval()
    ws = []
    with torch.no_grad():
        for s in range(0, len(X_cpu), bs):
            xb = X_cpu[s:s+bs].to(dev, non_blocking=True)
            xc = model.s1(xb)
            x0,x1,x2 = model.d0(xc), model.d1(xc), model.d2(xc)
            c01,c12,c02 = x0*x1, x1*x2, x0*x2
            g = torch.stack([t.mean([1,2,3]) for t in [x0,x1,x2,c01,c12,c02]],1)
            ws.append(F.softmax(model.csoa(g), 1).cpu())
    return torch.cat(ws).mean(0).numpy()

def train_one_loso(model, tr_X, tr_Y, val_X, val_Y, test_X, test_Y,
                   ep_n, lr, dev, amp, bs=32, patience=10, eval_bs=64):
    crit_hard = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ep_n, eta_min=lr*0.01)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)

    best_val, wait = 0.0, 0
    best_test_acc, best_test_f1 = 0.0, 0.0
    best_state, last_ep = None, 0
    n_train = len(tr_X)

    for ep in range(1, ep_n+1):
        model.train()
        perm = np.random.permutation(n_train)
        for s in range(0, n_train, bs):
            idx = perm[s:s+bs]
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
            CSOANet2026.max_norm(model)
        sch.step(); last_ep = ep

        val_acc, _, _ = evaluate(model, val_X, val_Y, dev, eval_bs, amp)
        if val_acc > best_val:
            best_val, wait = val_acc, 0
            ta, tf, _ = evaluate(model, test_X, test_Y, dev, eval_bs, amp)
            best_test_acc, best_test_f1 = ta, tf
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience: break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_test_acc, best_test_f1, best_val, last_ep

def main():
    global LOSO_CACHE_DIR
    pa = argparse.ArgumentParser()
    pa.add_argument("--subjects", nargs="+", default=None)
    pa.add_argument("--epochs", type=int, default=50)
    pa.add_argument("--lr", type=float, default=1e-3)
    pa.add_argument("--device", default="cuda")
    pa.add_argument("--patience", type=int, default=10)
    pa.add_argument("--val-frac", type=float, default=0.1)
    pa.add_argument("--no-amp", action="store_true")
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--out", default="eeg_project/results/csoanet2026_loso.csv")
    pa.add_argument("--cache-dir", default=str(LOSO_CACHE_DIR),
                    help="Where to store aligned-channel tensor caches "
                         "(~340 MB/subject × 46 subjects ≈ 16 GB). "
                         "Default avoids OneDrive-synced Desktop tree.")
    a = pa.parse_args()

    # Late-bind cache dir from CLI (module-global so preload_time_aligned sees it)
    LOSO_CACHE_DIR = Path(a.cache_dir)

    # CRITICAL: --subjects controls which folds get held out for evaluation,
    # NOT which subjects are used to train. Train pool is always the full
    # cohort minus the held-out subject. This lets us re-run individual folds
    # without breaking the LOSO contract.
    all_subs = ALL_SUBJECTS
    test_subs = a.subjects or ALL_SUBJECTS
    if any(s not in all_subs for s in test_subs):
        bad = [s for s in test_subs if s not in all_subs]
        raise ValueError(f"--subjects contains IDs not in cohort: {bad}")
    subs = all_subs                       # alias retained for back-compat
    dev = torch.device("cuda" if torch.cuda.is_available() and a.device=="cuda" else "cpu")
    amp = dev.type=="cuda" and not a.no_amp
    N = len(test_subs)
    N_pool = len(all_subs)
    gn = torch.cuda.get_device_name(0) if dev.type=="cuda" else "CPU"

    print(f"\n  [LOSO CSOANet2026+CSOA] {gn} | {N} test fold(s) over {N_pool}-subject cohort | seed={a.seed}")
    print(f"  Cache dir: {LOSO_CACHE_DIR}")
    if N != N_pool:
        print(f"  Held-out subset: {test_subs}  (train pool always uses all {N_pool} subjects minus the held-out one)")
    LOSO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    common_chs = compute_common_channels(all_subs)
    chs_hash = hashlib.md5(",".join(common_chs).encode()).hexdigest()[:8]
    print(f"  Channel hash: {chs_hash} (cache key)")

    print(f"  Loading subjects with aligned channels...")
    data, t_pre = {}, time.time()
    for sid in all_subs:
        X, Y = preload_time_aligned(sid, common_chs, chs_hash)
        data[sid] = (X, Y)
    print(f"  Preloaded {N_pool} subjects in {fmt(time.time()-t_pre)}")

    shapes = {data[s][0].shape[2] for s in subs}
    assert len(shapes) == 1, f"Channel dims still differ: {shapes}"

    # Capture metadata, then free the heavy dict — fold loop reads from D:/ cache.
    # Keeps peak RAM ≈ tr_X (24.5 GB at T=3501) + val_X (~2.7 GB) + test_X (~0.6 GB)
    # ≈ 30 GB instead of 76 GB.
    sizes = {s: len(data[s][1]) for s in subs}
    y_dtype = data[subs[0]][1].dtype
    sample_X = data[subs[0]][0]
    sample_shape = sample_X.shape[1:]   # (1, C, T)
    sample_C, sample_T = sample_shape[-2], sample_shape[-1]
    x_dtype = sample_X.dtype
    init_sample = sample_X[0:1].clone()  # keep one sample for model init
    n_total = sum(sizes.values())
    gb_total = sum(data[s][0].numel() for s in subs) * sample_X.element_size() / 1e9
    print(f"  Total trials: {n_total} | sample shape: {tuple(sample_shape)} "
          f"| dict mem: {gb_total:.2f} GB")

    import gc
    del data, sample_X
    gc.collect()
    print(f"  Released dict; fold loop will stream from {LOSO_CACHE_DIR}")

    m0 = CSOANet2026(init_sample, nc=2)
    np0 = sum(p.numel() for p in m0.parameters())
    cfg = auto_cfg(sample_C, sample_T, 2)
    del m0
    print(f"  Model: {np0:,} params | kt={cfg['kt']} | ep={a.epochs} | "
          f"patience={a.patience} | val_frac={a.val_frac}\n")

    csv = Path(a.out); csv.parent.mkdir(parents=True, exist_ok=True)
    done, res = set(), []
    if csv.exists():
        df_old = pd.read_csv(csv)
        done = set(df_old["held_out"].astype(str).str.zfill(2))
        res = df_old.to_dict("records")
        if done: print(f"  Skip {len(done)} done.\n")

    rng = np.random.RandomState(a.seed)
    t0 = time.time()

    def _cache_path(s):
        return LOSO_CACHE_DIR / f"sub_{s}_C{len(common_chs)}_{chs_hash}.pt"

    for i, test_sid in enumerate(test_subs):
        ts = time.time()
        el = ts - t0
        eta = (el / max(i, 1)) * (N - i) if i > 0 else 0
        if test_sid in done:
            print(f"  [{i+1}/{N}] held-out=Sub-{test_sid} SKIP")
            continue
        print(f"  [{i+1}/{N}] held-out=Sub-{test_sid}{' | ETA '+fmt(eta) if i>0 else ''}")

        try:
            # Plan train/val sizes up front so we can pre-allocate (no temp copies)
            train_subs = [s for s in subs if s != test_sid]
            ks = {s: max(1, int(sizes[s] * a.val_frac)) for s in train_subs}
            n_train = sum(sizes[s] - ks[s] for s in train_subs)
            n_val = sum(ks[s] for s in train_subs)

            tr_X = torch.empty(n_train, *sample_shape, dtype=x_dtype)
            tr_Y = torch.empty(n_train, dtype=y_dtype)
            val_X = torch.empty(n_val, *sample_shape, dtype=x_dtype)
            val_Y = torch.empty(n_val, dtype=y_dtype)

            # Stream subjects from cache; copy slices in place; free immediately
            t_io = time.time()
            tr_ofs, val_ofs = 0, 0
            test_X, test_Y = None, None
            for s in subs:
                d = torch.load(_cache_path(s), weights_only=False)
                Xs, Ys = d["X"], d["Y"]
                if s == test_sid:
                    test_X, test_Y = Xs, Ys
                else:
                    n = sizes[s]; k = ks[s]
                    p = np.arange(n); rng.shuffle(p)
                    val_X[val_ofs:val_ofs+k] = Xs[p[:k]]
                    val_Y[val_ofs:val_ofs+k] = Ys[p[:k]]
                    val_ofs += k
                    tr_X[tr_ofs:tr_ofs+(n-k)] = Xs[p[k:]]
                    tr_Y[tr_ofs:tr_ofs+(n-k)] = Ys[p[k:]]
                    tr_ofs += (n-k)
                    del Xs, Ys, d
            assert tr_ofs == n_train and val_ofs == n_val
            io_secs = time.time() - t_io

            torch.manual_seed(a.seed + i)
            model = CSOANet2026(init_sample, nc=2).to(dev)

            ta, tf, va, last_ep = train_one_loso(
                model, tr_X, tr_Y, val_X, val_Y, test_X, test_Y,
                a.epochs, a.lr, dev, amp, bs=32, patience=a.patience
            )

            gates = probe_csoa(model, test_X, dev)
            ws = ' '.join(f'{GATE_LABELS[k]}={gates[k]:.2f}' for k in range(6))

            row = {
                "held_out": test_sid,
                "test_acc": round(float(ta), 4),
                "test_f1": round(float(tf), 4),
                "val_acc": round(float(va), 4),
                "n_train": int(n_train),
                "n_val": int(n_val),
                "n_test": int(len(test_X)),
                "epochs_trained": int(last_ep),
                **{f"gate_{GATE_LABELS[k]}": round(float(gates[k]), 4) for k in range(6)},
            }
            res.append(row)
            print(f"  >> Test Acc={ta:.4f} F1={tf:.4f} | Val={va:.4f} | "
                  f"ep={last_ep} | CSOA[{ws}] | io={fmt(io_secs)} | total={fmt(time.time()-ts)}")
            pd.DataFrame(res).to_csv(csv, index=False)
            print(f"  >> Saved ({len(res)})\n")

            del model, tr_X, tr_Y, val_X, val_Y, test_X, test_Y
            gc.collect()
            if dev.type == "cuda": torch.cuda.empty_cache()
        except Exception as e:
            print(f"  >> FAIL: {e}\n")
            import traceback; traceback.print_exc()

    if not res: return
    v = np.array([r["test_acc"] for r in res])
    f = np.array([r["test_f1"] for r in res])
    print(f"  DONE {fmt(time.time()-t0)} | N={len(res)}")
    print(f"  LOSO Acc = {v.mean():.4f} +/- {v.std():.4f}  [min {v.min():.4f}, max {v.max():.4f}]")
    print(f"  LOSO F1  = {f.mean():.4f} +/- {f.std():.4f}")

if __name__ == "__main__":
    main()
