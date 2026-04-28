"""Train ShallowConvNet and DeepConvNet baselines on all subjects.

Schirrmeister et al., Human Brain Mapping 38 (2017).

Usage:
    python batch_run_convnets.py                     # all 46 subjects
    python batch_run_convnets.py --subjects 04 10 22 # quick test
"""
import warnings; warnings.filterwarnings("ignore")
import mne; mne.set_log_level("ERROR")

import os, sys, argparse, time
import numpy as np, pandas as pd
import torch, torch.nn as nn
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

os.environ.setdefault("MPLBACKEND", "Agg")
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from eeg_project.scripts.datasets.time_dataset import TimeDataset, TimeConfig
from eeg_project.scripts.models.convnets import ShallowConvNet, DeepConvNet

ALL_SUBJECTS = [
    "01","02","03","04","05","06","08","09","10","11","14","15","16","17","18","19",
    "20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35",
    "36","37","38","41","42","43","44","45","46","47","48","49","50","54",
]

def to_binary(y):
    c = sorted(np.unique(y))
    return np.vectorize({v: i for i, v in enumerate(c)}.get)(y).astype(np.int64)

def load_epo(sid):
    d = Path("eeg_project/data/processed")
    for n in [f"sub_{sid}_epo.fif", f"sub{sid}-epo.fif"]:
        if (d / n).exists():
            return mne.read_epochs(str(d / n), preload=True, verbose=False)
    raise FileNotFoundError(f"No epoch file for sub-{sid}")

def preload_time(epochs, y, sid):
    cd = Path("eeg_project/data/time_cache"); cd.mkdir(parents=True, exist_ok=True)
    cp = cd / f"sub_{sid}_time.pt"
    if cp.exists():
        print("[cache] ", end="", flush=True)
        d = torch.load(cp, weights_only=False); return d["X"], d["Y"]
    print("[load] ", end="", flush=True)
    ds = TimeDataset(epochs, y, TimeConfig())
    xs, ys = zip(*[ds[i] for i in range(len(ds))])
    X, Y = torch.stack(xs), torch.stack(ys)
    torch.save({"X": X, "Y": Y}, cp); return X, Y

def train_fold(model, Xt, Yt, Xv, Yv, ep_n, lr, dev, amp):
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    best, wait, bp, bl = 0.0, 0, None, None; bs = 32
    for _ in range(1, ep_n + 1):
        model.train(); perm = torch.randperm(len(Yt), device=dev)
        for s in range(0, len(Yt), bs):
            idx = perm[s:s+bs]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=amp):
                loss = crit(model(Xt[idx]), Yt[idx])
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        model.eval()
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=amp):
            ps = [model(Xv[s:s+bs]).argmax(1) for s in range(0, len(Yv), bs)]
            preds = torch.cat(ps)
        acc = (preds == Yv).float().mean().item()
        if acc > best:
            best, bp, bl, wait = acc, preds.cpu().numpy(), Yv.cpu().numpy(), 0
        else:
            wait += 1
            if wait >= 10: break
    return best, f1_score(bl, bp, average="macro", zero_division=0)

def fmt(s):
    if s < 60: return f"{s:.0f}s"
    if s < 3600: return f"{s//60:.0f}m{s%60:.0f}s"
    return f"{s//3600:.0f}h{(s%3600)//60:.0f}m"

def run_sub(sid, ep_n, lr, dev, nf, amp, ModelClass, model_name):
    epo = load_epo(sid); y = to_binary(epo.events[:, 2])
    X, Y = preload_time(epo, y, sid)
    Xg, Yg = X.to(dev), Y.to(dev)
    skf = StratifiedKFold(n_splits=nf, shuffle=True, random_state=42)
    aa, ff = [], []
    for fi, (ti, vi) in enumerate(skf.split(np.zeros(len(Y)), Y.numpy())):
        ti_t = torch.tensor(ti, dtype=torch.long, device=dev)
        vi_t = torch.tensor(vi, dtype=torch.long, device=dev)
        torch.manual_seed(42 + fi)
        m = ModelClass(n_channels=Xg.shape[2], n_timepoints=Xg.shape[3], n_classes=2).to(dev)
        a, f = train_fold(m, Xg[ti_t], Yg[ti_t], Xg[vi_t], Yg[vi_t], ep_n, lr, dev, amp)
        aa.append(a); ff.append(f)
        print(f"    Fold {fi+1}/{nf}: Acc={a:.4f} F1={f:.4f}")
    del Xg, Yg; torch.cuda.empty_cache()
    return np.mean(aa), np.std(aa), np.mean(ff), aa

def main():
    pa = argparse.ArgumentParser(description="Train ShallowConvNet + DeepConvNet")
    pa.add_argument("--subjects", nargs="+", default=None)
    pa.add_argument("--epochs", type=int, default=30)
    pa.add_argument("--lr", type=float, default=1e-3)
    pa.add_argument("--device", default="cuda")
    pa.add_argument("--folds", type=int, default=5)
    pa.add_argument("--no-amp", action="store_true")
    a = pa.parse_args()
    subs = a.subjects or ALL_SUBJECTS
    dev = torch.device("cuda" if torch.cuda.is_available() and a.device == "cuda" else "cpu")
    amp = dev.type == "cuda" and not a.no_amp; N = len(subs)
    gn = torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU"

    models = [
        ("ShallowConvNet", ShallowConvNet, "shallow_all_subjects.csv"),
        ("DeepConvNet",    DeepConvNet,    "deep_all_subjects.csv"),
    ]

    for model_name, ModelClass, csv_name in models:
        m0 = ModelClass(); np0 = m0.count_parameters(); del m0
        print(f"\n  [{model_name}] {gn} | {np0:,} params | AMP:{'ON' if amp else 'OFF'}")
        print(f"  {N} subjects | {a.folds}-fold CV | epochs={a.epochs}\n")

        csv = Path("eeg_project/results") / csv_name
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
                m, s, f1, aa = run_sub(sid, a.epochs, a.lr, dev, a.folds, amp, ModelClass, model_name)
                res.append({"subject": sid, "acc_mean": round(m, 4), "acc_std": round(s, 4),
                            "f1_mean": round(f1, 4),
                            **{f"fold_{j+1}": round(aa[j], 4) for j in range(a.folds)}})
                print(f"  >> Acc={m:.4f}+/-{s:.4f} F1={f1:.4f} | {fmt(time.time()-ts)}")
                pd.DataFrame(res).to_csv(csv, index=False)
                print(f"  >> Saved ({len(res)})\n")
            except Exception as e:
                print(f"  >> FAIL: {e}\n"); import traceback; traceback.print_exc()

        if res:
            v = [r["acc_mean"] for r in res]
            print(f"  {model_name} DONE | N={len(res)} | Mean={np.mean(v):.4f}+/-{np.std(v):.4f}")

if __name__ == "__main__":
    main()
