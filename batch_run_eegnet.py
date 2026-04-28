import warnings
warnings.filterwarnings("ignore")
import mne
mne.set_log_level("ERROR")

import os, sys, argparse, time
import numpy as np, pandas as pd
import torch, torch.nn as nn
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

os.environ.setdefault("MPLBACKEND", "Agg")
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from eeg_project.scripts.datasets.time_dataset import TimeDataset, TimeConfig
from eeg_project.scripts.models.eegnet import EEGNet

ALL_SUBJECTS = [
    "01","02","03","04","05","06","08","09","10","11","14","15","16","17","18","19",
    "20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35",
    "36","37","38","41","42","43","44","45","46","47","48","49","50","54",
]

def to_binary(y):
    c = sorted(np.unique(y))
    return np.vectorize({v:i for i,v in enumerate(c)}.get)(y).astype(np.int64)

def load_epo(sid):
    d = Path("eeg_project/data/processed")
    for n in [f"sub_{sid}_epo.fif", f"sub{sid}-epo.fif"]:
        if (d/n).exists(): return mne.read_epochs(str(d/n), preload=True, verbose=False)
    raise FileNotFoundError(f"sub_{sid}_epo.fif")

def preload_time(epochs, y, sid):
    """Time-domain: fast (just z-score), with cache."""
    cache_dir = Path("eeg_project/data/time_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"sub_{sid}_time.pt"
    if cache_path.exists():
        print(f"[cache hit] ", end="", flush=True)
        d = torch.load(cache_path, weights_only=False)
        return d["X"], d["Y"]
    print(f"[loading {len(epochs)} trials] ", end="", flush=True)
    ds = TimeDataset(epochs, y, TimeConfig())
    xs, ys = [], []
    for i in range(len(ds)):
        x, l = ds[i]; xs.append(x); ys.append(l)
    X = torch.stack(xs); Y = torch.stack(ys)
    torch.save({"X": X, "Y": Y}, cache_path)
    return X, Y

def train_fold_gpu(model, Xt, Yt, Xv, Yv, ep_n, lr, dev, amp):
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    best, wait, bp = 0.0, 0, None; bs = 32
    for _ in range(1, ep_n+1):
        model.train(); perm = torch.randperm(len(Yt), device=dev)
        for s in range(0, len(Yt), bs):
            idx = perm[s:s+bs]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=amp):
                loss = crit(model(Xt[idx]), Yt[idx])
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        model.eval()
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=amp):
            preds_list = []
            for s in range(0, len(Yv), bs):
                preds_list.append(model(Xv[s:s+bs]).argmax(1))
            preds = torch.cat(preds_list)
        acc = (preds == Yv).float().mean().item()
        if acc > best: best, bp, wait = acc, preds.cpu().numpy(), 0
        else:
            wait += 1
            if wait >= 7: break
    return best, f1_score(Yv.cpu().numpy(), bp, average="macro", zero_division=0)

def fmt(s):
    if s<60: return f"{s:.0f}s"
    if s<3600: return f"{s//60:.0f}m{s%60:.0f}s"
    return f"{s//3600:.0f}h{(s%3600)//60:.0f}m"

def run_sub(sid, ep_n, lr, dev, nf, amp):
    epo = load_epo(sid); y = to_binary(epo.events[:,2])
    X, Y = preload_time(epo, y, sid)
    nch, ntp = X.shape[2], X.shape[3]
    Xg, Yg = X.to(dev), Y.to(dev)
    skf = StratifiedKFold(n_splits=nf, shuffle=True, random_state=42)
    aa, ff = [], []
    for fi, (ti, vi) in enumerate(skf.split(np.zeros(len(Y)), Y.numpy())):
        ti_t = torch.tensor(ti, dtype=torch.long, device=dev)
        vi_t = torch.tensor(vi, dtype=torch.long, device=dev)
        torch.manual_seed(42+fi)
        m = EEGNet(n_channels=nch, n_timepoints=ntp, n_classes=2).to(dev)
        a, f = train_fold_gpu(m, Xg[ti_t], Yg[ti_t], Xg[vi_t], Yg[vi_t], ep_n, lr, dev, amp)
        aa.append(a); ff.append(f)
        print(f"    Fold {fi+1}/{nf}: Acc={a:.4f} F1={f:.4f}")
    del Xg, Yg; torch.cuda.empty_cache()
    return np.mean(aa), np.std(aa), np.mean(ff), aa

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--subjects", nargs="+", default=None)
    pa.add_argument("--epochs", type=int, default=30)
    pa.add_argument("--lr", type=float, default=1e-3)
    pa.add_argument("--device", default="cuda")
    pa.add_argument("--folds", type=int, default=5)
    pa.add_argument("--no-amp", action="store_true")
    a = pa.parse_args()
    subs = a.subjects or ALL_SUBJECTS
    dev = torch.device("cuda" if torch.cuda.is_available() and a.device=="cuda" else "cpu")
    amp = dev.type=="cuda" and not a.no_amp; N = len(subs)
    gn = torch.cuda.get_device_name(0) if dev.type=="cuda" else "CPU"
    gm = f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB" if dev.type=="cuda" else ""
    print(f"\n  [EEGNet-8,2] {gn} {gm} | AMP:{'ON' if amp else 'OFF'}")
    print(f"  {N} subjects | {a.folds}-fold CV | epochs={a.epochs}\n")
    res, fail = [], []; t0 = time.time()
    csv = Path("eeg_project/results/eegnet_all_subjects.csv")
    done_subs = set()
    if csv.exists():
        df_old = pd.read_csv(csv)
        done_subs = set(df_old["subject"].astype(str).str.zfill(2))
        res = df_old.to_dict("records")
        print(f"  Loaded {len(done_subs)} completed subjects, skipping them.\n")
    for i, sid in enumerate(subs):
        ts = time.time(); el = ts-t0; eta = (el/max(i,1))*(N-i) if i>0 else 0
        pct = i/N*100
        if sid in done_subs:
            print(f"  === [{i+1}/{N}] {pct:.0f}% Sub-{sid} | SKIP (already done) ===")
            continue
        print(f"  === [{i+1}/{N}] {pct:.0f}% Sub-{sid} {'| ETA '+fmt(eta) if i>0 else ''} ===")
        try:
            m,s,f1,aa = run_sub(sid, a.epochs, a.lr, dev, a.folds, amp)
            d = time.time()-ts
            res.append({"subject":sid,"cv_eegnet":round(m,4),"cv_eegnet_std":round(s,4),"cv_eegnet_f1":round(f1,4),
                        **{f"fold_{j+1}":round(aa[j],4) for j in range(a.folds)}})
            print(f"  >>> Acc={m:.4f}+/-{s:.4f} F1={f1:.4f} | {fmt(d)}")
            pd.DataFrame(res).to_csv(csv, index=False)
            print(f"  >>> Saved ({len(res)} subjects)\n")
        except Exception as e: print(f"  >>> FAIL: {e}\n"); fail.append(sid)
    tt = time.time()-t0
    if not res: print("No results."); return
    df = pd.DataFrame(res); v = df["cv_eegnet"].values
    csv.parent.mkdir(parents=True,exist_ok=True); df.to_csv(csv,index=False)
    print(f"  DONE in {fmt(tt)} | N={len(res)} Mean={np.mean(v):.4f}+/-{np.std(v):.4f}")
    if fail: print(f"  Failed: {fail}")
    print(f"  Saved: {csv}")
    sc = Path("eeg_project/results/across_subject_summary.csv")
    if sc.exists():
        ds = pd.read_csv(sc); ds["subject"]=ds["subject"].astype(str).str.zfill(2)
        ds["cv_eegnet"]=ds["subject"].map(dict(zip(df["subject"],df["cv_eegnet"])))
        ds.to_csv(sc,index=False); print(f"  Updated: {sc}")

if __name__ == "__main__": main()
