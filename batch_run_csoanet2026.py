import warnings
warnings.filterwarnings("ignore")
import mne
mne.set_log_level("ERROR")

import os, sys, argparse, time
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

os.environ.setdefault("MPLBACKEND", "Agg")
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from eeg_project.scripts.datasets.time_dataset import TimeDataset, TimeConfig
from eeg_project.scripts.models.csoanet2026 import CSOANet2026

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
    cd = Path("eeg_project/data/time_cache"); cd.mkdir(parents=True, exist_ok=True)
    cp = cd / f"sub_{sid}_time.pt"
    if cp.exists():
        print(f"[cache] ", end="", flush=True)
        d = torch.load(cp, weights_only=False); return d["X"], d["Y"]
    print(f"[load] ", end="", flush=True)
    ds = TimeDataset(epochs, y, TimeConfig())
    xs, ys = zip(*[ds[i] for i in range(len(ds))])
    X, Y = torch.stack(xs), torch.stack(ys)
    torch.save({"X": X, "Y": Y}, cp); return X, Y

def augment(X, Y, noise=0.02, shift=5, crop=0.9, ch_drop=0.1, mixup_alpha=0.2):
    """All augmentations: noise + shift + crop + channel drop + Mixup."""
    B,_,C,T = X.shape
    # Standard augmentations
    if noise > 0: X = X + torch.randn_like(X) * noise
    if shift > 0:
        s = torch.randint(-shift, shift+1, (1,)).item()
        if s: X = torch.roll(X, s, dims=-1)
    if crop < 1.0:
        cl = int(T*crop); st = torch.randint(0, T-cl+1, (1,)).item()
        out = torch.zeros_like(X); out[:,:,:,:cl] = X[:,:,:,st:st+cl]; X = out
    if ch_drop > 0:
        X = X * (torch.rand(B,1,C,1, device=X.device) > ch_drop).float()
    # Mixup: blend pairs of trials for better generalization
    if mixup_alpha > 0 and B > 1:
        lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().to(X.device)
        perm = torch.randperm(B, device=X.device)
        X = lam * X + (1 - lam) * X[perm]
        Y_onehot = F.one_hot(Y, 2).float()
        Y_mix = lam * Y_onehot + (1 - lam) * Y_onehot[perm]
        return X, Y_mix, True  # flag: mixed labels
    return X, Y, False

def train_fold(model, Xt, Yt, Xv, Yv, ep_n, lr, dev, amp):
    crit_hard = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ep_n, eta_min=lr*0.01)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    best, wait, bp, bl = 0.0, 0, None, None; bs = 32

    for _ in range(1, ep_n+1):
        model.train(); perm = torch.randperm(len(Yt), device=dev)
        for s in range(0, len(Yt), bs):
            idx = perm[s:s+bs]
            xb, yb, mixed = augment(Xt[idx], Yt[idx])
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=amp):
                logits = model(xb)
                if mixed:
                    loss = -((yb * F.log_softmax(logits, 1)).sum(1)).mean()
                else:
                    loss = crit_hard(logits, yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            CSOANet2026.max_norm(model)
        sch.step()
        model.eval()
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=amp):
            ps = []
            for s in range(0, len(Yv), bs):
                ps.append(model(Xv[s:s+bs]).argmax(1))
            preds = torch.cat(ps)
        acc = (preds == Yv).float().mean().item()
        if acc > best: best, bp, bl, wait = acc, preds.cpu().numpy(), Yv.cpu().numpy(), 0
        else:
            wait += 1
            if wait >= 15: break
    return best, f1_score(bl, bp, average="macro", zero_division=0)

def fmt(s):
    if s<60: return f"{s:.0f}s"
    if s<3600: return f"{s//60:.0f}m{s%60:.0f}s"
    return f"{s//3600:.0f}h{(s%3600)//60:.0f}m"

def run_sub(sid, ep_n, lr, dev, nf, amp):
    epo = load_epo(sid); y = to_binary(epo.events[:,2])
    X, Y = preload_time(epo, y, sid)
    Xg, Yg = X.to(dev), Y.to(dev)
    skf = StratifiedKFold(n_splits=nf, shuffle=True, random_state=42)
    aa, ff = [], []
    for fi, (ti, vi) in enumerate(skf.split(np.zeros(len(Y)), Y.numpy())):
        ti_t = torch.tensor(ti, dtype=torch.long, device=dev)
        vi_t = torch.tensor(vi, dtype=torch.long, device=dev)
        torch.manual_seed(42+fi)
        m = CSOANet2026(Xg[0:1], nc=2).to(dev)
        a, f = train_fold(m, Xg[ti_t], Yg[ti_t], Xg[vi_t], Yg[vi_t], ep_n, lr, dev, amp)
        aa.append(a); ff.append(f)
        m.eval()
        with torch.no_grad():
            xc = m.s1(Xg[vi_t])
            x0,x1,x2 = m.d0(xc),m.d1(xc),m.d2(xc)
            c01,c12,c02 = x0*x1,x1*x2,x0*x2
            g = torch.stack([t.mean([1,2,3]) for t in [x0,x1,x2,c01,c12,c02]],1)
            w = F.softmax(m.csoa(g),1).mean(0)
        lb = ['B/G','Alp','T/D','BxA','AxT','BxT']
        ws = ' '.join(f'{lb[i]}={w[i]:.2f}' for i in range(6))
        print(f"    Fold {fi+1}/{nf}: Acc={a:.4f} F1={f:.4f} | CSOA[{ws}]")
    del Xg, Yg; torch.cuda.empty_cache()
    return np.mean(aa), np.std(aa), np.mean(ff), aa

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--subjects", nargs="+", default=None)
    pa.add_argument("--epochs", type=int, default=50)
    pa.add_argument("--lr", type=float, default=1e-3)
    pa.add_argument("--device", default="cuda")
    pa.add_argument("--folds", type=int, default=5)
    pa.add_argument("--no-amp", action="store_true")
    a = pa.parse_args()
    subs = a.subjects or ALL_SUBJECTS
    dev = torch.device("cuda" if torch.cuda.is_available() and a.device=="cuda" else "cpu")
    amp = dev.type=="cuda" and not a.no_amp; N = len(subs)
    gn = torch.cuda.get_device_name(0) if dev.type=="cuda" else "CPU"
    m0 = CSOANet2026(); np0 = sum(p.numel() for p in m0.parameters()); del m0
    c0 = auto_cfg(62, 2000, 2)
    print(f"\n  [CSOANet2026+CSOA Final] {gn} | {np0:,}p | kt={c0['kt']}")
    print(f"  {N} sub | {a.folds}-fold | ep={a.epochs} | Mixup+crop+noise+cosine+maxnorm\n")

    csv = Path("eeg_project/results/csoanet2026_all_subjects.csv")
    csv.parent.mkdir(parents=True, exist_ok=True)
    done, res = set(), []
    if csv.exists():
        df_old = pd.read_csv(csv)
        done = set(df_old["subject"].astype(str).str.zfill(2))
        res = df_old.to_dict("records")
        if done: print(f"  Skip {len(done)} done.\n")

    t0 = time.time()
    for i, sid in enumerate(subs):
        ts = time.time(); el = ts-t0; eta = (el/max(i,1))*(N-i) if i>0 else 0
        if sid in done: print(f"  [{i+1}/{N}] Sub-{sid} SKIP"); continue
        print(f"  [{i+1}/{N}] Sub-{sid} {'| ETA '+fmt(eta) if i>0 else ''}")
        try:
            m,s,f1,aa = run_sub(sid, a.epochs, a.lr, dev, a.folds, amp)
            res.append({"subject":sid,"cv_basic_core_2d":round(m,4),"cv_basic_core_2d_std":round(s,4),
                        "cv_basic_core_2d_f1":round(f1,4),**{f"fold_{j+1}":round(aa[j],4) for j in range(a.folds)}})
            print(f"  >> Acc={m:.4f}+/-{s:.4f} F1={f1:.4f} | {fmt(time.time()-ts)}")
            pd.DataFrame(res).to_csv(csv, index=False); print(f"  >> Saved ({len(res)})\n")
        except Exception as e:
            print(f"  >> FAIL: {e}\n"); import traceback; traceback.print_exc()

    if not res: return
    v = [r["cv_basic_core_2d"] for r in res]
    print(f"  DONE {fmt(time.time()-t0)} | N={len(res)} Mean={np.mean(v):.4f}+/-{np.std(v):.4f}")

if __name__ == "__main__":
    from eeg_project.scripts.models.csoanet2026 import auto_cfg
    main()
