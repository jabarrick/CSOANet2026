"""Confirmatory runs of the registered study (OSF 8st47): architecture x recipe x fold/subject x seed.

    python deploy/prereg_run.py --cell F1a --synthetic --epochs 2          # smoke test on a stand-in
    python deploy/prereg_run.py --cell F2                                  # locked until the freeze

Cells F1a (Lee2019-MI cross-subject), F1b (Cho2017 cross-subject), F2 (Lee2019-MI within-subject).
Architectures: eegnet (EEGNet-8,2), shallow (ShallowConvNet), mibminet (MI-BMInet), csoanet
(CSOANet-v2 long kernel, v2-longkt), each with its own weight constraints and per-protocol dropout.
Recipes R1-R4 and T (deploy/prereg_recipes.py); seeds 0, 1, 2.

Order of work for one cell:
  1. load (locked), exclusion table written to <cell>_exclusions.csv before any training;
  2. condition T: the 24 shared configurations scored on the tuning split only (<cell>_tuning.csv);
  3. every run: train, select the checkpoint on validation data, test in float32 and int8
     (ONNX Runtime static QDQ, per-channel symmetric weights, MinMax on 256 training trials).
     One row per test subject and seed in <cell>_per_subject.csv.  A run whose training loss becomes
     non-finite is repeated once with seed + 100; if that fails too it is kept as it is (flagged).
  4. params, MACs and peak int8 RAM per architecture in <cell>_profiles.csv.

Runs are resumable: finished (arch, recipe, fold, seed) rows are skipped.  Settings are recorded in
<cell>_settings.json and a mismatch stops the script.  --epochs / --patience / --t-configs exist only
for smoke tests and are refused without --synthetic.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deploy import prereg_data as pdata  # noqa: E402
from deploy import prereg_recipes as prec  # noqa: E402
from deploy.baselines import EEGNetSpatialFirst, make_eegnet  # noqa: E402
from deploy.mibminet import MIBMInet  # noqa: E402
from deploy.profile_model import profile_model  # noqa: E402
from deploy.quantize import int8_report  # noqa: E402
from deploy.train_physionet_mi import constrain  # noqa: E402
from eeg_project.scripts.models.convnets import ShallowConvNet  # noqa: E402
from eeg_project.scripts.models.csoanet2026_v2 import CSOANetV2  # noqa: E402

ARCHS = ("eegnet", "shallow", "mibminet", "csoanet")
RECIPE_NAMES = ("R1", "R2", "R3", "R4", "T")
SEEDS = (0, 1, 2)
N_CLASSES = 2
N_CALIB = 256
N_TUNING_SUBJECTS = 10
RETRY_OFFSET = 100


def build_model(arch, ch, t, protocol, nc=N_CLASSES):
    """The four registered architectures at 250 Hz, kernels per each model's own rule."""
    within = protocol == "within-subject"
    if arch == "eegnet":                      # temporal kernel 125, separable 31; dropout 0.25 / 0.5
        return make_eegnet(ch, t, nc, fs=pdata.FS, drop=0.5 if within else 0.25)
    if arch == "shallow":                     # 40 x 25 temporal, 40 spatial, square, mean-pool 75 / 15, log; dropout 0.5
        return ShallowConvNet(ch, t, nc)
    if arch == "mibminet":                    # Nk 32, Nf 64 at 250 Hz; dropout 0.2 / 0.5
        return MIBMInet(ch, t, nc, fs=pdata.FS, p_dropout=0.5 if within else 0.2)
    if arch == "csoanet":                     # v2-longkt: temporal kernel 125, static gate, bounded, ReLU
        return CSOANetV2(ch, t, nc, kt=int(pdata.FS / 2) | 1)
    raise ValueError(arch)


def deploy_form(model):
    return model.fuse() if hasattr(model, "fuse") else model


# ----------------------------------------------------------------------------- data
def load_data(cell, out, synthetic, n_subjects):
    """(data, exclusion table).  Real cells are locked, and so is their cache."""
    if synthetic:
        data = pdata.synthetic_cell(cell, n_subjects=n_subjects)
        subjects = np.unique(data["subject"])
        doc = {int(s): ({0: int(((data["subject"] == s) & (data["session"] == 0)).sum()),
                         1: int(((data["subject"] == s) & (data["session"] == 1)).sum())}
                        if cell == "F2" else int((data["subject"] == s).sum())) for s in subjects}
        return data, pdata.exclusion_table(cell, data, doc, subjects)
    pdata.require_unlock(pdata.CELLS[cell]["dataset"])
    cache = out / f"cache_{pdata.CELLS[cell]['dataset']}_{cell}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        data = {k: d[k] for k in ("X", "y", "subject", "session", "run")}
        data.update(fs=float(d["fs"]), failed=json.loads(str(d["failed"])))
        data["failed"] = {int(k): v for k, v in data["failed"].items()}
        all_subjects = d["all_subjects"]
    else:
        ds = pdata.confirmatory_dataset(pdata.CELLS[cell]["dataset"])
        all_subjects = np.asarray(ds.subject_list)
        data = pdata.load_cell(cell, dataset=ds)
        np.savez(cache, **{k: data[k] for k in ("X", "y", "subject", "session", "run")}, fs=data["fs"],
                 failed=json.dumps(data["failed"]), all_subjects=all_subjects)
    loaded = [int(s) for s in all_subjects if int(s) not in data["failed"]]
    doc = pdata.documented_trials(cell, loaded)
    return data, pdata.exclusion_table(cell, data, doc, all_subjects)


def fixed_exclusions(path, table):
    """The exclusion list is decided once, before any training; later calls must reproduce it."""
    if path.exists():
        old = pd.read_csv(path)
        if sorted(old.loc[old.excluded, "subject"]) != sorted(table.loc[table.excluded, "subject"]):
            sys.exit(f"{path} lists different exclusions than the data now give; stopping.")
    else:
        table.to_csv(path, index=False)


# ----------------------------------------------------------------------------- one run
class Fold:
    """Normalised tensors of one fold (one gain per channel from its training trials)."""

    def __init__(self, data, f):
        X, gain = data["X"], data["X"][f["train"]].std(axis=(0, 2), keepdims=True) + 1e-8
        self.f = f

        def t(idx):
            return torch.from_numpy((X[idx] / gain).astype(np.float32)).unsqueeze(1)
        self.Xtr, self.Xva = t(f["train"]), t(f["val"])
        self.Ytr, self.Yva = (torch.from_numpy(data["y"][f[k]]) for k in ("train", "val"))
        self._t = t
        self.ch, self.t = X.shape[1], X.shape[2]

    def test(self):
        return self._t(self.f["test"])


def fit(arch, recipe, fold, protocol, seed, dev, epochs=None, patience=None):
    """Train with `seed`; on a non-finite loss repeat once with seed + 100 and keep that result."""
    for attempt, s in enumerate((seed, seed + RETRY_OFFSET)):
        torch.manual_seed(s)
        model = build_model(arch, fold.ch, fold.t, protocol).to(dev)
        res = prec.train_recipe(model, recipe, fold.Xtr, fold.Ytr, fold.Xva, fold.Yva, N_CLASSES, protocol, dev,
                                constrain=constrain, epochs=epochs, patience=patience)
        if res["finite"]:
            break
    res.update(seed_used=s, retried=attempt > 0)
    return model.cpu().eval(), res


def evaluate(model, arch, fold, data, seed, workdir, name):
    """Per-subject float32 / int8 balanced accuracy and agreement on the fold's test trials."""
    tr = fold.f["train"]
    rng = np.random.default_rng(seed)
    calib = fold.Xtr[rng.choice(len(tr), min(N_CALIB, len(tr)), replace=False)].numpy()
    Xte = fold.test().numpy()
    yt, gt = data["y"][fold.f["test"]], data["subject"][fold.f["test"]]
    forms = {"": deploy_form(model)}
    if arch == "eegnet":
        forms["_sf"] = EEGNetSpatialFirst.from_module(model)       # extra columns; the standard graph is primary
    preds = {}
    for suffix, m in forms.items():
        try:
            fp, q, (fp_path, _) = int8_report(m, calib, Xte, workdir, name + suffix)
            preds[suffix] = (fp.argmax(1), q.argmax(1))
        except Exception as e:                                      # e.g. non-finite weights after a failed retry
            print(f"    int8 export failed for {name}{suffix}: {type(e).__name__}: {e}")
            with torch.no_grad():
                p = model(torch.from_numpy(Xte)).argmax(1).numpy()
            preds[suffix] = (p, np.full_like(p, -1))
    rows = []
    for s in fold.f["test_subjects"]:
        k = gt == s
        p_fp, p_q = preds[""][0][k], preds[""][1][k]
        r = dict(subject=int(s), n_test=int(k.sum()), bacc_fp32=prec.balanced_accuracy(yt[k], p_fp),
                 bacc_int8=prec.balanced_accuracy(yt[k], p_q), agreement=float((p_fp == p_q).mean()))
        if "_sf" in preds:
            q_sf = preds["_sf"][1][k]
            r.update(bacc_int8_sf=prec.balanced_accuracy(yt[k], q_sf), agreement_sf=float((p_fp == q_sf).mean()))
        rows.append(r)
    return rows


# ----------------------------------------------------------------------------- condition T
def tuning_units(cell, data, folds):
    """The tuning split: fold 1's training / validation subjects (cross), or session 1 of the 10 lowest
    subject IDs with their registered 80/20 split (within).  No test trials."""
    if pdata.CELLS[cell]["protocol"] == "cross-subject":
        f = folds[0]
        return [dict(f, test=np.array([], dtype=np.int64), test_subjects=np.array([]))]
    low = sorted(int(s) for s in np.unique(data["subject"]))[:N_TUNING_SUBJECTS]
    return [dict(f, test=np.array([], dtype=np.int64), test_subjects=np.array([])) for f in folds if f["fold"] in low]


def tune(cell, data, folds, archs, out, tag, dev, n_configs, epochs=None, patience=None):
    """Score the shared T configurations per architecture; return {arch: best Recipe}."""
    protocol = pdata.CELLS[cell]["protocol"]
    configs = prec.sample_t_configs(n_configs)
    path = out / f"{tag}_tuning.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {(r["arch"], r["config"]) for r in rows}
    units = [Fold(data, u) for u in tuning_units(cell, data, folds)]
    for arch in archs:
        for i, cfg in enumerate(configs):
            if (arch, cfg.name) in done:
                continue
            t0 = time.time()
            scores = [fit(arch, cfg, u, protocol, 0, dev, epochs, patience)[1]["val_bacc"] for u in units]
            rows.append(dict(arch=arch, config=cfg.name, index=i, score=float(np.mean(scores)),
                             recipe=json.dumps(cfg.as_dict())))
            pd.DataFrame(rows).to_csv(path, index=False)
            print(f"  tune {arch:<9} {cfg.name} score {np.mean(scores):.3f} ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    best = {}
    for arch in archs:
        d = df[df.arch == arch].sort_values(["score", "index"], ascending=[False, True])
        best[arch] = prec.recipe_from_dict(json.loads(d.iloc[0].recipe))
    return best


# ----------------------------------------------------------------------------- main
def check_settings(path, settings):
    if path.exists():
        old = json.loads(path.read_text())
        if old != settings:
            sys.exit(f"{path} was written with different settings:\n  old {old}\n  new {settings}\n"
                     f"Use another --out.")
    path.write_text(json.dumps(settings, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", choices=list(pdata.CELLS), required=True)
    ap.add_argument("--archs", nargs="+", choices=ARCHS, default=list(ARCHS))
    ap.add_argument("--recipes", nargs="+", choices=RECIPE_NAMES, default=list(RECIPE_NAMES))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--out", default=str(ROOT / "deploy" / "prereg_results"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--synthetic", action="store_true", help="stand-in data with the documented shapes")
    ap.add_argument("--synthetic-subjects", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=None, help="smoke tests only (--synthetic)")
    ap.add_argument("--patience", type=int, default=None, help="smoke tests only (--synthetic)")
    ap.add_argument("--t-configs", type=int, default=prec.N_T_CONFIGS, help="smoke tests only (--synthetic)")
    a = ap.parse_args(argv)
    if not a.synthetic and (a.epochs or a.patience or a.t_configs != prec.N_T_CONFIGS or a.seeds != list(SEEDS)):
        ap.error("--epochs, --patience, --t-configs and --seeds are fixed by the registration; "
                 "they can only be changed with --synthetic")

    spec = pdata.CELLS[a.cell]
    protocol = spec["protocol"]
    dev = torch.device("cuda" if a.device == "cuda" and torch.cuda.is_available() else "cpu")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = a.cell + ("_synthetic" if a.synthetic else "")
    check_settings(out / f"{tag}_settings.json",
                   dict(cell=a.cell, synthetic=a.synthetic, synthetic_subjects=a.synthetic_subjects if a.synthetic else None,
                        epochs=a.epochs, patience=a.patience, t_configs=a.t_configs))

    data, excl = load_data(a.cell, out, a.synthetic, a.synthetic_subjects)
    fixed_exclusions(out / f"{tag}_exclusions.csv", excl)
    if excl.excluded.any():
        print("Excluded before training:\n" + excl[excl.excluded].to_string(index=False))
    data = pdata.apply_exclusions(data, excl)
    folds = pdata.make_folds(a.cell, data)
    ch, t = data["X"].shape[1:]
    print(f"{tag}: {len(data['y'])} trials, {len(np.unique(data['subject']))} subjects, {ch} x {t} @ 250 Hz | "
          f"{protocol}, {len(folds)} folds | {dev}")

    recipes = {r: prec.RECIPES[r] for r in a.recipes if r != "T"}
    t_best = tune(a.cell, data, folds, a.archs, out, tag, dev, a.t_configs, a.epochs, a.patience) \
        if "T" in a.recipes else {}

    csv = out / f"{tag}_per_subject.csv"
    rows = pd.read_csv(csv).to_dict("records") if csv.exists() else []
    done = {(r["arch"], r["recipe"], r["fold"], r["seed"]) for r in rows}
    for f in folds:
        todo = [(arch, rn, seed) for arch in a.archs for rn in a.recipes for seed in a.seeds
                if (arch, rn, f["fold"], seed) not in done]
        if not todo:
            continue
        fold = Fold(data, f)
        for arch, rn, seed in todo:
            recipe = t_best[arch] if rn == "T" else recipes[rn]
            t0 = time.time()
            model, res = fit(arch, recipe, fold, protocol, seed, dev, a.epochs, a.patience)
            per_sub = evaluate(model, arch, fold, data, res["seed_used"], out / "onnx",
                               f"{tag}_{arch}_{rn}_fold{f['fold']}_seed{seed}")
            for r in per_sub:
                rows.append(dict(cell=a.cell, dataset=spec["dataset"], protocol=protocol, arch=arch, recipe=rn,
                                 t_config=recipe.name if rn == "T" else "", fold=f["fold"], seed=seed, **r,
                                 **{k: res[k] for k in ("seed_used", "retried", "finite", "train_acc", "val_acc",
                                                         "val_loss", "val_bacc", "selected_epoch", "epochs_run")}))
            pd.DataFrame(rows).to_csv(csv, index=False)
            m = np.mean([r["bacc_fp32"] for r in per_sub])
            print(f"  fold {f['fold']} {arch:<9} {rn} seed {seed} | val {res['val_acc']:.3f} | test bacc fp32 {m:.3f} | "
                  f"{res['epochs_run']} ep{' RETRIED' if res['retried'] else ''}, {time.time() - t0:.0f}s")

    prof = []
    x = torch.randn(1, 1, ch, t)
    for arch in a.archs:
        m = build_model(arch, ch, t, protocol).eval()
        forms = [("", deploy_form(m))] + ([("spatial-first", EEGNetSpatialFirst.from_module(m))] if arch == "eegnet" else [])
        for form, mm in forms:
            p = profile_model(mm, x)
            prof.append(dict(arch=arch, form=form or "standard", params=p["params"], macs=p["macs"],
                             peak_int8_ram_bytes=p["peak_ram_bytes"]))
    pd.DataFrame(prof).to_csv(out / f"{tag}_profiles.csv", index=False)
    print(f"Done.  Per-subject rows in {csv}")
    return csv


if __name__ == "__main__":
    main()
