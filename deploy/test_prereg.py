"""Checks for the preregistered study code (OSF 8st47).  Run: python deploy/test_prereg.py

Everything here runs on synthetic stand-ins.  Nothing in this file downloads, loads or inspects
Lee2019-MI or Cho2017, and nothing sets CSOANET_CONFIRMATORY_UNLOCK.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deploy import prereg_recipes as pr  # noqa: E402


# ----------------------------------------------------------------------------- item 2: recipes
def test_recipes_match_registration():
    R = pr.RECIPES
    assert set(R) == {"R1", "R2", "R3", "R4"}
    r1 = R["R1"]
    assert (r1.optimizer, r1.lr, r1.weight_decay, r1.schedule, r1.batch) == ("adamw", 1e-3, 1e-3, "cosine", None)
    assert all(r1.augmentations().values()) and r1.label_smoothing == 0.1 and r1.selection == "val_acc"
    r2 = R["R2"]
    assert (r2.optimizer, r2.lr, r2.schedule, r2.selection, r2.label_smoothing) == ("adam", 1e-3, "constant", "val_loss", 0)
    assert not any(r2.augmentations().values())
    r3 = R["R3"]
    assert (r3.optimizer, r3.lr, r3.eps, r3.schedule, r3.batch, r3.selection) == ("adam", 1e-2, 1e-7, "step", 16, "val_loss")
    assert not any(r3.augmentations().values()) and r3.label_smoothing == 0
    r4 = R["R4"]                                       # R1 without augmentation and label smoothing
    assert not any(r4.augmentations().values()) and r4.label_smoothing == 0
    assert (r4.optimizer, r4.lr, r4.weight_decay, r4.schedule, r4.selection) == ("adamw", 1e-3, 1e-3, "cosine", "val_acc")
    assert pr.PROTOCOL == {"cross-subject": dict(epochs=100, batch=64, patience=15),
                           "within-subject": dict(epochs=500, batch=32, patience=80)}


def _lrs(recipe, E):
    m = nn.Linear(2, 2)
    opt, sch = pr.make_optimizer(m, recipe, E)
    out = []
    for _ in range(E):
        out.append(opt.param_groups[0]["lr"])
        opt.step()
        if sch:
            sch.step()
    return opt, out


def test_schedules():
    opt, lrs = _lrs(pr.RECIPES["R3"], 100)
    assert isinstance(opt, torch.optim.Adam) and opt.param_groups[0]["eps"] == 1e-7
    assert np.allclose(lrs[:40], 1e-2) and np.allclose(lrs[40:80], 1e-3) and np.allclose(lrs[80:], 1e-4)
    opt, lrs = _lrs(pr.RECIPES["R1"], 100)
    assert isinstance(opt, torch.optim.AdamW) and opt.param_groups[0]["weight_decay"] == 1e-3
    assert lrs[0] == 1e-3 and lrs[-1] < 1.1e-5 and all(a >= b for a, b in zip(lrs, lrs[1:]))
    opt, sch = pr.make_optimizer(nn.Linear(2, 2), pr.RECIPES["R1"], 100)
    for _ in range(100):
        opt.step()
        sch.step()
    assert abs(opt.param_groups[0]["lr"] - 1e-5) < 1e-12     # cosine to 1e-5 over E
    _, lrs = _lrs(pr.RECIPES["R2"], 10)
    assert np.allclose(lrs, 1e-3)


def test_r1_augmentation_reproduces_pilot():
    from deploy.train_physionet_mi import augment as pilot_augment
    X, Y = torch.randn(16, 1, 8, 200), torch.randint(0, 2, (16,))
    torch.manual_seed(3)
    a = pilot_augment(X, Y, 2)
    torch.manual_seed(3)
    b = pr.augment(X, Y, 2, pr.RECIPES["R1"])
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_no_augmentation_is_identity():
    X, Y = torch.randn(8, 1, 4, 100), torch.randint(0, 2, (8,))
    for name in ("R2", "R3", "R4"):
        xa, ya = pr.augment(X, Y, 2, pr.RECIPES[name])
        assert torch.equal(xa, X) and torch.equal(ya, nn.functional.one_hot(Y, 2).float())
    xa, ya = pr.augment(X, Y, 2, pr.Recipe("ls", label_smoothing=0.1))
    assert torch.allclose(ya.max(1).values, torch.full((8,), 0.95))


def test_each_augmentation_switch():
    X, Y = torch.randn(8, 1, 4, 100) + 3, torch.randint(0, 2, (8,))
    for k in pr.AUGMENTATIONS:
        torch.manual_seed(0)
        xa, ya = pr.augment(X, Y, 2, pr.Recipe("one", **{k: True}))
        if k == "crop":
            assert (xa[..., 90:] == 0).all()                  # 90 % crop, zero padded at the end
        if k == "ch_drop":
            assert set(torch.unique(xa / X).tolist()) <= {0.0, 1.0}
        if k == "mixup":
            assert ((ya > 0) & (ya < 1)).any() or torch.equal(ya, nn.functional.one_hot(Y, 2).float())
        else:
            assert torch.equal(ya, nn.functional.one_hot(Y, 2).float())
        if k == "noise":
            assert 0.01 < float((xa - X).std()) < 0.03


def test_t_space():
    a, b = pr.sample_t_configs(), pr.sample_t_configs()
    assert len(a) == 24 and a == b                            # deterministic: seed 0, shared by architectures
    assert a != pr.sample_t_configs(seed=1)
    lrs = []
    for r in a:
        assert (r.optimizer, r.weight_decay) in pr.T_OPTIMIZERS
        assert r.schedule in pr.T_SCHEDULES and r.batch in pr.T_BATCHES
        assert r.label_smoothing in (0.0, 0.1) and r.selection in ("val_loss", "val_acc")
        assert 1e-4 <= r.lr <= 1e-2
        lrs.append(r.lr)
        assert pr.recipe_from_dict(r.as_dict()) == r
    big = pr.sample_t_configs(2000)
    lg = np.log10([r.lr for r in big])
    assert abs(lg.mean() + 3) < 0.05                           # log-uniform on [1e-4, 1e-2]
    assert {(r.optimizer, r.weight_decay) for r in big} == set(pr.T_OPTIMIZERS)
    assert all(np.mean([getattr(r, k) for r in big]) > 0.4 for k in pr.AUGMENTATIONS)


class _Tiny(nn.Module):
    def __init__(self, ch=4, t=64, nc=2):
        super().__init__()
        self.fc = nn.Linear(ch * t, nc)

    def forward(self, x):
        return self.fc(x.flatten(1))


def _toy(n=96, ch=4, t=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    Y = torch.randint(0, 2, (n,), generator=g)
    X = torch.randn(n, 1, ch, t, generator=g)
    X[:, :, 0] += (Y.float() * 2 - 1).view(-1, 1, 1) * 1.5
    return X, Y


def test_train_recipe_selection_and_retry_flag():
    X, Y = _toy()
    for name in ("R1", "R2", "R3", "R4"):
        torch.manual_seed(0)
        res = pr.train_recipe(_Tiny(), pr.RECIPES[name], X[:64], Y[:64], X[64:], Y[64:], 2, "cross-subject",
                              torch.device("cpu"), constrain=lambda m: None, epochs=6)
        assert res["finite"] and 1 <= res["selected_epoch"] <= 6 and res["val_acc"] > 0.6, (name, res)
    # val_loss selection runs all epochs
    res = pr.train_recipe(_Tiny(), pr.RECIPES["R2"], X[:64], Y[:64], X[64:], Y[64:], 2, "cross-subject",
                          torch.device("cpu"), constrain=lambda m: None, epochs=7)
    assert res["epochs_run"] == 7
    # early stopping on val accuracy stops after `patience` epochs without improvement
    res = pr.train_recipe(_Tiny(), pr.RECIPES["R4"], X[:64], Y[:64], X[64:], Y[64:], 2, "cross-subject",
                          torch.device("cpu"), constrain=lambda m: None, epochs=200, patience=2)
    assert res["epochs_run"] == res["selected_epoch"] + 2
    # a non-finite loss stops training and is reported
    m = _Tiny()
    with torch.no_grad():
        m.fc.weight.fill_(float("nan"))
    res = pr.train_recipe(m, pr.RECIPES["R2"], X[:64], Y[:64], X[64:], Y[64:], 2, "cross-subject",
                          torch.device("cpu"), constrain=lambda m: None, epochs=3)
    assert not res["finite"] and res["epochs_run"] == 0


def test_balanced_accuracy():
    assert pr.balanced_accuracy([0, 0, 0, 1], [0, 0, 0, 0]) == 0.5
    assert pr.balanced_accuracy([0, 1, 1, 1], [0, 1, 1, 0]) == (1 + 2 / 3) / 2


# ----------------------------------------------------------------------------- item 3: data
import os  # noqa: E402
import re  # noqa: E402
import tempfile  # noqa: E402
from unittest import mock  # noqa: E402

from deploy import prereg_data as pdata  # noqa: E402


def _locked_env():
    """The environment with the unlock variable removed (it is never set here)."""
    env = {k: v for k, v in os.environ.items() if k != pdata.UNLOCK_VAR}
    return mock.patch.dict(os.environ, env, clear=True)


def _stand_in(cell, n_subjects=3, n_events=20, tones=()):
    """A MOABB FakeDataset with the cell's documented channels, rate and sessions.  `tones`: sinusoids
    (Hz, amplitude in V) added to every channel, to check the band-pass."""
    import mne
    from moabb.datasets.fake import FakeDataset
    spec = pdata.CELLS[cell]
    names = tuple(mne.channels.make_standard_montage("standard_1005").ch_names[:spec["n_ch"]])
    n_sess = 2 if spec["dataset"] == "Lee2019_MI" else 1

    class StandIn(FakeDataset):
        def _generate_raw(self, n, d):
            raw = super()._generate_raw(n, d)
            t = raw.times
            data = raw.get_data()
            for f, a in tones:
                data[:-1] += a * np.sin(2 * np.pi * f * t)
            raw._data[:] = data
            return raw

    return StandIn(event_list=pdata.EVENTS, n_sessions=n_sess, n_runs=2, n_subjects=n_subjects,
                   channels=names, sfreq=spec["sfreq"], duration=n_events * 6.5, n_events=n_events, seed=0)


def test_confirmatory_lock():
    from moabb.datasets import Cho2017, Lee2019_MI
    lee, cho = object.__new__(Lee2019_MI), object.__new__(Cho2017)    # no __init__: nothing is touched
    assert pdata.is_confirmatory(lee) and pdata.is_confirmatory(cho)
    assert not pdata.is_confirmatory(_stand_in("F1a", 1))
    with _locked_env():
        for cell in pdata.CELLS:
            for call in (lambda: pdata.load_cell(cell),
                         lambda: pdata.load_cell(cell, dataset=lee if "Lee" in pdata.CELLS[cell]["dataset"] else cho)):
                try:
                    call()
                except pdata.ConfirmatoryLocked as e:
                    assert pdata.UNLOCK_VAR in str(e)
                else:
                    raise AssertionError(f"{cell} loaded without unlock")
        for name in pdata.CONFIRMATORY_CLASSES:
            try:
                pdata.confirmatory_dataset(name)
                raise AssertionError(name)
            except pdata.ConfirmatoryLocked:
                pass
        for ds in (None, cho):
            try:
                pdata.documented_trials("F1b", [1], dataset=ds)
                raise AssertionError("Cho2017 documented counts read without unlock")
            except pdata.ConfirmatoryLocked:
                pass
    with mock.patch.dict(os.environ, {pdata.UNLOCK_VAR: "true"}):       # only exactly "1" unlocks
        try:
            pdata.require_unlock("x")
            raise AssertionError
        except pdata.ConfirmatoryLocked:
            pass


def test_no_unguarded_confirmatory_access():
    """Nothing in deploy/ constructs Lee2019_MI / Cho2017 except confirmatory_dataset, and no source file
    sets the unlock variable."""
    for f in (ROOT / "deploy").glob("*.py"):
        src = f.read_text()
        if f.name != "test_prereg.py":
            assert not re.search(pdata.UNLOCK_VAR + r"\s*['\"]?\s*\]?\s*=\s*['\"]?1", src), f
            assert "setenv" not in src or pdata.UNLOCK_VAR not in src, f
        calls = re.findall(r"\b(Lee2019_MI|Cho2017)\(", src)
        if f.name == "prereg_data.py":
            assert len(calls) == 2, calls
            body = src[src.index("def confirmatory_dataset"):src.index("# ---", src.index("def confirmatory_dataset"))]
            assert body.index("require_unlock(") < body.index("Lee2019_MI(") and "Cho2017()" in body
        elif f.name != "test_prereg.py":
            assert not calls, (f, calls)


def test_preprocessing_on_stand_ins():
    for cell, n_t in (("F1a", 1000), ("F1b", 750), ("F2", 1000)):
        spec = pdata.CELLS[cell]
        d = pdata.load_cell(cell, dataset=_stand_in(cell, 2, tones=((10.0, 2e-5), (60.0, 2e-5), (0.2, 2e-4))))
        assert d["X"].shape[1:] == (spec["n_ch"], n_t) and d["X"].dtype == np.float32 and d["fs"] == 250.0
        assert set(d["y"]) == {0, 1} and set(d["subject"]) == {1, 2} and not d["failed"]
        assert np.abs(d["X"].mean(axis=1)).max() < 1e-4 * d["X"].std()          # average reference
        n_sess = 2 if spec["dataset"] == "Lee2019_MI" else 1
        assert set(d["session"]) == set(range(n_sess))
        if spec["dataset"] == "Lee2019_MI":
            assert len(set(d["run"])) == 2                                       # both runs of each session
        # band-pass 1-40 Hz: the 10 Hz tone passes, 60 Hz and 0.2 Hz are removed
        X = d["X"] - d["X"].mean(axis=1, keepdims=True)
        P = np.abs(np.fft.rfft(X[:, 0], axis=-1)).mean(0)
        f = np.fft.rfftfreq(n_t, 1 / 250.0)
        p10 = P[np.argmin(abs(f - 10))]
        assert P[np.argmin(abs(f - 60))] < p10 / 20, cell


def test_cho2017_documented_counts_from_mat():
    from scipy.io import savemat
    with tempfile.TemporaryDirectory() as d:
        good, bad = Path(d) / "s01.mat", Path(d) / "s02.mat"
        savemat(good, {"eeg": {"n_imagery_trials": 120, "srate": 512}})
        savemat(bad, {"eeg": {"srate": 512}})
        assert pdata.read_cho2017_documented(good) == 240

        class StandIn:                                    # not a Cho2017 instance: no lock involved
            subject_list = [1, 2]

            def data_path(self, s):
                return [str(good if s == 1 else bad)]
        assert pdata.documented_trials("F1b", [1], StandIn()) == {1: 240}
        try:
            pdata.documented_trials("F1b", [1, 2], StandIn())
            raise AssertionError("missing n_imagery_trials must stop the run")
        except pdata.ExclusionError:
            pass
    assert pdata.documented_trials("F1a", [3]) == {3: 400}
    assert pdata.documented_trials("F2", [3]) == {3: {0: 200, 1: 200}}


def _fake_data(counts, failed=()):
    """counts: {subject: [trials in session 0, trials in session 1]}"""
    g, s = [], []
    for sub, per in counts.items():
        for sess, n in enumerate(per):
            g += [sub] * n
            s += [sess] * n
    n = len(g)
    return dict(X=np.zeros((n, 2, 4), np.float32), y=np.arange(n) % 2, subject=np.array(g), session=np.array(s),
                run=np.zeros(n, str), fs=250.0, failed={f: "boom" for f in failed})


def test_exclusions():
    d = _fake_data({1: [200, 200], 2: [200, 121], 3: [161, 161], 4: [100, 100]}, failed=[5])
    t = pdata.exclusion_table("F1a", d, pdata.documented_trials("F1a", [1, 2, 3, 4, 5]), [1, 2, 3, 4, 5])
    assert t.set_index("subject").excluded.to_dict() == {1: False, 2: False, 3: False, 4: True, 5: True}
    t2 = pdata.exclusion_table("F2", d, pdata.documented_trials("F2", [1, 2, 3, 4, 5]), [1, 2, 3, 4, 5])
    assert t2.set_index("subject").excluded.to_dict() == {1: False, 2: True, 3: False, 4: True, 5: True}
    assert "session 2" in t2.set_index("subject").reason[2]
    kept = pdata.apply_exclusions(d, t2)
    assert set(kept["subject"]) == {1, 3} and len(kept["X"]) == len(kept["y"]) == 400 + 322
    tc = pdata.exclusion_table("F1b", _fake_data({1: [200], 2: [191]}), {1: 240, 2: 240}, [1, 2])
    assert tc.excluded.tolist() == [False, True]                     # 192 of 240 is the 80 % line


def test_splits():
    d = _fake_data({s: [10, 10] for s in range(1, 55)})
    folds = pdata.make_folds("F1a", d)
    assert len(folds) == 5
    tests = np.concatenate([f["test_subjects"] for f in folds])
    assert sorted(tests) == list(range(1, 55))                       # every subject tested exactly once
    for f in folds:
        tr, va, te = (set(d["subject"][f[k]]) for k in ("train", "val", "test"))
        assert not (tr & va or tr & te or va & te) and len(tr | va | te) == 54
        assert len(va) == round(0.1 * (54 - len(te)))
    again = pdata.make_folds("F1a", d)
    assert all(np.array_equal(a["test"], b["test"]) and np.array_equal(a["val"], b["val"]) for a, b in zip(folds, again))
    w = pdata.make_folds("F2", d)
    assert len(w) == 54
    for f in w:
        assert (d["session"][f["train"]] == 0).all() and (d["session"][f["val"]] == 0).all()
        assert (d["session"][f["test"]] == 1).all() and len(f["val"]) == 2 and len(f["train"]) == 8
    # fresh rng per subject: a subject's split does not depend on the other subjects
    d2 = _fake_data({7: [10, 10], 9: [10, 10]})
    a = {f["fold"]: f for f in pdata.make_folds("F2", d2)}
    b = {f["fold"]: f for f in pdata.make_folds("F2", _fake_data({9: [10, 10]}))}
    assert np.array_equal(a[9]["val"] - a[9]["val"].min(), b[9]["val"] - b[9]["val"].min())
    X = np.random.default_rng(0).standard_normal((30, 3, 50)).astype(np.float32) * np.array([1, 5, 9])[:, None]
    Xn = pdata.normalise(X, np.arange(20))
    assert np.allclose(Xn[:20].std(axis=(0, 2)), 1, atol=1e-5)


def test_synthetic_cells_have_documented_shapes():
    for cell, shape in (("F1a", (62, 1000)), ("F1b", (64, 750)), ("F2", (62, 1000))):
        d = pdata.synthetic_cell(cell, n_subjects=2, trials_per_session=4)
        assert d["X"].shape[1:] == shape
        assert set(d["session"]) == ({0, 1} if cell != "F1b" else {0})


# ----------------------------------------------------------------------------- item 4: runner
from deploy import prereg_run as prun  # noqa: E402


def test_registered_architectures():
    from deploy.mibminet import MIBMInet
    from eeg_project.scripts.models.convnets import ShallowConvNet
    from eeg_project.scripts.models.csoanet2026_v2 import CSOANetV2
    from eeg_project.scripts.models.eegnet import EEGNet
    for ch, t in ((62, 1000), (64, 750)):
        for protocol, e_drop, m_drop in (("cross-subject", 0.25, 0.2), ("within-subject", 0.5, 0.5)):
            e = prun.build_model("eegnet", ch, t, protocol)
            assert isinstance(e, EEGNet) and e.block1[0].kernel_size == (1, 125) and e.block2[0].kernel_size == (1, 31)
            assert {m.p for m in e.modules() if isinstance(m, nn.Dropout)} == {e_drop}
            sh = prun.build_model("shallow", ch, t, protocol)
            assert isinstance(sh, ShallowConvNet) and sh.temporal.kernel_size == (1, 25)
            assert {m.p for m in sh.classifier.modules() if isinstance(m, nn.Dropout)} == {0.5}
            mb = prun.build_model("mibminet", ch, t, protocol)
            assert isinstance(mb, MIBMInet) and (mb.c["nk"], mb.c["nf"], mb.c["p_dropout"]) == (32, 64, m_drop)
            cs = prun.build_model("csoanet", ch, t, protocol)
            assert isinstance(cs, CSOANetV2) and cs.temporal.kernel_size == (1, 125)
            assert (cs.c["gate"], cs.c["bound"], cs.c["act"]) == ("static", True, "relu")
    # own constraints under every recipe
    from deploy.train_physionet_mi import constrain
    for arch in prun.ARCHS:
        m = prun.build_model(arch, 62, 1000, "cross-subject")
        with torch.no_grad():
            for p in m.parameters():
                p.mul_(50)
        constrain(m)
        if arch == "eegnet":
            assert float(m.block1[2].weight.flatten(1).norm(dim=1).max()) <= 1 + 1e-5
            assert float(m.classifier[1].weight.norm(dim=1).max()) <= 0.25 + 1e-5
        elif arch == "csoanet":
            assert float(m.spatial.weight.flatten(1).norm(dim=1).max()) <= 0.25 + 1e-5
        else:
            assert float(next(m.parameters()).abs().max()) > 1


def test_nonfinite_retry():
    calls = []
    real = prun.prec.train_recipe

    def fake(model, recipe, *a, **k):
        calls.append(torch.initial_seed())
        r = real(model, recipe, *a, **{**k, "epochs": 1})
        r["finite"] = len(calls) > 1 if fake.mode == "once" else False
        return r
    X, Y = torch.randn(40, 1, 62, 1000), torch.randint(0, 2, (40,))
    fold = type("F", (), dict(Xtr=X[:30], Ytr=Y[:30], Xva=X[30:], Yva=Y[30:], ch=62, t=1000))()
    try:
        prun.prec.train_recipe = fake
        for mode, finite in (("once", True), ("twice", False)):
            calls.clear()
            fake.mode = mode
            _, res = prun.fit("mibminet", pr.RECIPES["R2"], fold, "cross-subject", 2, torch.device("cpu"))
            assert calls == [2, 102] and res["seed_used"] == 102 and res["retried"] and res["finite"] == finite
    finally:
        prun.prec.train_recipe = real


def test_tuning_never_sees_test_trials():
    d = pdata.synthetic_cell("F2", n_subjects=12, trials_per_session=4)
    folds = pdata.make_folds("F2", d)
    units = prun.tuning_units("F2", d, folds)
    assert [u["fold"] for u in units] == list(range(1, 11))          # the 10 lowest subject IDs
    for u in units:
        assert len(u["test"]) == 0 and (d["session"][np.r_[u["train"], u["val"]]] == 0).all()
    d = pdata.synthetic_cell("F1a", n_subjects=20, trials_per_session=2)
    folds = pdata.make_folds("F1a", d)
    (u,) = prun.tuning_units("F1a", d, folds)
    assert np.array_equal(u["train"], folds[0]["train"]) and np.array_equal(u["val"], folds[0]["val"])
    assert len(u["test"]) == 0
    seen = []
    orig = prun.Fold.test
    prun.Fold.test = lambda self: seen.append(1) or orig(self)
    try:
        with tempfile.TemporaryDirectory() as out:
            best = prun.tune("F1a", d, folds, ["csoanet"], Path(out), "x", torch.device("cpu"), 2, epochs=1)
    finally:
        prun.Fold.test = orig
    assert not seen and best["csoanet"].name in ("T00", "T01")


def _run(out, *extra):
    return prun.main(["--cell", "F2", "--synthetic", "--synthetic-subjects", "3", "--epochs", "1", "--patience", "1",
                      "--t-configs", "2", "--device", "cpu", "--out", str(out), "--archs", "csoanet", "eegnet",
                      *extra])


def test_runner_end_to_end_and_resume():
    import pandas as pd
    with tempfile.TemporaryDirectory() as out:
        csv = _run(out, "--recipes", "R2", "T", "--seeds", "0", "1")
        df = pd.read_csv(csv)
        assert len(df) == 2 * 2 * 3 * 2                                   # arch x recipe x subject x seed
        assert set(df.recipe) == {"R2", "T"} and set(df.t_config.dropna()) <= {"T00", "T01"}
        cols = {"cell", "arch", "recipe", "fold", "subject", "seed", "seed_used", "retried", "bacc_fp32", "bacc_int8",
                "agreement", "train_acc", "val_acc", "selected_epoch", "n_test"}
        assert cols <= set(df.columns)
        assert df.bacc_fp32.between(0, 1).all() and df.bacc_int8.between(0, 1).all()
        assert df[df.arch == "eegnet"].bacc_int8_sf.notna().all()       # spatial-first extra column
        assert (df.n_test == 40).all()                                    # session 2 of each subject
        mtime = Path(csv).stat().st_mtime_ns
        _run(out, "--recipes", "R2", "T", "--seeds", "0", "1")            # resume: nothing left to do
        assert Path(csv).stat().st_mtime_ns == mtime and len(pd.read_csv(csv)) == len(df)
        _run(out, "--recipes", "R2", "R4", "--seeds", "0", "1")           # adds R4 only
        assert len(pd.read_csv(csv)) == len(df) + 12
        prof = pd.read_csv(Path(out) / "F2_synthetic_profiles.csv")
        assert set(prof.form) == {"standard", "spatial-first"} and (prof.macs > 0).all()
        assert (Path(out) / "F2_synthetic_exclusions.csv").exists()
        try:
            prun.main(["--cell", "F2", "--synthetic", "--synthetic-subjects", "3", "--epochs", "2", "--out", out,
                       "--device", "cpu"])
            raise AssertionError("changed settings must stop the run")
        except SystemExit:
            pass


def test_runner_refuses_real_cells_and_overrides():
    with _locked_env(), tempfile.TemporaryDirectory() as out:
        for cell in pdata.CELLS:
            try:
                prun.main(["--cell", cell, "--out", out, "--device", "cpu"])
                raise AssertionError(cell)
            except pdata.ConfirmatoryLocked:
                pass
        for extra in (["--epochs", "3"], ["--seeds", "0"], ["--t-configs", "3"]):
            try:
                prun.main(["--cell", "F1a", "--out", out, *extra])
                raise AssertionError(extra)
            except SystemExit:
                pass
        # a cache file of a confirmatory cell is the dataset itself: reading it is locked too
        np.savez(Path(out) / "cache_Lee2019_MI_F1a.npz", X=np.zeros(1))
        try:
            prun.load_data("F1a", Path(out), False, 0)
            raise AssertionError
        except pdata.ConfirmatoryLocked:
            pass


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(tests)} passed")
