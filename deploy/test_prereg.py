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


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(tests)} passed")
