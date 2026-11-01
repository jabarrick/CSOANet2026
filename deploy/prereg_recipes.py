"""Registered training recipes R1-R4 and the condition-T search space (OSF 8st47).

    from deploy.prereg_recipes import RECIPES, sample_t_configs, train_recipe
    result = train_recipe(model, RECIPES["R1"], Xtr, Ytr, Xva, Yva, nc, protocol="cross-subject", dev=dev)

E (epochs) and B (batch) come from the protocol: cross-subject 100 / 64, within-subject 500 / 32.
A recipe can fix its own batch (R3: 16 in every cell).  Every architecture keeps its own weight
constraints under every recipe (train_physionet_mi.constrain).  See deploy/PREREG_DECISIONS.md for
the clarifications Yu gave on R4, T and non-finite losses.
"""
import math
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTOCOL = {"cross-subject": dict(epochs=100, batch=64, patience=15),
            "within-subject": dict(epochs=500, batch=32, patience=80)}
AUGMENTATIONS = ("mixup", "noise", "shift", "crop", "ch_drop")
AUG_PARAMS = dict(mixup_alpha=0.2, noise=0.02, shift=5, crop=0.9, ch_drop=0.1)   # R1's values


@dataclass(frozen=True)
class Recipe:
    name: str
    optimizer: str = "adam"          # "adam" (no weight decay) or "adamw"
    lr: float = 1e-3
    weight_decay: float = 0.0        # AdamW only
    eps: float = 1e-8
    schedule: str = "constant"       # "constant", "cosine" (to lr / 100 over E), "step" (x0.1 at 40 % / 80 % of E)
    batch: int = None                # None: the protocol's B
    mixup: bool = False
    noise: bool = False
    shift: bool = False
    crop: bool = False
    ch_drop: bool = False
    label_smoothing: float = 0.0
    selection: str = "val_loss"      # "val_loss": all E epochs, lowest val loss; "val_acc": early stopping on val accuracy

    def augmentations(self):
        return {k: getattr(self, k) for k in AUGMENTATIONS}

    def as_dict(self):
        return asdict(self)


_R1 = Recipe("R1", optimizer="adamw", lr=1e-3, weight_decay=1e-3, schedule="cosine",
             mixup=True, noise=True, shift=True, crop=True, ch_drop=True, label_smoothing=0.1, selection="val_acc")
RECIPES = {
    "R1": _R1,
    "R2": Recipe("R2", optimizer="adam", lr=1e-3, schedule="constant", selection="val_loss"),
    "R3": Recipe("R3", optimizer="adam", lr=1e-2, eps=1e-7, schedule="step", batch=16, selection="val_loss"),
    # R1 without the augmentation package; label smoothing counts as part of it (Yu, 2026-11-01)
    "R4": replace(_R1, name="R4", mixup=False, noise=False, shift=False, crop=False, ch_drop=False,
                  label_smoothing=0.0),
}

# ------------------------------------------------------------------ condition T
T_OPTIMIZERS = (("adam", 0.0), ("adamw", 0.0), ("adamw", 1e-4), ("adamw", 1e-3))
T_SCHEDULES = ("constant", "cosine", "step")
T_BATCHES = (16, 32, 64)
T_LABEL_SMOOTHING = (0.0, 0.1)
T_SELECTION = ("val_loss", "val_acc")
T_LR = (1e-4, 1e-2)
N_T_CONFIGS = 24


def sample_t_configs(n=N_T_CONFIGS, seed=0):
    """The n random configurations of condition T.  One draw per cell, shared by all architectures."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        opt, wd = T_OPTIMIZERS[int(rng.integers(len(T_OPTIMIZERS)))]
        lr = float(10 ** rng.uniform(math.log10(T_LR[0]), math.log10(T_LR[1])))
        sched = T_SCHEDULES[int(rng.integers(len(T_SCHEDULES)))]
        batch = T_BATCHES[int(rng.integers(len(T_BATCHES)))]
        augs = {k: bool(rng.integers(2)) for k in AUGMENTATIONS}
        ls = T_LABEL_SMOOTHING[int(rng.integers(len(T_LABEL_SMOOTHING)))]
        sel = T_SELECTION[int(rng.integers(len(T_SELECTION)))]
        out.append(Recipe(f"T{i:02d}", optimizer=opt, lr=lr, weight_decay=wd, schedule=sched, batch=batch,
                          label_smoothing=ls, selection=sel, **augs))
    return out


def recipe_from_dict(d):
    return Recipe(**d)


# ------------------------------------------------------------------ augmentation
def augment(X, Y, nc, recipe, p=AUG_PARAMS):
    """R1's augmentation with each part switchable.  With every part on it draws the same random numbers
    in the same order as train_physionet_mi.augment, so R1 reproduces the pilot recipe exactly."""
    B, _, C, T = X.shape
    if recipe.noise:
        X = X + torch.randn_like(X) * p["noise"]
    if recipe.shift:
        s = int(torch.randint(-p["shift"], p["shift"] + 1, (1,)))
        if s:
            X = torch.roll(X, s, dims=-1)
    if recipe.crop:
        cl = int(T * p["crop"])
        st = int(torch.randint(0, T - cl + 1, (1,)))
        out = torch.zeros_like(X)
        out[..., :cl] = X[..., st:st + cl]
        X = out
    if recipe.ch_drop:
        X = X * (torch.rand(B, 1, C, 1, device=X.device) > p["ch_drop"]).float()
    ls = recipe.label_smoothing
    Y1 = F.one_hot(Y, nc).float() * (1 - ls) + ls / nc
    if recipe.mixup:
        lam = torch.distributions.Beta(p["mixup_alpha"], p["mixup_alpha"]).sample().to(X.device)
        perm = torch.randperm(B, device=X.device)
        X, Y1 = lam * X + (1 - lam) * X[perm], lam * Y1 + (1 - lam) * Y1[perm]
    return X, Y1


# ------------------------------------------------------------------ training
def make_optimizer(model, recipe, epochs):
    if recipe.optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=recipe.lr, eps=recipe.eps)
    elif recipe.optimizer == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=recipe.lr, eps=recipe.eps, weight_decay=recipe.weight_decay)
    else:
        raise ValueError(recipe.optimizer)
    if recipe.schedule == "constant":
        sch = None
    elif recipe.schedule == "cosine":
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=recipe.lr * 0.01)
    elif recipe.schedule == "step":
        sch = torch.optim.lr_scheduler.MultiStepLR(opt, [int(0.4 * epochs), int(0.8 * epochs)], 0.1)
    else:
        raise ValueError(recipe.schedule)
    return opt, sch


@torch.no_grad()
def logits_of(model, X, dev, bs=256):
    model.eval()
    return torch.cat([model(X[i:i + bs].to(dev)).float().cpu() for i in range(0, len(X), bs)])


def balanced_accuracy(y_true, y_pred):
    """Mean per-class recall over the classes present in y_true."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean([(y_pred[y_true == c] == c).mean() for c in np.unique(y_true)]))


def train_recipe(model, recipe, Xtr, Ytr, Xva, Yva, nc, protocol, dev, constrain=None, epochs=None, patience=None):
    """Train `model` with `recipe` and load the selected checkpoint.

    Returns a dict: val_acc / val_loss / val_bacc at the selected checkpoint, selected_epoch, epochs_run,
    train_acc (un-augmented, all training trials, selected checkpoint) and finite (False if a training loss
    was non-finite; training stops at that step).  `epochs` / `patience` overrides are for smoke tests only.
    """
    if constrain is None:
        from deploy.train_physionet_mi import constrain
    P = PROTOCOL[protocol]
    E = epochs or P["epochs"]
    patience = patience or P["patience"]
    batch = recipe.batch or P["batch"]
    opt, sch = make_optimizer(model, recipe, E)
    best_state, best_key, wait, sel_ep, finite, ep = None, None, 0, 0, True, -1
    sel = dict(val_acc=float("nan"), val_loss=float("nan"), val_bacc=float("nan"))
    for ep in range(E):
        model.train()
        perm = torch.randperm(len(Ytr))
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            if len(idx) < 2:
                continue
            xb, yb = augment(Xtr[idx].to(dev), Ytr[idx].to(dev), nc, recipe)
            opt.zero_grad(set_to_none=True)
            loss = -(yb * F.log_softmax(model(xb).float(), 1)).sum(1).mean()
            if not torch.isfinite(loss):
                finite = False
                break
            loss.backward()
            opt.step()
            constrain(model)
        if not finite:
            break
        if sch is not None:
            sch.step()
        lg = logits_of(model, Xva, dev)
        vl, pred = float(F.cross_entropy(lg, Yva)), lg.argmax(1).numpy()
        va = float((pred == Yva.numpy()).mean())
        key = vl if recipe.selection == "val_loss" else -va          # lower is better
        if best_key is None or key < best_key:
            best_key = key
            sel = dict(val_acc=va, val_loss=vl, val_bacc=balanced_accuracy(Yva.numpy(), pred))
            best_state, sel_ep, wait = {k: v.detach().clone() for k, v in model.state_dict().items()}, ep + 1, 0
        elif recipe.selection == "val_acc":
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    pred = logits_of(model, Xtr, dev).argmax(1).numpy()
    return dict(sel, selected_epoch=sel_ep, epochs_run=ep + 1 if finite else ep,
                train_acc=float((pred == Ytr.numpy()).mean()), finite=finite)
