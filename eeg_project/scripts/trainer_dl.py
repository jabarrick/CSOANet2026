from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def _to_device(batch, device):
    x, y = batch
    return x.to(device), y.to(device)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    out_dir: Path,
    device: torch.device,
) -> Tuple[nn.Module, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

    for ep in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for batch in train_loader:
            x, y = _to_device(batch, device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * x.size(0)
            preds = logits.argmax(dim=1)
            correct += int((preds == y).sum().item())
            total += int(x.size(0))
        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)

        model.eval()
        with torch.no_grad():
            v_loss, v_correct, v_total = 0.0, 0, 0
            for batch in val_loader:
                x, y = _to_device(batch, device)
                logits = model(x)
                loss = criterion(logits, y)
                v_loss += float(loss.item()) * x.size(0)
                preds = logits.argmax(dim=1)
                v_correct += int((preds == y).sum().item())
                v_total += int(x.size(0))
        val_loss = v_loss / max(1, v_total)
        val_acc = v_correct / max(1, v_total)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), out_dir / "best_model.pt")

        print(f"Epoch {ep}/{epochs} - train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

    # 曲线图
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].plot(history["train_loss"], label="train")
    ax[0].plot(history["val_loss"], label="val")
    ax[0].set_title("Loss")
    ax[0].legend()
    ax[1].plot(history["train_acc"], label="train")
    ax[1].plot(history["val_acc"], label="val")
    ax[1].set_title("Accuracy")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss_acc_curve.png", dpi=160)
    plt.close(fig)

    return model, history


def evaluate_and_plot(model: nn.Module, loader: DataLoader, out_dir: Path, device: torch.device) -> Tuple[np.ndarray, str]:
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            p = logits.argmax(dim=1).cpu().numpy()
            preds.append(p)
            ys.append(y.numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Confusion Matrix (DL)")
    ax.set_xlabel("Pred")
    ax.set_ylabel("True")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "confusion_matrix_dl.png", dpi=160)
    plt.close(fig)

    (out_dir / "metrics_dl.txt").write_text(report, encoding="utf-8")
    return cm, report


