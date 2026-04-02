import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from model import DeepfakeDetector, build_model
from dataset import make_dataloaders
from evaluator import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None

    def step(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def train_one_epoch(
    model: DeepfakeDetector,
    loader: DataLoader,
    optimizer,
    criterion,
    scaler: GradScaler,
    device: str,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        if (batch_idx + 1) % 50 == 0:
            logger.info(f"Epoch {epoch} [{batch_idx+1}/{len(loader)}] loss={loss.item():.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def validate(
    model: DeepfakeDetector,
    loader: DataLoader,
    criterion,
    device: str,
) -> Dict:
    model.eval()
    all_probs, all_labels = [], []
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels_dev = labels.float().to(device, non_blocking=True)
        with autocast():
            logits = model(images)
            loss = criterion(logits, labels_dev)
        total_loss += loss.item()
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.numpy().tolist())

    metrics = compute_metrics(all_labels, all_probs)
    metrics["val_loss"] = total_loss / len(loader)
    return metrics


def train(
    data_dir: str = "data",
    output_dir: str = "checkpoints",
    backbone: str = "efficientnet_b4",
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    device: str = "cuda",
    resume: Optional[str] = None,
):
    device = device if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on {device}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_dl, val_dl = make_dataloaders(Path(data_dir), batch_size=batch_size)
    logger.info(f"Train: {len(train_dl.dataset)} | Val: {len(val_dl.dataset)}")

    model = build_model(backbone=backbone, pretrained=True, device=device, checkpoint_path=resume)

    pos_weight = torch.tensor([1.0], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = GradScaler()
    early_stop = EarlyStopping(patience=7)

    best_auc = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        if epoch == model.freeze_backbone_epochs + 1:
            model.unfreeze_backbone()
            logger.info("Backbone unfrozen")

        train_loss = train_one_epoch(model, train_dl, optimizer, criterion, scaler, device, epoch)
        val_metrics = validate(model, val_dl, criterion, device)
        scheduler.step()

        auc = val_metrics["auc"]
        logger.info(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['val_loss']:.4f} | "
            f"AUC={auc:.4f} | F1={val_metrics['f1']:.4f}"
        )

        epoch_entry = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(epoch_entry)

        if auc > best_auc:
            best_auc = auc
            torch.save({"epoch": epoch, "model": model.state_dict(), "metrics": val_metrics},
                       output_path / "best.pt")
            logger.info(f"  → Best saved (AUC={best_auc:.4f})")

        torch.save({"epoch": epoch, "model": model.state_dict(), "metrics": val_metrics},
                   output_path / "last.pt")

        if early_stop.step(auc):
            logger.info(f"Early stop at epoch {epoch}")
            break

    with open(output_path / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"Training done. Best AUC: {best_auc:.4f}")
    return history


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output-dir", default="checkpoints")
    p.add_argument("--backbone", default="efficientnet_b4")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume")
    args = p.parse_args()
    train(**vars(args))
