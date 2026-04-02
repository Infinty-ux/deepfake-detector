from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image


REAL_LABEL = 0
FAKE_LABEL = 1

TRAIN_TRANSFORMS = A.Compose([
    A.RandomResizedCrop(224, 224, scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 7)),
        A.ImageCompression(quality_lower=60, quality_upper=95),
        A.GaussNoise(var_limit=(10, 50)),
    ], p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

VAL_TRANSFORMS = A.Compose([
    A.Resize(256, 256),
    A.CenterCrop(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


class DeepfakeDataset(Dataset):
    EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(
        self,
        root: Path,
        split: str = "train",
        transform=None,
    ):
        self.root = Path(root) / split
        self.transform = transform or (TRAIN_TRANSFORMS if split == "train" else VAL_TRANSFORMS)
        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self):
        for label_name, label_idx in [("real", REAL_LABEL), ("fake", FAKE_LABEL)]:
            label_dir = self.root / label_name
            if not label_dir.exists():
                continue
            for f in label_dir.iterdir():
                if f.suffix.lower() in self.EXTENSIONS:
                    self.samples.append((f, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = np.array(Image.open(path).convert("RGB"))
        img = self.transform(image=img)["image"]
        return img, label

    def class_counts(self) -> Tuple[int, int]:
        labels = [s[1] for s in self.samples]
        return labels.count(REAL_LABEL), labels.count(FAKE_LABEL)


def make_weighted_sampler(dataset: DeepfakeDataset) -> WeightedRandomSampler:
    labels = [s[1] for s in dataset.samples]
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts
    sample_weights = torch.tensor([weights[l] for l in labels], dtype=torch.float32)
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def make_dataloaders(
    root: Path,
    batch_size: int = 32,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = DeepfakeDataset(root, split="train")
    val_ds = DeepfakeDataset(root, split="val")

    sampler = make_weighted_sampler(train_ds) if use_weighted_sampler else None

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_dl, val_dl
