"""
FedMedVision — Chest X-Ray Consortium PyTorch Dataset & Medical Augmentations
Provides labeled clinical augmentations, dual weak/strong unlabeled augmentations
for semi-supervised training, and deterministic evaluation transforms.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from utils.config_loader import load_config


def get_medical_transforms(
    image_size: int = 224,
    mode: Literal["labeled", "unlabeled_weak", "unlabeled_strong", "eval"] = "labeled",
) -> A.Compose:
    """
    Build domain-specific albumentations transforms for chest radiographs.
    """
    # Standard ImageNet normalization (compatible with pretrained timm weights)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if mode == "eval":
        return A.Compose(
            [
                A.Resize(height=image_size, width=image_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )

    if mode == "labeled":
        return A.Compose(
            [
                A.Resize(height=image_size, width=image_size),
                A.HorizontalFlip(p=0.5),
                A.Affine(
                    scale=(0.92, 1.08),
                    translate_percent=(-0.05, 0.05),
                    rotate=(-10, 10),
                    p=0.5,
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )

    if mode == "unlabeled_weak":
        return A.Compose(
            [
                A.Resize(height=image_size, width=image_size),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )

    if mode == "unlabeled_strong":
        return A.Compose(
            [
                A.Resize(height=image_size, width=image_size),
                A.HorizontalFlip(p=0.5),
                A.Affine(
                    scale=(0.90, 1.10),
                    translate_percent=(-0.08, 0.08),
                    rotate=(-15, 15),
                    p=0.6,
                ),
                # Medical contrast enhancement & structural cutout
                A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.6),
                A.CoarseDropout(
                    num_holes_range=(1, 4),
                    hole_height_range=(16, 40),
                    hole_width_range=(16, 40),
                    p=0.5,
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )

    raise ValueError(f"Unknown transform mode: {mode}")


class ChestXrayConsortiumDataset(Dataset):
    """
    Consortium Dataset loader supporting:
      - mode='labeled': returns (image_tensor, multi_hot_labels, sample_id)
      - mode='unlabeled_dual': returns (weak_aug_tensor, strong_aug_tensor, sample_id)
      - mode='eval': returns (image_tensor, multi_hot_labels, sample_id) with deterministic transforms
    """

    def __init__(
        self,
        df_or_csv: pd.DataFrame | Path | str,
        data_root: Path | str | None = None,
        mode: Literal["labeled", "unlabeled_dual", "eval"] = "labeled",
        pathologies: list[str] | None = None,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.image_size = image_size

        if isinstance(df_or_csv, (str, Path)):
            self.df = pd.read_csv(df_or_csv)
        else:
            self.df = df_or_csv.copy()

        if data_root is None:
            cfg = load_config()
            self.data_root = cfg.project_root / "data" / "processed"
        else:
            self.data_root = Path(data_root)

        if pathologies is None:
            cfg = load_config()
            self.pathologies = cfg.consortium.pathologies
        else:
            self.pathologies = pathologies

        # Build transforms
        if self.mode == "labeled":
            self.transform = get_medical_transforms(image_size=image_size, mode="labeled")
        elif self.mode == "unlabeled_dual":
            self.transform_weak = get_medical_transforms(image_size=image_size, mode="unlabeled_weak")
            self.transform_strong = get_medical_transforms(image_size=image_size, mode="unlabeled_strong")
        elif self.mode == "eval":
            self.transform = get_medical_transforms(image_size=image_size, mode="eval")
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, rel_or_abs_path: str) -> np.ndarray:
        p = Path(rel_or_abs_path)
        if not p.is_absolute():
            p = self.data_root / p

        if not p.exists():
            raise FileNotFoundError(f"Chest X-ray image not found at: {p}")

        with Image.open(p) as img:
            rgb_img = img.convert("RGB")
            return np.array(rgb_img)

    def __getitem__(self, idx: int) -> tuple[Any, ...]:
        row = self.df.iloc[idx]
        sample_id = str(row.get("sample_id", f"sample_{idx}"))
        image_np = self._load_image(str(row["image_path"]))

        # Extract multi-hot label vector if columns exist
        labels_list = []
        for p in self.pathologies:
            if p in row:
                labels_list.append(float(row[p]))
            else:
                labels_list.append(0.0)
        labels_tensor = torch.tensor(labels_list, dtype=torch.float32)

        if self.mode in ("labeled", "eval"):
            augmented = self.transform(image=image_np)
            tensor_img = augmented["image"]
            return tensor_img, labels_tensor, sample_id

        elif self.mode == "unlabeled_dual":
            weak_aug = self.transform_weak(image=image_np)["image"]
            strong_aug = self.transform_strong(image=image_np)["image"]
            return weak_aug, strong_aug, sample_id

        raise RuntimeError(f"Invalid dataset mode: {self.mode}")
