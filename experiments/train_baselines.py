"""
FedMedVision — Baseline Training & Evaluation (Lower & Upper Bounds)
Trains:
  1. Isolated Local Hospital Baseline (Hub 0 only) -> results/checkpoints/isolated_hub_0.pt
  2. Centralized Pooled Oracle Baseline (Hub 0 + Hub 1) -> results/checkpoints/centralized_oracle.pt
Evaluates on:
  - Consortium Validation/Test Set (val_test.csv)
  - External Zero-Shot Test Cohort (external_test.csv)
Outputs results to results/metrics/phase1_baselines.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import ChestXrayConsortiumDataset
from models.convnext_fedbn import FedMedConvNeXt
from models.metrics import compute_ece, compute_multilabel_metrics
from utils.config_loader import load_config
from utils.logger import get_logger, log_system_event
from utils.reproducibility import get_device_info, seed_everything

logger = get_logger("fedmedvision.baselines")


def train_single_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
) -> float:
    model.train()
    total_loss = 0.0
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))

    for images, targets, _ in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, targets)

        if use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * len(images)

    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    class_names: list[str],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    all_targets = []
    all_probs = []

    for images, targets, _ in dataloader:
        images = images.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits)

        all_targets.append(targets.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0)
    y_probs = np.concatenate(all_probs, axis=0)

    metrics = compute_multilabel_metrics(y_true, y_probs, class_names)
    metrics["ece"] = compute_ece(y_true, y_probs)
    return metrics


def run_experiment(
    exp_name: str,
    train_csv: Path,
    val_csv: Path,
    ext_csv: Path,
    checkpoint_path: Path,
    config: Any,
    epochs: int,
    device_info: Any,
    pretrained: bool = False,
) -> dict[str, Any]:
    logger.info(f"=== Starting Training: {exp_name} ===")
    device = device_info.device
    batch_size = min(config.training.batch_size, 32)
    pathologies = config.consortium.pathologies

    # Datasets & Loaders
    train_dataset = ChestXrayConsortiumDataset(
        train_csv,
        mode="labeled",
        pathologies=pathologies,
        image_size=config.consortium.image_size,
    )
    val_dataset = ChestXrayConsortiumDataset(
        val_csv,
        mode="eval",
        pathologies=pathologies,
        image_size=config.consortium.image_size,
    )
    ext_dataset = ChestXrayConsortiumDataset(
        ext_csv,
        mode="eval",
        pathologies=pathologies,
        image_size=config.consortium.image_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # safe on Windows / low core limits
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    ext_loader = DataLoader(ext_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Class weighting for BCE loss to address chest X-ray imbalance
    train_df = pd.read_csv(train_csv)
    pos_weights = []
    for p in pathologies:
        pos_count = train_df[p].sum() if p in train_df else 1
        neg_count = len(train_df) - pos_count
        weight = max(1.0, float(neg_count) / max(1.0, float(pos_count)))
        pos_weights.append(min(weight, 10.0))  # clip extreme weights
    pos_weight_tensor = torch.tensor(pos_weights, dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # Model instantiation
    model = FedMedConvNeXt(
        num_classes=len(pathologies),
        pretrained=pretrained,
        dropout_p=config.training.dropout_p,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )

    start_time = time.time()
    for ep in range(1, epochs + 1):
        loss = train_single_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            use_amp=config.training.use_amp,
        )
        logger.info(f"[{exp_name}] Epoch {ep:02d}/{epochs:02d} - Loss: {loss:.4f}")

    train_time_sec = round(time.time() - start_time, 2)

    # Save checkpoint
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "pathologies": pathologies,
            "exp_name": exp_name,
            "epochs": epochs,
        },
        checkpoint_path,
    )
    logger.info(f"Saved checkpoint to {checkpoint_path}")

    # Evaluate
    val_metrics = evaluate_model(model, val_loader, pathologies, device)
    ext_metrics = evaluate_model(model, ext_loader, pathologies, device)

    return {
        "exp_name": exp_name,
        "epochs": epochs,
        "train_time_sec": train_time_sec,
        "train_samples": len(train_dataset),
        "validation_metrics": val_metrics,
        "external_test_metrics": ext_metrics,
        "checkpoint_path": str(checkpoint_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="FedMedVision Phase 1 Reference Baselines")
    parser.add_argument("--mode", choices=["real", "synthetic"], default="synthetic", help="Baseline training data mode")
    parser.add_argument("--synthetic-dryrun", action="store_true", help="Run with synthetic dry-run data")
    parser.add_argument("--epochs", type=int, default=2, help="Number of baseline training epochs")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights")
    args = parser.parse_args()

    config = load_config()
    seed_everything(config.consortium.seeds[0] if config.consortium.seeds else 42)
    device_info = get_device_info()

    log_system_event(
        logger,
        "baselines_training_start",
        mode=args.mode,
        device=device_info.device_name,
        epochs=args.epochs,
    )

    splits_dir = config.project_root / "data" / "splits"
    hub_0_csv = splits_dir / "hub_0.csv"
    hub_1_csv = splits_dir / "hub_1.csv"
    val_csv = splits_dir / "val_test.csv"
    ext_csv = splits_dir / "external_test.csv"

    # Verify input splits exist or auto-create if requested
    if not hub_0_csv.exists() or not val_csv.exists():
        if args.synthetic_dryrun or args.mode == "synthetic":
            from data.prepare_consortium import build_synthetic_consortium
            data_proc = config.project_root / "data" / "processed"
            build_synthetic_consortium(data_proc, splits_dir, config)
        else:
            from data.prepare_consortium import prepare_real_consortium
            data_proc = config.project_root / "data" / "processed"
            prepare_real_consortium(data_proc, splits_dir, config)

    checkpoints_dir = config.project_root / "results" / "checkpoints"
    metrics_dir = config.project_root / "results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # 1. Baseline 1: Isolated Local Hospital (Hub 0 only)
    res_hub_0 = run_experiment(
        exp_name="isolated_hub_0",
        train_csv=hub_0_csv,
        val_csv=val_csv,
        ext_csv=ext_csv,
        checkpoint_path=checkpoints_dir / "isolated_hub_0.pt",
        config=config,
        epochs=args.epochs,
        device_info=device_info,
        pretrained=args.pretrained,
    )

    # 2. Baseline 2: Centralized Pooled Oracle (Hub 0 + Hub 1)
    pooled_df = pd.concat([pd.read_csv(hub_0_csv), pd.read_csv(hub_1_csv)], ignore_index=True)
    pooled_csv = splits_dir / "centralized_pooled_train.csv"
    pooled_df.to_csv(pooled_csv, index=False)

    res_oracle = run_experiment(
        exp_name="centralized_oracle",
        train_csv=pooled_csv,
        val_csv=val_csv,
        ext_csv=ext_csv,
        checkpoint_path=checkpoints_dir / "centralized_oracle.pt",
        config=config,
        epochs=args.epochs,
        device_info=device_info,
        pretrained=args.pretrained,
    )

    # Consolidate results JSON
    baselines_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "device": str(device_info.device),
            "gpu_name": device_info.device_name,
            "vram_gb": device_info.total_vram_gb,
        },
        "pathologies": config.consortium.pathologies,
        "baselines": {
            "isolated_hub_0": res_hub_0,
            "centralized_oracle": res_oracle,
        },
    }

    metrics_file = metrics_dir / "phase1_baselines.json"
    with metrics_file.open("w", encoding="utf-8") as f:
        json.dump(baselines_summary, f, indent=2)

    logger.info(f"✅ Phase 1 baselines complete. Summary written to {metrics_file}")
    print("\n========================================================")
    print("  FedMedVision Phase 1 Baselines Summary")
    print("========================================================")
    print(f"Isolated Hub 0     -> Val Macro-AUROC: {res_hub_0['validation_metrics']['macro_auroc']:.4f} | ECE: {res_hub_0['validation_metrics']['ece']:.4f}")
    print(f"Centralized Oracle -> Val Macro-AUROC: {res_oracle['validation_metrics']['macro_auroc']:.4f} | ECE: {res_oracle['validation_metrics']['ece']:.4f}")
    print(f"Results saved to: {metrics_file}")
    print("========================================================\n")


if __name__ == "__main__":
    main()
