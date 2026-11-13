"""
FedMedVision — Labeled Urban Hub Client (Nodes 0 & 1)
Trains on real NIH ChestX-ray14 partitions with:
- FedBN local normalization privacy
- FedProx proximal regularization
- FedProto class-wise prototype alignment
- Green AI adaptive layer freezing & real NVML power tracking
"""
from __future__ import annotations

import io
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import flwr as fl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import ChestXrayConsortiumDataset
from models.convnext_fedbn import FedMedConvNeXt
from models.fedproto_loss import (
    compute_local_prototypes,
    fedproto_alignment_loss,
    fedprox_loss,
)
from models.green_freezer import EnergyMonitor, GreenAdaptiveFreezer
from utils.config_loader import load_config
from utils.logger import get_logger

logger = get_logger("fedmedvision.fl.hub")


class LabeledHubClient(fl.client.NumPyClient):
    """
    Flower client representing a labeled urban hospital hub (e.g. Node 0 or Node 1).
    """

    def __init__(
        self,
        node_id: str,
        split_csv: str | Path,
        config: Any | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.node_id = node_id
        self.split_csv = Path(split_csv)
        self.config = config or load_config()
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # Model instantiation
        self.num_classes = len(self.config.consortium.pathologies)
        self.model = FedMedConvNeXt(
            num_classes=self.num_classes,
            pretrained=False,
            dropout_p=self.config.training.dropout_p,
        ).to(self.device)

        # Dataset & DataLoader
        self.dataset = ChestXrayConsortiumDataset(
            self.split_csv,
            mode="labeled",
            pathologies=self.config.consortium.pathologies,
            image_size=self.config.consortium.image_size,
        )
        batch_size = min(self.config.training.batch_size, 32)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
        )

        # Loss and weighting
        df = pd.read_csv(self.split_csv)
        pos_weights = []
        for p in self.config.consortium.pathologies:
            pos_c = df[p].sum() if p in df else 1
            neg_c = len(df) - pos_c
            w = max(1.0, float(neg_c) / max(1.0, float(pos_c)))
            pos_weights.append(min(w, 10.0))
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weights, device=self.device))

        # Optimizer with Cosine Annealing (Hub LR = 3e-4)
        hub_lr = 3e-4
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=hub_lr,
            weight_decay=self.config.training.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=15,
            eta_min=1e-5,
        )

        # Green AI & Energy Tracking
        green_cfg = self.config.security.green_ai
        self.freezer = GreenAdaptiveFreezer(
            grad_norm_tol=green_cfg.get("grad_norm_tol", 1e-3),
            grad_norm_patience=green_cfg.get("grad_norm_patience", 3),
        )
        self.energy_monitor = EnergyMonitor(
            co2_kg_per_kwh=green_cfg.get("co2_kg_per_kwh", 0.42)
        )

    def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
        """Return shared parameters (excluding local FedBN normalization layers)."""
        shared_dict = self.model.get_shared_state_dict()
        return [v.cpu().numpy() for v in shared_dict.values()]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        """Load shared parameters into local model, keeping FedBN weights private."""
        shared_keys = list(self.model.get_shared_state_dict().keys())
        shared_state_dict = {
            k: torch.tensor(v, device=self.device) for k, v in zip(shared_keys, parameters)
        }
        self.model.load_shared_state_dict(shared_state_dict)

    def fit(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[list[np.ndarray], int, dict[str, Any]]:
        self.set_parameters(parameters)
        shared_keys = list(self.model.get_shared_state_dict().keys())
        global_shared_params = {
            k: torch.tensor(v, device=self.device) for k, v in zip(shared_keys, parameters)
        }

        # Apply frozen stages if received from server
        frozen_stages_str = config.get("frozen_stages", "")
        if frozen_stages_str:
            frozen_stages = set(frozen_stages_str.split(","))
            self.freezer.apply_frozen_stages(self.model, frozen_stages)

        # Parse global prototypes
        global_prototypes: dict[int, torch.Tensor] = {}
        for c in range(self.num_classes):
            key = f"proto_{c}"
            if key in config:
                arr = np.array(config[key], dtype=np.float32)
                global_prototypes[c] = torch.tensor(arr, device=self.device)

        mu = float(config.get("fedprox_mu", getattr(self.config.training, "fedprox_mu", 0.01)))
        lambda_proto = float(config.get("proto_lambda", getattr(self.config.training, "fedproto_lambda", 0.1)))
        local_epochs = int(config.get("local_epochs", 1))

        # DP-SGD Privacy Wrapper if requested
        enable_dp = bool(config.get("enable_dp", False))
        dp_wrapper = None
        if enable_dp:
            from security.dp_privacy import DPPrivacyWrapper
            dp_wrapper = DPPrivacyWrapper(
                model=self.model,
                optimizer=self.optimizer,
                dataloader=self.dataloader,
                target_epsilon=float(config.get("target_epsilon", 8.0)),
                target_delta=float(config.get("target_delta", 1e-5)),
                max_grad_norm=float(config.get("max_grad_norm", 1.0)),
            )

        self.energy_monitor.start()
        self.model.train()
        total_loss = 0.0

        all_features = []
        all_targets = []
        # When DP is active, standard autocast scaling interacts with hooks; use clean train loop
        use_amp = (self.device.type == "cuda" and not enable_dp)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        for _ in range(local_epochs):
            for images, targets, _ in self.dataloader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                if dp_wrapper:
                    dp_wrapper.private_optimizer.zero_grad()
                else:
                    self.optimizer.zero_grad()

                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits, features = self.model.forward_with_features(images)
                    loss_bce = self.criterion(logits, targets)
                    loss_prox = fedprox_loss(self.model, global_shared_params, mu=mu)
                    loss_proto = fedproto_alignment_loss(features, targets, global_prototypes)
                    loss = loss_bce + loss_prox + (lambda_proto * loss_proto)

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if dp_wrapper:
                        dp_wrapper.step()
                        dp_wrapper.private_optimizer.zero_grad()
                    else:
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                total_loss += loss.item() * len(images)
                all_features.append(features.detach().cpu())
                all_targets.append(targets.detach().cpu())

        # Compute updated local prototypes
        cat_features = torch.cat(all_features, dim=0)
        cat_targets = torch.cat(all_targets, dim=0)
        local_protos, proto_counts = compute_local_prototypes(
            cat_features, cat_targets, num_classes=self.num_classes
        )

        # Record gradient norms for Green AI
        stage_grad_norms = self.freezer.record_stage_gradients(self.model)
        freezing_stats = self.freezer.get_freezing_stats(self.model)
        energy_stats = self.energy_monitor.stop()

        # Build metrics dict
        metrics: dict[str, Any] = {
            "node_id": self.node_id,
            "loss": float(total_loss / max(1, len(self.dataset))),
            "energy_wh": energy_stats["energy_wh"],
            "co2_grams": energy_stats["co2_grams"],
            "frozen_ratio": freezing_stats["frozen_parameter_ratio"],
        }
        if dp_wrapper:
            metrics["privacy_budget"] = dp_wrapper.get_privacy_budget()
        for s, norm in stage_grad_norms.items():
            metrics[f"grad_norm_{s}"] = float(norm)

        # Serialize local prototypes into metrics
        for c, p_tensor in local_protos.items():
            metrics[f"proto_{c}"] = p_tensor.numpy().tolist()
            metrics[f"proto_count_{c}"] = proto_counts.get(c, 0)

        updated_params = self.get_parameters(config)
        return updated_params, len(self.dataset), metrics

