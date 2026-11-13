"""
FedMedVision — Unlabeled Regional / Rural Clinic Client (Nodes 2, 3, 4, 5)
Semi-supervised federated client featuring:
- Real CheXpert dual-view unlabeled loader (weak & strong augmentations)
- MC-Dropout predictive uncertainty gating
- Masked consistency loss + FedProx
- Energy consumption & CO2 tracking
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import ChestXrayConsortiumDataset
from models.convnext_fedbn import FedMedConvNeXt
from models.fedproto_loss import fedprox_loss
from models.green_freezer import EnergyMonitor, GreenAdaptiveFreezer
from utils.config_loader import load_config
from utils.logger import get_logger

logger = get_logger("fedmedvision.fl.clinic")


class UnlabeledClinicClient(fl.client.NumPyClient):
    """
    Flower client representing an unlabeled clinic (e.g. Node 2, 3, 4, or 5).
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

        self.num_classes = len(self.config.consortium.pathologies)
        self.model = FedMedConvNeXt(
            num_classes=self.num_classes,
            pretrained=False,
            dropout_p=self.config.training.dropout_p,
        ).to(self.device)

        # Unlabeled Dual-View Dataset
        self.dataset = ChestXrayConsortiumDataset(
            self.split_csv,
            mode="unlabeled_dual",
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

        self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        # Optimizer with Cosine Annealing (Clinic LR = 1e-4)
        clinic_lr = 1e-4
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=clinic_lr,
            weight_decay=self.config.training.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=15,
            eta_min=1e-5,
        )

        green_cfg = self.config.security.green_ai
        self.freezer = GreenAdaptiveFreezer(
            grad_norm_tol=green_cfg.get("grad_norm_tol", 1e-3),
            grad_norm_patience=green_cfg.get("grad_norm_patience", 3),
        )
        self.energy_monitor = EnergyMonitor(
            co2_kg_per_kwh=green_cfg.get("co2_kg_per_kwh", 0.42)
        )

    def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
        shared_dict = self.model.get_shared_state_dict()
        return [v.cpu().numpy() for v in shared_dict.values()]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
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

        # Apply frozen stages if received
        frozen_stages_str = config.get("frozen_stages", "")
        if frozen_stages_str:
            frozen_stages = set(frozen_stages_str.split(","))
            self.freezer.apply_frozen_stages(self.model, frozen_stages)

        mu = float(config.get("fedprox_mu", getattr(self.config.training, "fedprox_mu", 0.01)))
        semi_cfg = getattr(self.config.training, "semi_supervised", {})
        default_lambda_u = semi_cfg.get("unsup_lambda", 0.5) if isinstance(semi_cfg, dict) else 0.5
        
        # Supervised Warmup Schedule:
        # Rounds 1-3: lambda_u = 0.0 (prevents early noisy pseudo-labels from corrupting representations)
        # Rounds 4-10: linear ramp from 0.15 to 0.50
        server_round = int(config.get("server_round", 1))
        if server_round <= 3:
            lambda_u = 0.0
        elif server_round <= 10:
            lambda_u = 0.15 + (0.50 - 0.15) * ((server_round - 4) / 6.0)
        else:
            lambda_u = float(config.get("lambda_u", default_lambda_u))

        local_epochs = int(config.get("local_epochs", 1))

        self.energy_monitor.start()
        total_loss = 0.0
        total_pseudo_accepted = 0
        total_pseudo_candidates = 0

        scaler = torch.amp.GradScaler("cuda", enabled=(self.device.type == "cuda"))

        for _ in range(local_epochs):
            for weak_imgs, strong_imgs, _ in self.dataloader:
                weak_imgs = weak_imgs.to(self.device)
                strong_imgs = strong_imgs.to(self.device)

                # 1. MC-Dropout Uncertainty Gating on weak image
                with torch.no_grad():
                    mean_probs, entropy = self.model.predict_with_uncertainty(weak_imgs, num_passes=5)
                    # Pseudo-label criteria: high certainty (prob >= 0.85 or <= 0.15) & entropy <= 0.35
                    cert_mask = ((mean_probs >= 0.85) | (mean_probs <= 0.15)) & (entropy <= 0.35)
                    pseudo_labels = (mean_probs >= 0.5).float()

                    total_pseudo_accepted += int(cert_mask.sum().item())
                    total_pseudo_candidates += cert_mask.numel()

                # 2. Train on strong image with masked consistency loss
                self.model.train()
                self.optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
                    strong_logits = self.model(strong_imgs)
                    raw_bce = self.criterion(strong_logits, pseudo_labels)  # (B, num_classes)
                    masked_bce = (raw_bce * cert_mask.float()).sum() / max(1.0, cert_mask.float().sum())
                    loss_prox = fedprox_loss(self.model, global_shared_params, mu=mu)
                    loss = (lambda_u * masked_bce) + loss_prox

                if self.device.type == "cuda":
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item() * len(weak_imgs)

        stage_grad_norms = self.freezer.record_stage_gradients(self.model)
        freezing_stats = self.freezer.get_freezing_stats(self.model)
        energy_stats = self.energy_monitor.stop()

        acceptance_rate = (total_pseudo_accepted / max(1, total_pseudo_candidates))

        metrics: dict[str, Any] = {
            "node_id": self.node_id,
            "loss": float(total_loss / max(1, len(self.dataset))),
            "pseudo_labels_accepted": int(total_pseudo_accepted),
            "pseudo_acceptance_rate": round(float(acceptance_rate), 4),
            "energy_wh": energy_stats["energy_wh"],
            "co2_grams": energy_stats["co2_grams"],
            "frozen_ratio": freezing_stats["frozen_parameter_ratio"],
        }
        for s, norm in stage_grad_norms.items():
            metrics[f"grad_norm_{s}"] = float(norm)

        updated_params = self.get_parameters(config)

        # Byzantine Rogue Attack Simulation (Node 5)
        if config.get("byzantine_attack_active", False) and self.node_id == "node_5":
            from security.byzantine_defense import simulate_byzantine_attack
            attack_type = config.get("byzantine_attack_type", "sign_flip")
            scale = float(config.get("byzantine_attack_scale", 5.0))
            updated_params = simulate_byzantine_attack(updated_params, attack_type=attack_type, scale=scale)
            metrics["byzantine_attack_injected"] = True

        return updated_params, len(self.dataset), metrics

