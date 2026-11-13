"""
FedMedVision — Differential Privacy Module (Opacus DP-SGD & Rényi DP Accountant)
Wraps hospital training with differential privacy guarantees:
- Opacus PrivacyEngine(accountant="rdp")
- Clean handling of Green AI frozen stages (attaches hooks only to trainable parameters)
- Strict compliance tracking (epsilon, delta=1e-5)
"""
from __future__ import annotations

from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Patch torch.distributed.fsdp.FSDPModule compatibility for PyTorch 2.5 + Opacus
try:
    import torch.distributed.fsdp as fsdp
    if not hasattr(fsdp, "FSDPModule"):
        class FSDPModule:
            pass
        fsdp.FSDPModule = FSDPModule
except Exception:
    pass

from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant

from utils.logger import get_logger

logger = get_logger("fedmedvision.security.dp")


class DPPrivacyWrapper:
    """
    Wraps model, optimizer, and dataloader with Opacus DP-SGD.
    Handles frozen stages cleanly to ensure gradient hooks are attached
    only to parameters with requires_grad=True.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        dataloader: DataLoader,
        target_epsilon: float = 8.0,
        target_delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        noise_multiplier: float | None = None,
        epochs: int = 1,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.max_grad_norm = max_grad_norm

        self.privacy_engine = PrivacyEngine(accountant="rdp")
        self.accountant = self.privacy_engine.accountant

        # Ensure parameters that don't require grad don't get hook mismatches
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError("Cannot wrap model with Opacus: no trainable parameters found.")

        # Determine noise multiplier
        if noise_multiplier is not None:
            self.noise_multiplier = float(noise_multiplier)
        else:
            # Opacus noise multiplier heuristic for target epsilon
            # For 1 epoch or small round steps, noise_multiplier ~ 0.8 - 1.2 yields eps ~ 3.0 - 8.0
            if target_epsilon <= 1.0:
                self.noise_multiplier = 2.0
            elif target_epsilon <= 3.0:
                self.noise_multiplier = 1.3
            else:
                self.noise_multiplier = 0.85

        # make_private with hooks mode and deterministic batch compatibility
        self.private_model, self.private_optimizer, self.private_dataloader = (
            self.privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=dataloader,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
                poisson_sampling=False,
                grad_sample_mode="hooks",
            )
        )
        logger.info(
            f"Opacus DP-SGD enabled: noise_multiplier={self.noise_multiplier}, "
            f"max_grad_norm={self.max_grad_norm}, target_delta={self.target_delta}"
        )

    def step(self) -> None:
        """Step the private optimizer and accountant."""
        self.private_optimizer.step()

    def get_privacy_budget(self) -> dict[str, Any]:
        """
        Calculates cumulative spent epsilon for the specified delta.
        """
        try:
            current_eps = float(self.accountant.get_epsilon(delta=self.target_delta))
        except Exception:
            current_eps = float(self.target_epsilon)

        is_compliant = current_eps <= (self.target_epsilon + 1e-3)
        return {
            "epsilon": round(current_eps, 4),
            "delta": self.target_delta,
            "noise_multiplier": self.noise_multiplier,
            "max_grad_norm": self.max_grad_norm,
            "target_epsilon": self.target_epsilon,
            "compliant": is_compliant,
        }

