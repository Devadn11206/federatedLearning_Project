"""
FedMedVision — Domain-Adaptive ConvNeXt-FedBN Vision Backbone
Wraps timm's convnext_tiny with:
- Opacus DP-SGD compatibility validation & fix
- MC-Dropout classification head for predictive uncertainty estimation
- FedBN state-dict filtering (local normalization layers excluded from FL aggregation)
"""
from __future__ import annotations

from typing import Any

try:
    import opacus
    from opacus.validators import ModuleValidator
    _OPACUS_AVAILABLE = True
except ImportError:
    opacus = None
    ModuleValidator = None
    _OPACUS_AVAILABLE = False

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class FedMedConvNeXt(nn.Module):
    """
    Domain-Adaptive ConvNeXt architecture tailored for medical federated learning:
      1. Pretrained convnext_tiny backbone with Opacus compatibility fix.
      2. MC-Dropout head (p=0.2) + multi-label linear projection.
      3. Supports uncertainty-gated prediction via Monte Carlo dropout sampling.
      4. Private local normalization layers for scanner domain adaptation (FedBN).
    """

    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        dropout_p: float = 0.2,
        opacus_strict: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dropout_p = dropout_p

        # 1. Instantiate backbone without classification head
        backbone = timm.create_model(
            "convnext_tiny",
            pretrained=pretrained,
            num_classes=0,  # pooled features output
        )

        # 2. Ensure Opacus DP-SGD compatibility (if Opacus is installed)
        if _OPACUS_AVAILABLE and ModuleValidator is not None:
            if not ModuleValidator.is_valid(backbone):
                backbone = ModuleValidator.fix(backbone)

            if opacus_strict:
                errors = ModuleValidator.validate(backbone, strict=True)
                if errors:
                    raise ValueError(f"Backbone failed Opacus validation: {errors}")

        self.backbone = backbone
        self.num_features = self.backbone.num_features

        # 3. MC-Dropout Classification Head
        self.dropout = nn.Dropout(p=dropout_p)
        self.classifier = nn.Linear(self.num_features, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract global pooled latent feature representation."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning raw logits (for BCEWithLogitsLoss).
        """
        features = self.extract_features(x)
        features = self.dropout(features)
        logits = self.classifier(features)
        return logits

    def forward_with_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both logits and latent features
        (useful for FedProto prototype alignment).
        """
        features = self.extract_features(x)
        dropped = self.dropout(features)
        logits = self.classifier(dropped)
        return logits, features

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        num_passes: int = 5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform Monte Carlo Dropout inference to estimate prediction probabilities
        and predictive entropy. Keeps dropout enabled during forward passes.

        Returns:
            mean_probs: torch.Tensor of shape (B, num_classes)
            predictive_entropy: torch.Tensor of shape (B, num_classes)
        """
        # Save training state and activate dropout
        was_training = self.training
        self.eval()
        self.dropout.train()  # force dropout active

        preds = []
        for _ in range(num_passes):
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            preds.append(probs.unsqueeze(0))

        # Restore previous training state
        self.train(was_training)

        # Stack predictions across passes: shape (num_passes, B, num_classes)
        stacked = torch.cat(preds, dim=0)
        mean_probs = torch.mean(stacked, dim=0)

        # Vectorized binary Shannon entropy per pathology:
        # H(p) = - [p * log(p + eps) + (1 - p) * log(1 - p + eps)]
        eps = 1e-7
        p = torch.clamp(mean_probs, eps, 1.0 - eps)
        entropy = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))

        return mean_probs, entropy

    def get_shared_state_dict(self) -> dict[str, torch.Tensor]:
        """Convenience method to retrieve shared parameters excluding local FedBN layers."""
        return get_shared_state_dict(self)

    def load_shared_state_dict(self, shared_dict: dict[str, torch.Tensor]) -> None:
        """Convenience method to load shared parameters excluding local FedBN layers."""
        load_shared_state_dict(self, shared_dict)


# ---------------------------------------------------------------------------
# FedBN State-Dict Filtering Utilities
# ---------------------------------------------------------------------------

def is_norm_layer_param(key: str) -> bool:
    """
    Check if a parameter/buffer key belongs to a local normalization layer
    (BatchNorm, LayerNorm, GroupNorm) to be kept private per hospital in FedBN.
    """
    lower_key = key.lower()
    # Common normalization identifiers in ConvNeXt, PyTorch, and timm
    norm_identifiers = [
        ".norm.",
        ".norm_downsample.",
        "norm.weight",
        "norm.bias",
        "bn.",
        "gn.",
        "layernorm",
        "head.norm",
    ]
    return any(ident in lower_key for ident in norm_identifiers)


def get_shared_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """
    Extract only shared parameters for federated server aggregation,
    strictly filtering out local FedBN normalization layers.
    """
    full_dict = model.state_dict()
    shared_dict = {
        k: v.clone().detach()
        for k, v in full_dict.items()
        if not is_norm_layer_param(k)
    }
    return shared_dict


def load_shared_state_dict(model: nn.Module, shared_dict: dict[str, torch.Tensor]) -> None:
    """
    Load received shared federated parameters into local model,
    leaving local normalization parameters untouched.
    """
    current_dict = model.state_dict()
    for k, v in shared_dict.items():
        if k in current_dict and not is_norm_layer_param(k):
            current_dict[k].copy_(v)
    model.load_state_dict(current_dict, strict=True)

