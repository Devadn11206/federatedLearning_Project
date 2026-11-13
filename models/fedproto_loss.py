"""
FedMedVision — FedProto Class-Wise Latent Prototype Alignment & FedProx Loss
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any


def compute_local_prototypes(
    features: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 5,
) -> tuple[dict[int, torch.Tensor], dict[int, int]]:
    """
    Computes L2-normalized mean latent embedding vector per pathology class
    for positive samples in a client batch / dataset.

    Args:
        features: (B, D) latent feature embeddings
        targets: (B, num_classes) binary multi-hot targets
        num_classes: total number of target classes

    Returns:
        prototypes: dict mapping class_idx -> (D,) normalized prototype tensor
        counts: dict mapping class_idx -> integer positive sample count
    """
    prototypes: dict[int, torch.Tensor] = {}
    counts: dict[int, int] = {}
    device = features.device

    for c in range(num_classes):
        # Select embeddings where class c is positive
        pos_mask = (targets[:, c] == 1.0)
        pos_count = int(pos_mask.sum().item())
        counts[c] = pos_count

        if pos_count > 0:
            pos_features = features[pos_mask]  # (K, D)
            mean_vec = pos_features.mean(dim=0)  # (D,)
            norm_vec = F.normalize(mean_vec, p=2, dim=0)
            prototypes[c] = norm_vec.detach().cpu()

    return prototypes, counts


def fedproto_alignment_loss(
    features: torch.Tensor,
    targets: torch.Tensor,
    global_prototypes: dict[int, torch.Tensor],
) -> torch.Tensor:
    """
    Computes MSE distance between local sample embeddings and server global prototypes
    for matching positive classes. Gracefully returns scalar 0.0 when global_prototypes is empty.

    Args:
        features: (B, D) local sample embeddings
        targets: (B, num_classes) binary multi-hot targets
        global_prototypes: dict mapping class_idx -> (D,) global prototype tensor
    """
    if not global_prototypes:
        return torch.tensor(0.0, device=features.device, requires_grad=True)

    device = features.device
    norm_features = F.normalize(features, p=2, dim=1)  # (B, D)

    total_loss = torch.tensor(0.0, device=device)
    matched_count = 0

    for c, g_proto in global_prototypes.items():
        pos_mask = (targets[:, c] == 1.0)
        if pos_mask.sum() > 0:
            g_proto_tensor = g_proto.to(device)  # (D,)
            sample_feats = norm_features[pos_mask]  # (K, D)
            # MSE loss to global prototype
            mse = F.mse_loss(sample_feats, g_proto_tensor.unsqueeze(0).expand_as(sample_feats))
            total_loss = total_loss + mse
            matched_count += 1

    if matched_count > 0:
        return total_loss / float(matched_count)
    return torch.tensor(0.0, device=device, requires_grad=True)


def fedprox_loss(
    local_model: nn.Module,
    global_shared_params: dict[str, torch.Tensor],
    mu: float = 0.01,
) -> torch.Tensor:
    """
    Computes FedProx proximal regularization:
      L_prox = (mu / 2) * sum ||w - w^t||^2
    over shared parameters only, strictly excluding private FedBN normalization keys.
    """
    if mu <= 0.0 or not global_shared_params:
        device = next(local_model.parameters()).device
        return torch.tensor(0.0, device=device)

    device = next(local_model.parameters()).device
    prox_loss = torch.tensor(0.0, device=device)

    for name, param in local_model.named_parameters():
        if name in global_shared_params:
            g_param = global_shared_params[name].to(device)
            prox_loss = prox_loss + torch.sum((param - g_param) ** 2)

    return (mu / 2.0) * prox_loss

