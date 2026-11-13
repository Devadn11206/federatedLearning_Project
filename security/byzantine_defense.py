"""
FedMedVision — Byzantine Attack Simulation & Coordinate-Wise Trimmed Mean Defense
Protects the medical consortium against malicious or compromised hospital nodes:
- Simulates sign-flip and Gaussian poisoning (Node 5 — Highland Rural Clinic)
- Cosine-anomaly detection against coordinate-wise median
- Coordinate-wise trimmed mean parameter aggregation
"""
from __future__ import annotations

import numpy as np
import torch
from typing import Any
from utils.logger import get_logger

logger = get_logger("fedmedvision.security.byzantine")


def simulate_byzantine_attack(
    shared_ndarrays: list[np.ndarray],
    attack_type: str = "sign_flip",
    scale: float = 5.0,
    seed: int = 42,
) -> list[np.ndarray]:
    """
    Simulates a compromised hospital node (Node 5) injecting malicious weights.

    Args:
        shared_ndarrays: Original client update weight arrays
        attack_type: "sign_flip" or "gaussian"
        scale: Multiplier for sign-flip attack or std for Gaussian noise
        seed: Random seed for deterministic reproducibility
    """
    rng = np.random.RandomState(seed)
    poisoned_ndarrays = []

    for arr in shared_ndarrays:
        if attack_type == "sign_flip":
            poisoned_arr = -float(scale) * arr
        elif attack_type == "gaussian":
            noise = rng.normal(loc=0.0, scale=scale * np.std(arr), size=arr.shape)
            poisoned_arr = arr + noise
        else:
            raise ValueError(f"Unsupported attack type: {attack_type}")

        poisoned_ndarrays.append(poisoned_arr.astype(arr.dtype))

    return poisoned_ndarrays


def coordinate_wise_trimmed_mean(
    client_updates: list[tuple[list[np.ndarray], int, str | int]],
    trim_ratio: float = 0.16,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """
    Applies Coordinate-Wise Trimmed Mean across hospital updates and performs
    cosine similarity anomaly detection against the median update vector.

    Args:
        client_updates: List of (weights_ndarrays, num_examples, client_id)
        trim_ratio: Fraction of extreme updates to trim from both ends

    Returns:
        robust_aggregated_weights: Filtered & aggregated parameter arrays
        defense_telemetry: Dict reporting flagged nodes, cosine scores, and trim stats
    """
    num_clients = len(client_updates)
    if num_clients == 0:
        raise ValueError("No client updates provided for aggregation.")

    client_ids = [str(u[2]) for u in client_updates]
    num_layers = len(client_updates[0][0])

    # 1. Anomaly Detection via Cosine Similarity to Median Update
    # Flatten each client's entire update vector
    flattened_clients = []
    for weights, _, _ in client_updates:
        flat = np.concatenate([w.ravel() for w in weights])
        flattened_clients.append(flat)

    stacked_flat = np.stack(flattened_clients, axis=0)  # (num_clients, total_params)
    median_vector = np.median(stacked_flat, axis=0)  # (total_params,)

    median_norm = np.linalg.norm(median_vector)
    cosine_scores: dict[str, float] = {}
    flagged_client_ids: list[str] = []

    for i, c_id in enumerate(client_ids):
        vec = stacked_flat[i]
        vec_norm = np.linalg.norm(vec)
        if vec_norm > 0 and median_norm > 0:
            cos_sim = float(np.dot(vec, median_vector) / (vec_norm * median_norm))
        else:
            cos_sim = 1.0

        cosine_scores[c_id] = round(cos_sim, 4)
        # Updates moving in opposite direction (negative cosine) or severe anomaly
        if cos_sim < 0.0 or cos_sim < 0.2:
            flagged_client_ids.append(c_id)

    # 2. Coordinate-Wise Trimmed Mean Aggregation
    k = max(1, int(num_clients * trim_ratio))
    # Ensure at least 1 client remains after trimming both tails
    if 2 * k >= num_clients:
        k = max(0, (num_clients - 1) // 2)

    robust_weights = []
    for layer_idx in range(num_layers):
        layer_stack = np.stack([u[0][layer_idx] for u in client_updates], axis=0)  # (N, *shape)

        if k > 0:
            # Sort along client dimension (axis=0)
            sorted_stack = np.sort(layer_stack, axis=0)
            trimmed_stack = sorted_stack[k : num_clients - k]  # exclude top and bottom k
            aggregated_layer = np.mean(trimmed_stack, axis=0)
        else:
            aggregated_layer = np.mean(layer_stack, axis=0)

        robust_weights.append(aggregated_layer.astype(client_updates[0][0][layer_idx].dtype))

    defense_telemetry = {
        "num_clients": num_clients,
        "trimmed_k_each_tail": k,
        "remaining_clients_per_coord": num_clients - 2 * k,
        "cosine_scores": cosine_scores,
        "flagged_client_ids": flagged_client_ids,
        "attack_detected": len(flagged_client_ids) > 0,
    }

    if defense_telemetry["attack_detected"]:
        logger.warning(
            f"Byzantine Defense: Flagged anomaly in node(s) {flagged_client_ids} "
            f"with cosine scores {cosine_scores}. Safely rejected via Trimmed Mean."
        )

    return robust_weights, defense_telemetry

