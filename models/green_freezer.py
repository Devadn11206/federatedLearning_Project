"""
FedMedVision — Green AI Adaptive Layer Freezing & Real Hardware Energy Tracker
"""
from __future__ import annotations

import time
from typing import Any
import torch
import torch.nn as nn
from utils.logger import get_logger

logger = get_logger("fedmedvision.green")

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False


class GreenAdaptiveFreezer:
    """
    Green AI Adaptive Layer Freezing for ConvNeXt backbone.
    Tracks per-stage gradient L2 norms across federated rounds.
    When a stage's relative gradient norm change stays below grad_norm_tol
    for grad_norm_patience consecutive rounds, freezes that stage (requires_grad = False).
    """

    def __init__(
        self,
        grad_norm_tol: float = 1e-3,
        grad_norm_patience: int = 6,
        warmup_rounds: int = 6,
        stages: list[str] | None = None,
    ) -> None:
        self.grad_norm_tol = grad_norm_tol
        self.grad_norm_patience = grad_norm_patience
        self.warmup_rounds = warmup_rounds
        self.current_round: int = 0
        self.stage_names = stages or ["stages.0", "stages.1", "stages.2", "stages.3"]
        self.history: dict[str, list[float]] = {s: [] for s in self.stage_names}
        self.stable_counter: dict[str, int] = {s: 0 for s in self.stage_names}
        self.frozen_stages: set[str] = set()

    def record_stage_gradients(self, model: nn.Module) -> dict[str, float]:
        """Calculates gradient L2 norm for each stage."""
        norms: dict[str, float] = {}
        for s_name in self.stage_names:
            total_norm_sq = 0.0
            found = False
            for p_name, param in model.named_parameters():
                if s_name in p_name and param.grad is not None:
                    found = True
                    total_norm_sq += float(param.grad.detach().norm(2).item() ** 2)
            norms[s_name] = total_norm_sq ** 0.5 if found else 0.0
        return norms

    def update_and_check(self, stage_grad_norms: dict[str, float]) -> set[str]:
        """Updates gradient history and detects newly frozen stages."""
        self.current_round += 1
        newly_frozen = set()

        # Enforce warmup rounds before freezing any stages
        if self.current_round <= self.warmup_rounds:
            for s_name in self.stage_names:
                current_norm = stage_grad_norms.get(s_name, 0.0)
                self.history[s_name].append(current_norm)
            return newly_frozen

        for s_name in self.stage_names:
            if s_name in self.frozen_stages:
                continue

            current_norm = stage_grad_norms.get(s_name, 0.0)
            hist = self.history[s_name]

            if hist:
                prev_norm = hist[-1]
                rel_change = abs(current_norm - prev_norm) / max(1e-8, prev_norm)
                if rel_change < self.grad_norm_tol:
                    self.stable_counter[s_name] += 1
                else:
                    self.stable_counter[s_name] = 0

                if self.stable_counter[s_name] >= self.grad_norm_patience:
                    self.frozen_stages.add(s_name)
                    newly_frozen.add(s_name)
                    logger.info(f"Green AI: Freezing {s_name} (Stable for {self.grad_norm_patience} rounds).")

            hist.append(current_norm)

        return self.frozen_stages

    def apply_frozen_stages(self, model: nn.Module, frozen_stages: set[str] | list[str]) -> None:
        """Enforces requires_grad = False for parameters in frozen stages."""
        frozen_set = set(frozen_stages)
        self.frozen_stages = frozen_set
        for p_name, param in model.named_parameters():
            if any(s in p_name for s in frozen_set):
                param.requires_grad = False

    def get_freezing_stats(self, model: nn.Module) -> dict[str, Any]:
        """Returns statistics on frozen parameters and estimated FLOPs savings ratio."""
        total_params = 0
        frozen_params = 0
        for p in model.parameters():
            total_params += p.numel()
            if not p.requires_grad:
                frozen_params += p.numel()

        frozen_ratio = (frozen_params / total_params) if total_params > 0 else 0.0
        # Conv layers represent ~80% of total backward pass FLOPs; scale savings accordingly
        estimated_flops_savings = frozen_ratio * 0.85

        return {
            "frozen_stages": sorted(list(self.frozen_stages)),
            "total_parameters": total_params,
            "frozen_parameters": frozen_params,
            "frozen_parameter_ratio": round(frozen_ratio, 4),
            "estimated_flops_savings_ratio": round(estimated_flops_savings, 4),
        }


class EnergyMonitor:
    """
    Monitors real-time GPU power draw via NVML on RTX 3050.
    Calculates cumulative energy (Wh) and estimated CO2 emissions (grams).
    Falls back gracefully to TDP/wall-clock estimation if hardware NVML is restricted.
    """

    def __init__(self, device_index: int = 0, co2_kg_per_kwh: float = 0.42, fallback_tdp_watts: float = 45.0) -> None:
        self.device_index = device_index
        self.co2_kg_per_kwh = co2_kg_per_kwh
        self.fallback_tdp = fallback_tdp_watts
        self.nvml_active = False
        self.nvml_handle = None

        if HAS_PYNVML and torch.cuda.is_available():
            try:
                pynvml.nvmlInit()
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                self.nvml_active = True
            except Exception:
                self.nvml_active = False

        self.start_time: float = 0.0
        self.power_readings: list[float] = []

    def start(self) -> None:
        self.start_time = time.time()
        self.power_readings.clear()
        self.sample()

    def sample(self) -> float:
        """Sample instantaneous power draw in Watts."""
        watts = self.fallback_tdp
        if self.nvml_active and self.nvml_handle:
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self.nvml_handle)
                watts = float(power_mw) / 1000.0
            except Exception:
                watts = self.fallback_tdp
        self.power_readings.append(watts)
        return watts

    def stop(self) -> dict[str, float]:
        """Calculates energy in Wh and CO2 in grams for the monitored interval."""
        self.sample()
        duration_sec = max(0.001, time.time() - self.start_time)
        avg_power_watts = (sum(self.power_readings) / len(self.power_readings)) if self.power_readings else self.fallback_tdp

        duration_hours = duration_sec / 3600.0
        energy_wh = avg_power_watts * duration_hours
        energy_kwh = energy_wh / 1000.0
        co2_grams = energy_kwh * self.co2_kg_per_kwh * 1000.0

        return {
            "duration_sec": round(duration_sec, 2),
            "avg_power_watts": round(avg_power_watts, 2),
            "energy_wh": round(energy_wh, 4),
            "co2_grams": round(co2_grams, 4),
        }

    def close(self) -> None:
        if self.nvml_active:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self.nvml_active = False

