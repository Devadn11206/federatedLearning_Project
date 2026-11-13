"""
FedMedVision — Custom Federated Strategy & Server Orchestrator
Coordinates:
- Shared weights aggregation (FedBN local scanner normalization preserved)
- Prototype contrastive alignment (FedProto) across Hubs 0 & 1
- Green AI adaptive stage freezing broadcasts
- Evaluation on held-out Consortium Val set & External zero-shot VinDr cohort
- Comprehensive per-round telemetry recording
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import ChestXrayConsortiumDataset
from models.convnext_fedbn import FedMedConvNeXt
from models.green_freezer import GreenAdaptiveFreezer
from models.metrics import compute_ece, compute_multilabel_metrics
from utils.config_loader import load_config
from utils.logger import get_logger

logger = get_logger("fedmedvision.fl.server")


class FedMedStrategy(fl.server.strategy.Strategy):
    """
    Custom Flower strategy for FedMedVision consortium.
    """

    def __init__(
        self,
        initial_parameters: Parameters,
        config: Any | None = None,
        val_csv: Path | str | None = None,
        ext_csv: Path | str | None = None,
        checkpoint_path: Path | str | None = None,
        telemetry_path: Path | str | None = None,
        enable_byzantine_defense: bool = False,
        trim_ratio: float = 0.16,
    ) -> None:
        super().__init__()
        self.config = config or load_config()
        self.current_parameters = initial_parameters
        self.enable_byzantine_defense = enable_byzantine_defense
        self.trim_ratio = trim_ratio
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pathologies = self.config.consortium.pathologies
        self.num_classes = len(self.pathologies)

        # Telemetry & Checkpoints
        self.checkpoint_path = Path(checkpoint_path or (self.config.project_root / "results" / "checkpoints" / "fedmedvision_global.pt"))
        self.telemetry_path = Path(telemetry_path or (self.config.project_root / "results" / "metrics" / "fl_telemetry.json"))
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)

        self.telemetry_history: list[dict[str, Any]] = []
        self.best_macro_auroc: float = -1.0

        # Global Prototypes & Green Freezing
        self.global_prototypes: dict[int, np.ndarray] = {}
        green_cfg = self.config.security.green_ai
        self.freezer = GreenAdaptiveFreezer(
            grad_norm_tol=green_cfg.get("grad_norm_tol", 1e-3),
            grad_norm_patience=green_cfg.get("grad_norm_patience", 3),
        )

        # Cumulative counters
        self.cumulative_energy_wh: float = 0.0
        self.cumulative_co2_grams: float = 0.0

        # Evaluation DataLoaders (Val + External)
        val_path = Path(val_csv or (self.config.project_root / "data" / "splits" / "val_test.csv"))
        ext_path = Path(ext_csv or (self.config.project_root / "data" / "splits" / "external_test.csv"))

        val_ds = ChestXrayConsortiumDataset(val_path, mode="eval", pathologies=self.pathologies)
        ext_ds = ChestXrayConsortiumDataset(ext_path, mode="eval", pathologies=self.pathologies)

        self.val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
        self.ext_loader = DataLoader(ext_ds, batch_size=32, shuffle=False, num_workers=0)

        # Global Evaluation Model
        self.eval_model = FedMedConvNeXt(
            num_classes=self.num_classes,
            pretrained=False,
            dropout_p=0.0,
        ).to(self.device)

    def initialize_parameters(self, client_manager: fl.server.client_manager.ClientManager) -> Parameters | None:
        return self.current_parameters

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training, broadcasting prototypes and frozen stages."""
        clients = client_manager.sample(num_clients=client_manager.num_available(), min_num_clients=1)

        fit_config: dict[str, Scalar] = {
            "round": server_round,
            "server_round": server_round,
            "local_epochs": 1,
            "fedprox_mu": float(getattr(self.config.training, "fedprox_mu", 0.01)),
            "proto_lambda": float(getattr(self.config.training, "fedproto_lambda", 0.1)),
            "lambda_u": float(self.config.training.semi_supervised.get("unsup_lambda", 0.5) if isinstance(self.config.training.semi_supervised, dict) else 0.5),
            "frozen_stages": ",".join(sorted(list(self.freezer.frozen_stages))),
        }

        # Broadcast global prototypes
        for c, proto in self.global_prototypes.items():
            fit_config[f"proto_{c}"] = proto.tolist()

        fit_ins = FitIns(parameters, fit_config)
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        if not results:
            return None, {}

        # 1. Parameter aggregation (FedAvg or Coordinate-Wise Trimmed Mean)
        # Apply source-aware weighting: 3.0x for labeled hubs (node_0, node_1), 0.5x for unlabeled clinics (node_2..5)
        def get_effective_examples(node_id: str, raw_count: int) -> float:
            if "0" in str(node_id) or "1" in str(node_id) or "hub" in str(node_id).lower():
                return float(raw_count) * 3.0
            return float(raw_count) * 0.5

        if self.enable_byzantine_defense:
            from security.byzantine_defense import coordinate_wise_trimmed_mean
            client_tuples = [
                (
                    parameters_to_ndarrays(fit_res.parameters),
                    int(get_effective_examples(fit_res.metrics.get("node_id", f"node_{i}"), fit_res.num_examples)),
                    fit_res.metrics.get("node_id", f"node_{i}"),
                )
                for i, (_, fit_res) in enumerate(results)
            ]
            aggregated_ndarrays, defense_telemetry = coordinate_wise_trimmed_mean(
                client_tuples, trim_ratio=self.trim_ratio
            )
        else:
            weights_results = [
                (
                    parameters_to_ndarrays(fit_res.parameters),
                    get_effective_examples(fit_res.metrics.get("node_id", f"node_{i}"), fit_res.num_examples),
                )
                for i, (_, fit_res) in enumerate(results)
            ]
            total_examples = sum(num_examples for _, num_examples in weights_results)
            aggregated_ndarrays = [
                np.zeros_like(weights) for weights in weights_results[0][0]
            ]
            for client_weights, num_examples in weights_results:
                weight_factor = num_examples / max(1e-8, total_examples)
                for i, layer in enumerate(client_weights):
                    aggregated_ndarrays[i] += layer * weight_factor

        self.current_parameters = ndarrays_to_parameters(aggregated_ndarrays)

        # 2. Prototype aggregation (from Hubs 0 & 1)
        proto_accum: dict[int, np.ndarray] = {}
        proto_weights: dict[int, int] = {}

        round_energy_wh = 0.0
        round_co2_grams = 0.0
        round_pseudo_accepted = 0
        stage_grad_norms: dict[str, list[float]] = {s: [] for s in self.freezer.stage_names}

        for _, fit_res in results:
            m = fit_res.metrics
            round_energy_wh += float(m.get("energy_wh", 0.0))
            round_co2_grams += float(m.get("co2_grams", 0.0))
            round_pseudo_accepted += int(m.get("pseudo_labels_accepted", 0))

            # Prototypes
            for c in range(self.num_classes):
                key = f"proto_{c}"
                cnt_key = f"proto_count_{c}"
                if key in m and cnt_key in m:
                    proto_vec = np.array(m[key], dtype=np.float32)
                    count = int(m[cnt_key])
                    if count > 0:
                        if c not in proto_accum:
                            proto_accum[c] = np.zeros_like(proto_vec)
                            proto_weights[c] = 0
                        proto_accum[c] += proto_vec * count
                        proto_weights[c] += count

            # Green AI Gradient Norms
            for s in self.freezer.stage_names:
                g_key = f"grad_norm_{s}"
                if g_key in m:
                    stage_grad_norms[s].append(float(m[g_key]))

        self.cumulative_energy_wh += round_energy_wh
        self.cumulative_co2_grams += round_co2_grams

        # Normalize aggregated prototypes
        for c, weighted_sum in proto_accum.items():
            w_total = proto_weights[c]
            if w_total > 0:
                mean_proto = weighted_sum / float(w_total)
                norm = np.linalg.norm(mean_proto)
                self.global_prototypes[c] = mean_proto / max(1e-8, norm)

        # Check Green AI Adaptive Freezing
        mean_stage_norms = {
            s: (float(np.mean(norms)) if norms else 0.0)
            for s, norms in stage_grad_norms.items()
        }
        newly_frozen = self.freezer.update_and_check(mean_stage_norms)

        # Calculate communication payload (MB)
        payload_bytes = sum(w.nbytes for w in aggregated_ndarrays) * len(results)
        payload_mb = round(payload_bytes / (1024.0 * 1024.0), 2)

        # 3. Server-side Evaluation on Real Val & External Test Sets
        val_metrics = self._evaluate_global(self.current_parameters, self.val_loader)
        ext_metrics = self._evaluate_global(self.current_parameters, self.ext_loader)

        val_auroc = val_metrics["macro_auroc"]
        ext_auroc = ext_metrics["macro_auroc"]
        val_ece = val_metrics["ece"]

        # Checkpoint saving
        if val_auroc > self.best_macro_auroc:
            self.best_macro_auroc = val_auroc
            torch.save(
                {
                    "round": server_round,
                    "state_dict": self.eval_model.state_dict(),
                    "val_macro_auroc": val_auroc,
                    "ext_macro_auroc": ext_auroc,
                    "ece": val_ece,
                    "optimal_temperature": val_metrics.get("optimal_temperature", 1.0),
                },
                self.checkpoint_path,
            )

            # Also persist calibration parameters
            cal_params_path = self.config.project_root / "results" / "metrics" / "calibration_params.json"
            with open(cal_params_path, "w", encoding="utf-8") as f:
                json.dump({
                    "optimal_temperature": val_metrics.get("optimal_temperature", 1.0),
                    "raw_ece": val_metrics.get("raw_ece", val_ece),
                    "calibrated_ece": val_ece,
                    "round": server_round,
                }, f, indent=2)

        telemetry_entry = {
            "round": server_round,
            "val_macro_auroc": val_auroc,
            "ext_macro_auroc": ext_auroc,
            "val_ece": val_ece,
            "pseudo_labels_accepted": round_pseudo_accepted,
            "frozen_stages": sorted(list(self.freezer.frozen_stages)),
            "round_energy_wh": round(round_energy_wh, 4),
            "cumulative_energy_wh": round(self.cumulative_energy_wh, 4),
            "cumulative_co2_grams": round(self.cumulative_co2_grams, 4),
            "uplink_payload_mb": payload_mb,
        }
        self.telemetry_history.append(telemetry_entry)
        with self.telemetry_path.open("w", encoding="utf-8") as f:
            json.dump(self.telemetry_history, f, indent=2)

        logger.info(
            f"Round {server_round:02d} | Val AUROC: {val_auroc:.4f} | "
            f"Ext AUROC: {ext_auroc:.4f} | ECE: {val_ece:.4f} | "
            f"Pseudo-labels: {round_pseudo_accepted} | Frozen: {len(self.freezer.frozen_stages)} | "
            f"Energy: {self.cumulative_energy_wh:.2f} Wh"
        )

        metrics_out: dict[str, Scalar] = {
            "val_macro_auroc": float(val_auroc),
            "ext_macro_auroc": float(ext_auroc),
            "val_ece": float(val_ece),
        }
        return self.current_parameters, metrics_out

    @torch.no_grad()
    def _evaluate_global(
        self,
        parameters: Parameters,
        dataloader: DataLoader,
    ) -> dict[str, Any]:
        """Evaluates global model using FP16 inference."""
        ndarrays = parameters_to_ndarrays(parameters)
        shared_keys = list(self.eval_model.get_shared_state_dict().keys())
        state_dict = {
            k: torch.tensor(v, device=self.device) for k, v in zip(shared_keys, ndarrays)
        }
        self.eval_model.load_shared_state_dict(state_dict)
        self.eval_model.eval()

        all_targets = []
        all_logits = []
        all_probs = []

        for images, targets, _ in dataloader:
            images = images.to(self.device)
            with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
                logits = self.eval_model(images)
                probs = torch.sigmoid(logits)

            all_targets.append(targets.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

        y_true = np.concatenate(all_targets, axis=0)
        y_logits = np.concatenate(all_logits, axis=0)
        y_probs = np.concatenate(all_probs, axis=0)

        metrics = compute_multilabel_metrics(y_true, y_probs, self.pathologies)
        raw_ece = compute_ece(y_true, y_probs)

        # Post-hoc Temperature Scaling calibration
        from models.metrics import fit_temperature_scaler
        opt_T, cal_probs = fit_temperature_scaler(y_logits, y_true)
        cal_ece = compute_ece(y_true, cal_probs)

        metrics["raw_ece"] = raw_ece
        metrics["ece"] = cal_ece
        metrics["optimal_temperature"] = opt_T
        return metrics

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        return []

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> tuple[float | None, dict[str, Scalar]]:
        return None, {}

    def evaluate(self, server_round: int, parameters: Parameters) -> tuple[float, dict[str, Scalar]] | None:
        return None

