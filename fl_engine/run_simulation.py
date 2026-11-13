"""
FedMedVision — Federated Simulation Execution Script
Runs the 6-hospital consortium using Flower simulation mode:
- Nodes 0 & 1: Labeled Urban Hubs (NIH CXR14)
- Nodes 2, 3, 4, 5: Unlabeled Regional Clinics (CheXpert MC-Dropout Gated)
- Sequential GPU execution to keep VRAM strictly within 4.0 GB budget
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import flwr as fl
from flwr.common import ndarrays_to_parameters
import torch

from fl_engine.client_clinic import UnlabeledClinicClient
from fl_engine.client_hub import LabeledHubClient
from fl_engine.server import FedMedStrategy
from models.convnext_fedbn import FedMedConvNeXt
from utils.config_loader import load_config
from utils.logger import get_logger, log_system_event

logger = get_logger("fedmedvision.simulation")


def client_fn_factory(config):
    splits_dir = config.project_root / "data" / "splits"

    client_mapping = {
        "0": ("node_0", splits_dir / "hub_0.csv", "hub"),
        "1": ("node_1", splits_dir / "hub_1.csv", "hub"),
        "2": ("node_2", splits_dir / "clinic_2.csv", "clinic"),
        "3": ("node_3", splits_dir / "clinic_3.csv", "clinic"),
        "4": ("node_4", splits_dir / "clinic_4.csv", "clinic"),
        "5": ("node_5", splits_dir / "clinic_5.csv", "clinic"),
    }

    def client_fn(cid: str) -> fl.client.Client:
        node_id, csv_path, client_type = client_mapping[str(cid)]
        if client_type == "hub":
            client = LabeledHubClient(node_id=node_id, split_csv=csv_path, config=config)
        else:
            client = UnlabeledClinicClient(node_id=node_id, split_csv=csv_path, config=config)
        return client.to_client()

    return client_fn


def main():
    parser = argparse.ArgumentParser(description="FedMedVision Federated Consortium Simulation")
    parser.add_argument("--rounds", type=int, default=15, help="Number of FL rounds")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local epochs per round")
    args = parser.parse_args()

    config = load_config()
    log_system_event(logger, "fl_simulation_start", rounds=args.rounds, local_epochs=args.local_epochs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Initialize shared model parameters (warm-started from pretrained / Hub 0 weights if available)
    init_model = FedMedConvNeXt(
        num_classes=len(config.consortium.pathologies),
        pretrained=False,
        dropout_p=config.training.dropout_p,
    )
    
    hub_0_ckpt = config.project_root / "results" / "checkpoints" / "isolated_hub_0.pt"
    if hub_0_ckpt.exists():
        logger.info(f"Warm-starting FL global initial weights from {hub_0_ckpt}")
        ckpt_data = torch.load(hub_0_ckpt, map_location="cpu", weights_only=False)
        sd = ckpt_data.get("state_dict", ckpt_data.get("model_state_dict", ckpt_data))
        init_model.load_state_dict(sd, strict=False)

    shared_dict = init_model.get_shared_state_dict()
    initial_ndarrays = [v.cpu().numpy() for v in shared_dict.values()]
    initial_parameters = ndarrays_to_parameters(initial_ndarrays)

    # Strategy
    strategy = FedMedStrategy(
        initial_parameters=initial_parameters,
        config=config,
    )

    client_fn = client_fn_factory(config)

    # Configure client resources: 1.0 GPU per client ensures strict sequential execution within 4 GB VRAM
    client_resources = {
        "num_cpus": 2,
        "num_gpus": 1.0 if torch.cuda.is_available() else 0.0,
    }

    logger.info(f"Starting Flower Simulation for {args.rounds} rounds across 6 hospital nodes...")
    start_t = time.time()

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=6,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        client_resources=client_resources,
    )

    elapsed = time.time() - start_t
    logger.info(f"Federated Simulation finished in {elapsed / 60.0:.2f} minutes.")
    log_system_event(logger, "fl_simulation_complete", elapsed_seconds=elapsed, rounds=args.rounds)


if __name__ == "__main__":
    main()

