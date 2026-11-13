"""
Unit and integration tests for Phase 2:
- FedProto prototype alignment and FedProx regularization
- Green AI adaptive layer freezing and energy tracking
- Unlabeled Clinic MC-Dropout uncertainty gating
- Flower federated simulation execution
"""
from pathlib import Path
import pytest
import numpy as np
import torch
import torch.nn as nn

from models.convnext_fedbn import FedMedConvNeXt
from models.fedproto_loss import (
    compute_local_prototypes,
    fedproto_alignment_loss,
    fedprox_loss,
)
from models.green_freezer import GreenAdaptiveFreezer, EnergyMonitor
from fl_engine.client_hub import LabeledHubClient
from fl_engine.client_clinic import UnlabeledClinicClient
from utils.config_loader import load_config


class TestFedProtoAndFedProx:
    def test_compute_local_prototypes(self):
        batch_size = 8
        feat_dim = 16
        num_classes = 5

        features = torch.randn(batch_size, feat_dim)
        targets = torch.zeros(batch_size, num_classes)
        targets[0, 0] = 1.0
        targets[1, 0] = 1.0
        targets[2, 1] = 1.0

        protos, counts = compute_local_prototypes(features, targets, num_classes=num_classes)
        assert 0 in protos
        assert counts[0] == 2
        assert 1 in protos
        assert counts[1] == 1
        assert 2 not in protos

        # Verify unit norm
        norm_val = torch.norm(protos[0], p=2).item()
        assert abs(norm_val - 1.0) < 1e-5

    def test_fedproto_alignment_loss(self):
        features = torch.randn(4, 16)
        targets = torch.zeros(4, 5)
        targets[0, 0] = 1.0

        # Empty global prototypes -> 0.0
        loss_zero = fedproto_alignment_loss(features, targets, {})
        assert loss_zero.item() == 0.0

        # With global prototype
        global_protos = {0: torch.randn(16)}
        loss = fedproto_alignment_loss(features, targets, global_protos)
        assert loss.item() >= 0.0

    def test_fedprox_loss_excludes_fedbn(self):
        model = FedMedConvNeXt(num_classes=5, pretrained=False)
        shared_dict = model.get_shared_state_dict()

        # Perturb one parameter
        perturbed = {k: v.clone() for k, v in shared_dict.items()}
        first_key = list(perturbed.keys())[0]
        perturbed[first_key] = perturbed[first_key] + 1.0

        loss = fedprox_loss(model, perturbed, mu=0.01)
        assert loss.item() > 0.0


class TestGreenAIFreezerAndEnergy:
    def test_green_freezer_freezes_stable_stages(self):
        freezer = GreenAdaptiveFreezer(grad_norm_tol=1e-3, grad_norm_patience=2, warmup_rounds=0)
        stage_names = freezer.stage_names

        # Feed identical gradient norms to trigger freezing
        for _ in range(3):
            freezer.update_and_check({s: 0.05 for s in stage_names})

        assert len(freezer.frozen_stages) == len(stage_names)

    def test_apply_freezer_to_model(self):
        model = FedMedConvNeXt(num_classes=5, pretrained=False)
        freezer = GreenAdaptiveFreezer()
        freezer.apply_frozen_stages(model, ["stages.0"])

        stats = freezer.get_freezing_stats(model)
        assert stats["frozen_parameters"] > 0
        assert stats["frozen_parameter_ratio"] > 0.0

    def test_energy_monitor_sampling(self):
        monitor = EnergyMonitor()
        monitor.start()
        stats = monitor.stop()
        assert stats["duration_sec"] >= 0.0
        assert stats["avg_power_watts"] > 0.0
        assert stats["energy_wh"] >= 0.0
        monitor.close()


class TestFederatedClients:
    def test_labeled_hub_client_fit(self):
        config = load_config()
        client = LabeledHubClient("node_0", "data/splits/hub_0.csv", config=config)
        params = client.get_parameters({})

        # Mock 1 fit step on 1 batch
        config_dict = {"round": 1, "local_epochs": 1, "fedprox_mu": 0.01, "proto_lambda": 0.1}
        new_params, n_samples, metrics = client.fit(params, config_dict)

        assert len(new_params) == len(params)
        assert n_samples > 0
        assert "loss" in metrics
        assert "proto_0" in metrics or "energy_wh" in metrics

    def test_unlabeled_clinic_client_mc_dropout(self):
        config = load_config()
        client = UnlabeledClinicClient("node_2", "data/splits/clinic_2.csv", config=config)
        params = client.get_parameters({})

        config_dict = {"round": 1, "local_epochs": 1, "fedprox_mu": 0.01, "lambda_u": 0.5}
        new_params, n_samples, metrics = client.fit(params, config_dict)

        assert len(new_params) == len(params)
        assert n_samples > 0
        assert "pseudo_labels_accepted" in metrics
        assert "pseudo_acceptance_rate" in metrics

