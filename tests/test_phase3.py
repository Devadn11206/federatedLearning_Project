"""
Unit and integration tests for Phase 3:
- Opacus DP-SGD wrapper & Rényi DP accounting
- Byzantine attack simulation & coordinate-wise trimmed mean defense
- Grad-CAM++ radiologist bounding-box IoU localization
- EU AI Act cryptographic audit ledger & tamper detection
"""
import json
from pathlib import Path
import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from PIL import Image
from explainability.gradcam_xai import MedicalGradCAMExplainer
from models.convnext_fedbn import FedMedConvNeXt
from security.audit_ledger import EUAIActAuditLedger
from security.byzantine_defense import coordinate_wise_trimmed_mean, simulate_byzantine_attack
from security.dp_privacy import DPPrivacyWrapper
from utils.config_loader import load_config


class TestDifferentialPrivacy:
    def test_dp_wrapper_initialization_and_budget(self):
        model = FedMedConvNeXt(num_classes=5, pretrained=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        dummy_x = torch.randn(16, 3, 224, 224)
        dummy_y = torch.zeros(16, 5)
        ds = TensorDataset(dummy_x, dummy_y)
        dl = DataLoader(ds, batch_size=4)

        wrapper = DPPrivacyWrapper(
            model=model,
            optimizer=optimizer,
            dataloader=dl,
            target_epsilon=8.0,
            target_delta=1e-5,
            max_grad_norm=1.0,
        )

        budget = wrapper.get_privacy_budget()
        assert "epsilon" in budget
        assert budget["delta"] == 1e-5
        assert budget["noise_multiplier"] > 0.0
        assert budget["compliant"] is True


class TestByzantineDefense:
    def test_simulate_byzantine_attack(self):
        weights = [np.ones((4, 4), dtype=np.float32), np.zeros((4,), dtype=np.float32)]
        poisoned = simulate_byzantine_attack(weights, attack_type="sign_flip", scale=3.0)

        assert len(poisoned) == len(weights)
        assert np.allclose(poisoned[0], -3.0 * np.ones((4, 4), dtype=np.float32))

    def test_coordinate_wise_trimmed_mean_detects_anomaly(self):
        # 5 normal clients, 1 rogue client
        w_shape = (2, 2)
        normal_weight = [np.ones(w_shape, dtype=np.float32)]
        rogue_weight = [-10.0 * np.ones(w_shape, dtype=np.float32)]

        client_updates = [
            (normal_weight, 100, f"node_{i}") for i in range(5)
        ]
        # Append poisoned client
        client_updates.append((rogue_weight, 100, "node_5"))

        robust_weights, defense_telem = coordinate_wise_trimmed_mean(client_updates, trim_ratio=0.16)

        assert len(robust_weights) == 1
        # Rogue client should be detected
        assert "node_5" in defense_telem["flagged_client_ids"]
        assert defense_telem["attack_detected"] is True
        # Trimmed mean should remove the negative extreme
        assert robust_weights[0][0, 0] > 0.0


class TestMedicalGradCAM:
    def test_single_image_explanation(self):
        model = FedMedConvNeXt(num_classes=5, pretrained=False)
        explainer = MedicalGradCAMExplainer(model)
        dummy_img = torch.randn(3, 224, 224)
        pil_img = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype(np.uint8))

        res = explainer.explain_single_image(pil_img, target_class_idx=0)
        assert res["heatmap"].shape == (224, 224)
        assert res["cam_overlay"].shape == (224, 224, 3)
        assert len(res["probs"]) == 5
        assert len(res["uncertainty"]) == 5

    def test_evaluate_bbox_localization_on_splits(self):
        model = FedMedConvNeXt(num_classes=5, pretrained=False)
        explainer = MedicalGradCAMExplainer(model)

        res = explainer.evaluate_bbox_localization(
            bbox_csv="data/splits/bbox_eval.csv",
            threshold=0.5,
            output_dir="results/xai_overlays",
        )
        assert res["total_evaluated"] > 0
        assert 0.0 <= res["mean_iou"] <= 1.0
        assert 0.0 <= res["pointing_game_accuracy"] <= 1.0


class TestEUAIActAuditLedger:
    def test_ledger_chain_and_tamper_detection(self, tmp_path):
        ledger_file = tmp_path / "test_ledger.jsonl"
        ledger = EUAIActAuditLedger(ledger_file=ledger_file)

        # 1. Append valid entries
        ledger.append_audit_entry(
            round_id=1,
            event_type="test_round_1",
            participating_nodes=["node_0", "node_1"],
            privacy_budget={"epsilon": 1.0},
            byzantine_status={"attack_detected": False},
            green_metrics={"energy_wh": 2.5},
        )
        ledger.append_audit_entry(
            round_id=2,
            event_type="test_round_2",
            participating_nodes=["node_0", "node_1"],
            privacy_budget={"epsilon": 2.1},
            byzantine_status={"attack_detected": False},
            green_metrics={"energy_wh": 5.1},
        )

        is_valid, tampered_line = ledger.verify_chain_integrity()
        assert is_valid is True
        assert tampered_line is None

        # 2. Simulate tampering with line 1
        with ledger_file.open("r", encoding="utf-8") as f:
            lines = f.readlines()

        tampered_entry = json.loads(lines[0])
        tampered_entry["privacy_budget"]["epsilon"] = 999.9  # unauthorized change
        lines[0] = json.dumps(tampered_entry) + "\n"

        with ledger_file.open("w", encoding="utf-8") as f:
            f.writelines(lines)

        # 3. Verification must fail immediately
        is_valid_after, tampered_line_after = ledger.verify_chain_integrity()
        assert is_valid_after is False
        assert tampered_line_after == 1
