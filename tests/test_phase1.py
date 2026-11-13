"""
FedMedVision — Phase 1 Verification Test Suite
Validates:
  1. Data Consortium generation (`prepare_consortium.py --synthetic-dryrun`)
  2. Dataset and multi-mode clinical augmentations (`dataset.py`)
  3. ConvNeXt-FedBN architecture, Opacus validation, MC-Dropout, and FedBN state dict filtering (`convnext_fedbn.py`)
  4. Clinical multi-label classification and ECE calibration metrics (`metrics.py`)
  5. Baseline training execution and metric persistence (`train_baselines.py`)
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import opacus
from opacus.validators import ModuleValidator
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import ChestXrayConsortiumDataset, get_medical_transforms
from data.prepare_consortium import build_synthetic_consortium
from models.convnext_fedbn import (
    FedMedConvNeXt,
    get_shared_state_dict,
    is_norm_layer_param,
    load_shared_state_dict,
)
from models.metrics import compute_ece, compute_multilabel_metrics
from utils.config_loader import load_config


# ===========================================================================
# 1. Data Consortium Splitter Tests
# ===========================================================================

class TestDataConsortium:
    """Verify synthetic dryrun generation and metadata consistency."""

    @pytest.fixture(scope="class", autouse=True)
    def setup_data(self):
        cfg = load_config()
        output_dir = cfg.project_root / "data" / "processed"
        splits_dir = cfg.project_root / "data" / "splits"
        build_synthetic_consortium(
            output_dir=output_dir,
            splits_dir=splits_dir,
            config=cfg,
            num_hub_samples=20,
            num_clinic_samples=15,
            num_val_samples=15,
            num_ext_samples=10,
            num_bbox_samples=8,
        )

    def test_split_files_exist(self):
        cfg = load_config()
        splits_dir = cfg.project_root / "data" / "splits"
        required_splits = [
            "hub_0.csv",
            "hub_1.csv",
            "clinic_2.csv",
            "clinic_3.csv",
            "clinic_4.csv",
            "clinic_5.csv",
            "val_test.csv",
            "external_test.csv",
            "bbox_eval.csv",
            "consortium_stats.json",
        ]
        for f in required_splits:
            assert (splits_dir / f).exists(), f"Missing split file: {f}"

    def test_consortium_stats_json(self):
        cfg = load_config()
        stats_path = cfg.project_root / "data" / "splits" / "consortium_stats.json"
        with stats_path.open("r", encoding="utf-8") as f:
            stats = json.load(f)

        assert "nodes" in stats
        assert len(stats["nodes"]) == 6
        assert stats["pathologies"] == cfg.consortium.pathologies

        # Verify Hubs are labeled and Clinics are marked unlabeled
        assert stats["nodes"]["node_0"]["role"] == "labeled_hub"
        assert stats["nodes"]["node_2"]["role"] == "unlabeled_clinic"

    def test_bbox_eval_format(self):
        cfg = load_config()
        bbox_path = cfg.project_root / "data" / "splits" / "bbox_eval.csv"
        df = pd.read_csv(bbox_path)
        expected_cols = {"sample_id", "image_path", "finding", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "image_size"}
        assert expected_cols.issubset(set(df.columns))
        assert len(df) > 0
        assert df["image_size"].iloc[0] == 224


# ===========================================================================
# 2. PyTorch Dataset & Augmentation Tests
# ===========================================================================

class TestConsortiumDataset:
    """Verify dataset loading and albumentations augmentations across all modes."""

    @pytest.fixture(scope="class")
    def sample_csv(self):
        cfg = load_config()
        return cfg.project_root / "data" / "splits" / "hub_0.csv"

    def test_labeled_mode(self, sample_csv):
        ds = ChestXrayConsortiumDataset(sample_csv, mode="labeled")
        assert len(ds) > 0
        img, labels, sample_id = ds[0]

        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 224, 224)
        assert isinstance(labels, torch.Tensor)
        assert labels.shape == (5,)
        assert isinstance(sample_id, str)

    def test_unlabeled_dual_mode(self, sample_csv):
        ds = ChestXrayConsortiumDataset(sample_csv, mode="unlabeled_dual")
        assert len(ds) > 0
        weak_img, strong_img, sample_id = ds[0]

        assert isinstance(weak_img, torch.Tensor)
        assert weak_img.shape == (3, 224, 224)
        assert isinstance(strong_img, torch.Tensor)
        assert strong_img.shape == (3, 224, 224)
        # Weak and strong should produce distinct augmented views
        assert not torch.equal(weak_img, strong_img)

    def test_eval_mode(self, sample_csv):
        ds = ChestXrayConsortiumDataset(sample_csv, mode="eval")
        img1, _, _ = ds[0]
        img2, _, _ = ds[0]
        # In eval mode, transforms are deterministic
        assert torch.equal(img1, img2)


# ===========================================================================
# 3. Model Architecture, Opacus & FedBN Tests
# ===========================================================================

class TestConvNeXtFedBN:
    """Verify FedMedConvNeXt properties: Opacus DP-SGD, MC-Dropout, FedBN filtering."""

    @pytest.fixture(scope="class")
    def model(self):
        # pretrained=False for fast deterministic unit testing
        return FedMedConvNeXt(num_classes=5, pretrained=False, dropout_p=0.2)

    def test_opacus_compatibility(self, model):
        # Strict validation should produce 0 errors
        errors = ModuleValidator.validate(model.backbone, strict=True)
        assert len(errors) == 0, f"Opacus validation errors: {errors}"
        assert ModuleValidator.is_valid(model.backbone) is True

    def test_forward_pass_shapes(self, model):
        x = torch.randn(2, 3, 224, 224)
        logits = model(x)
        assert logits.shape == (2, 5)

        logits, features = model.forward_with_features(x)
        assert logits.shape == (2, 5)
        assert features.shape[0] == 2
        assert features.shape[1] == model.num_features

    def test_mc_dropout_uncertainty(self, model):
        x = torch.randn(2, 3, 224, 224)
        mean_probs, entropy = model.predict_with_uncertainty(x, num_passes=5)

        assert mean_probs.shape == (2, 5)
        assert entropy.shape == (2, 5)
        assert torch.all(mean_probs >= 0.0) and torch.all(mean_probs <= 1.0)
        assert torch.all(entropy >= 0.0)

    def test_fedbn_state_dict_filtering(self, model):
        shared_dict = get_shared_state_dict(model)
        full_dict = model.state_dict()

        # Shared dict must be strictly smaller than full dict (excluding local norm layers)
        assert len(shared_dict) < len(full_dict)

        # Ensure no normalization weights exist in shared dict
        for k in shared_dict.keys():
            assert not is_norm_layer_param(k), f"Normalization layer found in shared state dict: {k}"

        # Ensure convolutional and linear weights ARE in shared dict
        has_conv_or_weight = any("weight" in k for k in shared_dict.keys())
        assert has_conv_or_weight

    def test_fedbn_load_shared_state_dict(self, model):
        shared_dict = get_shared_state_dict(model)

        # Mutate a shared linear weight
        with torch.no_grad():
            model.classifier.weight.fill_(0.1234)

        # Reload original shared dict
        load_shared_state_dict(model, shared_dict)
        assert not torch.allclose(model.classifier.weight, torch.full_like(model.classifier.weight, 0.1234))


# ===========================================================================
# 4. Clinical Metrics & Calibration Tests
# ===========================================================================

class TestClinicalMetrics:
    """Verify multi-label AUROC, AUPRC, F1, and ECE computation."""

    def test_multilabel_metrics_computation(self):
        class_names = ["Cardiomegaly", "Effusion", "Infiltration", "Pneumonia", "No Finding"]
        y_true = np.array([
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
        ])
        y_probs = np.array([
            [0.9, 0.1, 0.1, 0.0, 0.0],
            [0.8, 0.7, 0.2, 0.1, 0.0],
            [0.2, 0.8, 0.9, 0.1, 0.0],
            [0.1, 0.2, 0.7, 0.9, 0.0],
            [0.0, 0.1, 0.1, 0.0, 0.9],
            [0.1, 0.0, 0.0, 0.1, 0.8],
        ])

        metrics = compute_multilabel_metrics(y_true, y_probs, class_names)
        assert "macro_auroc" in metrics
        assert "macro_auprc" in metrics
        assert "macro_f1" in metrics
        assert metrics["macro_auroc"] >= 0.8

    def test_ece_computation(self):
        # Perfect calibration test
        y_true = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
        y_probs = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        ece_zero = compute_ece(y_true, y_probs, n_bins=10)
        assert ece_zero == 0.0

        # High miscalibration test
        y_probs_bad = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
        ece_high = compute_ece(y_true, y_probs_bad, n_bins=10)
        assert ece_high > 0.5


# ===========================================================================
# 5. Baseline Execution & Checkpointing Test
# ===========================================================================

class TestBaselineExecution:
    """Verify baseline training runs, outputs valid checkpoints and metrics JSON."""

    def test_baseline_checkpoints_and_metrics_generated(self):
        cfg = load_config()
        from experiments.train_baselines import main as run_baselines_main

        # Execute baselines for 1 synthetic epoch
        sys.argv = ["train_baselines.py", "--synthetic-dryrun", "--epochs", "1"]
        run_baselines_main()

        # Checkpoints check
        ckpt_hub_0 = cfg.project_root / "results" / "checkpoints" / "isolated_hub_0.pt"
        ckpt_oracle = cfg.project_root / "results" / "checkpoints" / "centralized_oracle.pt"
        metrics_file = cfg.project_root / "results" / "metrics" / "phase1_baselines.json"

        assert ckpt_hub_0.exists(), "isolated_hub_0.pt was not saved"
        assert ckpt_oracle.exists(), "centralized_oracle.pt was not saved"
        assert metrics_file.exists(), "phase1_baselines.json was not saved"

        with metrics_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        assert "baselines" in data
        assert "isolated_hub_0" in data["baselines"]
        assert "centralized_oracle" in data["baselines"]
        assert "macro_auroc" in data["baselines"]["isolated_hub_0"]["validation_metrics"]
        assert "ece" in data["baselines"]["isolated_hub_0"]["validation_metrics"]


# ===========================================================================
# 6. Real Clinical Data Split Integrity Tests
# ===========================================================================

class TestRealDataIntegrity:
    """Verify real clinical splits, non-empty cohorts, and valid bounding box annotations."""

    @pytest.fixture(scope="class", autouse=True)
    def setup_real_splits(self):
        cfg = load_config()
        from data.prepare_consortium import prepare_real_consortium
        data_proc = cfg.project_root / "data" / "processed"
        splits_dir = cfg.project_root / "data" / "splits"
        prepare_real_consortium(data_proc, splits_dir, config=cfg)

    def test_real_splits_non_empty(self):
        cfg = load_config()
        splits_dir = cfg.project_root / "data" / "splits"
        for split_name in ["hub_0.csv", "hub_1.csv", "val_test.csv", "external_test.csv"]:
            csv_path = splits_dir / split_name
            assert csv_path.exists(), f"Split missing: {split_name}"
            df = pd.read_csv(csv_path)
            assert len(df) > 0, f"Split is empty: {split_name}"

    def test_real_bbox_eval_integrity(self):
        cfg = load_config()
        bbox_csv = cfg.project_root / "data" / "splits" / "bbox_eval.csv"
        assert bbox_csv.exists(), "bbox_eval.csv missing"
        df_bbox = pd.read_csv(bbox_csv)
        assert len(df_bbox) > 0, "bbox_eval.csv has 0 records"

        # Verify box coordinates are positive and non-zero
        assert (df_bbox["bbox_w"] > 0).all(), "Bounding box width must be > 0"
        assert (df_bbox["bbox_h"] > 0).all(), "Bounding box height must be > 0"
        assert (df_bbox["bbox_x"] >= 0).all(), "Bounding box x must be >= 0"
        assert (df_bbox["bbox_y"] >= 0).all(), "Bounding box y must be >= 0"

    def test_real_images_exist_on_disk(self):
        cfg = load_config()
        splits_dir = cfg.project_root / "data" / "splits"
        proc_dir = cfg.project_root / "data" / "processed"

        hub_0_df = pd.read_csv(splits_dir / "hub_0.csv")
        sample_path = hub_0_df["image_path"].iloc[0]
        full_path = proc_dir / sample_path
        assert full_path.exists(), f"Real image file not found on disk: {full_path}"

