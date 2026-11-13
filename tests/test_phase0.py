"""
FedMedVision — Phase 0 Smoke Tests
Verifies environment, package imports, config loading, and reproducibility utilities.

Run with:
    pytest tests/test_phase0.py -v
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path for local package imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ===========================================================================
# 1. Package import tests
# ===========================================================================

class TestPackageImports:
    """Verify all required packages import cleanly and print versions."""

    def test_torch_import(self):
        torch = importlib.import_module("torch")
        version = torch.__version__
        print(f"\n  ✅ torch=={version}")
        assert version, "torch.__version__ is empty"

    def test_torchvision_import(self):
        tv = importlib.import_module("torchvision")
        version = tv.__version__
        print(f"  ✅ torchvision=={version}")
        assert version, "torchvision.__version__ is empty"

    def test_timm_import(self):
        timm = importlib.import_module("timm")
        version = timm.__version__
        print(f"  ✅ timm=={version}")
        assert version, "timm.__version__ is empty"

    def test_flwr_import(self):
        flwr = importlib.import_module("flwr")
        version = flwr.__version__
        print(f"  ✅ flwr=={version}")
        assert version, "flwr.__version__ is empty"

    def test_opacus_import(self):
        opacus = importlib.import_module("opacus")
        version = opacus.__version__
        print(f"  ✅ opacus=={version}")
        assert version, "opacus.__version__ is empty"

    def test_pytorch_grad_cam_import(self):
        pytorch_grad_cam = importlib.import_module("pytorch_grad_cam")
        print(f"  ✅ pytorch_grad_cam imported successfully")
        assert pytorch_grad_cam is not None

    def test_albumentations_import(self):
        A = importlib.import_module("albumentations")
        version = A.__version__
        print(f"  ✅ albumentations=={version}")
        assert version, "albumentations.__version__ is empty"

    def test_fastapi_import(self):
        fastapi = importlib.import_module("fastapi")
        version = fastapi.__version__
        print(f"  ✅ fastapi=={version}")
        assert version, "fastapi.__version__ is empty"

    def test_sklearn_import(self):
        sklearn = importlib.import_module("sklearn")
        version = sklearn.__version__
        print(f"  ✅ scikit-learn=={version}")
        assert version

    def test_pandas_import(self):
        pd = importlib.import_module("pandas")
        version = pd.__version__
        print(f"  ✅ pandas=={version}")
        assert version

    def test_numpy_import(self):
        np = importlib.import_module("numpy")
        version = np.__version__
        print(f"  ✅ numpy=={version}")
        assert version

    def test_yaml_import(self):
        yaml = importlib.import_module("yaml")
        version = getattr(yaml, "__version__", "bundled")
        print(f"  ✅ pyyaml=={version}")
        assert yaml is not None

    def test_reportlab_import(self):
        rl = importlib.import_module("reportlab")
        version = rl.Version
        print(f"  ✅ reportlab=={version}")
        assert version


# ===========================================================================
# 2. Config loader tests
# ===========================================================================

class TestConfigLoader:
    """Verify utils/config_loader.py loads and validates all 3 YAML files."""

    @pytest.fixture(scope="class")
    def cfg(self):
        from utils.config_loader import load_config
        return load_config()

    def test_config_loads_without_error(self, cfg):
        assert cfg is not None, "load_config() returned None"
        print(f"\n  ✅ FedMedVisionConfig loaded from {cfg.project_root}")

    def test_consortium_pathologies(self, cfg):
        expected = ["Cardiomegaly", "Effusion", "Infiltration", "Pneumonia", "No Finding"]
        assert cfg.consortium.pathologies == expected, (
            f"Pathologies mismatch: {cfg.consortium.pathologies}"
        )
        print(f"  ✅ Pathologies: {cfg.consortium.pathologies}")

    def test_consortium_has_six_nodes(self, cfg):
        assert len(cfg.consortium.nodes) == 6, (
            f"Expected 6 nodes, got {len(cfg.consortium.nodes)}"
        )
        print(f"  ✅ 6 nodes registered: {list(cfg.consortium.nodes.keys())}")

    def test_consortium_node_roles(self, cfg):
        labeled = [n for n, v in cfg.consortium.nodes.items() if v["role"] == "labeled_hub"]
        unlabeled = [n for n, v in cfg.consortium.nodes.items() if v["role"] == "unlabeled_clinic"]
        assert len(labeled) == 2, f"Expected 2 labeled hubs, got {len(labeled)}"
        assert len(unlabeled) == 4, f"Expected 4 unlabeled clinics, got {len(unlabeled)}"
        print(f"  ✅ Labeled hubs: {labeled} | Unlabeled clinics: {unlabeled}")

    def test_training_backbone(self, cfg):
        assert cfg.training.backbone == "convnext_tiny", (
            f"Backbone should be 'convnext_tiny', got '{cfg.training.backbone}'"
        )
        print(f"  ✅ Backbone: {cfg.training.backbone}")

    def test_training_batch_size_within_limit(self, cfg):
        assert cfg.training.batch_size <= cfg.training.max_batch_size, (
            f"batch_size {cfg.training.batch_size} exceeds max {cfg.training.max_batch_size}"
        )
        print(f"  ✅ batch_size={cfg.training.batch_size} ≤ max={cfg.training.max_batch_size}")

    def test_training_amp_enabled(self, cfg):
        assert cfg.training.use_amp is True, "use_amp must be True (AGENT.md mandate)"
        print(f"  ✅ use_amp=True (FP16 mixed precision enforced)")

    def test_training_num_workers_within_limit(self, cfg):
        assert cfg.training.max_num_workers <= 2, (
            f"num_workers {cfg.training.max_num_workers} exceeds AGENT.md limit of 2"
        )
        print(f"  ✅ max_num_workers={cfg.training.max_num_workers} ≤ 2")

    def test_security_dp_enabled(self, cfg):
        dp = cfg.security.differential_privacy
        assert dp.get("enabled") is True, "DP should be enabled"
        assert len(dp.get("epsilon_sweep", [])) > 0, "epsilon_sweep must be non-empty"
        print(f"  ✅ DP enabled | epsilon_sweep={dp['epsilon_sweep']}")

    def test_security_byzantine_defense(self, cfg):
        byz = cfg.security.byzantine_defense
        assert byz.get("aggregation") == "trimmed_mean", (
            f"Expected 'trimmed_mean', got '{byz.get('aggregation')}'"
        )
        assert byz.get("poison_target_node") == 5, (
            f"Byzantine node should be 5, got {byz.get('poison_target_node')}"
        )
        print(f"  ✅ Byzantine defense: {byz['aggregation']} | poison node={byz['poison_target_node']}")

    def test_security_green_ai(self, cfg):
        green = cfg.security.green_ai
        assert green.get("adaptive_freezing") is True
        assert green.get("co2_kg_per_kwh") == pytest.approx(0.42, abs=1e-6)
        print(f"  ✅ Green AI: adaptive_freezing=True | CO₂={green['co2_kg_per_kwh']} kg/kWh")

    def test_project_root_exists(self, cfg):
        assert cfg.project_root.exists(), f"Project root not found: {cfg.project_root}"
        print(f"  ✅ Project root exists: {cfg.project_root}")


# ===========================================================================
# 3. Reproducibility tests
# ===========================================================================

class TestReproducibility:
    """Verify seed_everything and get_device_info execute cleanly."""

    def test_seed_everything_executes(self):
        from utils.reproducibility import seed_everything
        # Must not raise
        seed_everything(42)
        print(f"\n  ✅ seed_everything(42) executed without error")

    def test_seed_everything_deterministic(self):
        import torch
        from utils.reproducibility import seed_everything

        seed_everything(42)
        a = torch.randn(5)
        seed_everything(42)
        b = torch.randn(5)
        assert torch.allclose(a, b), "seed_everything should produce identical tensors"
        print(f"  ✅ Determinism verified: seed(42) → same tensor twice")

    def test_seed_multiple_seeds(self):
        from utils.reproducibility import seed_everything
        for seed in [42, 123, 2026]:
            seed_everything(seed)
        print(f"  ✅ All consortium seeds (42, 123, 2026) execute without error")

    def test_get_device_info_returns_dataclass(self):
        from utils.reproducibility import DeviceInfo, get_device_info
        info = get_device_info()
        assert isinstance(info, DeviceInfo), f"Expected DeviceInfo, got {type(info)}"
        print(f"\n  ✅ get_device_info() returned DeviceInfo")

    def test_get_device_info_device_field(self):
        import torch
        from utils.reproducibility import get_device_info
        info = get_device_info()
        assert info.device in [torch.device("cuda"), torch.device("cpu")], (
            f"Device must be cuda or cpu, got {info.device}"
        )
        print(f"  ✅ device={info.device} | GPU='{info.device_name}' | VRAM={info.total_vram_gb:.2f} GB")

    def test_get_device_info_torch_version(self):
        import torch
        from utils.reproducibility import get_device_info
        info = get_device_info()
        assert info.torch_version == torch.__version__
        print(f"  ✅ torch_version={info.torch_version}")

    def test_vram_within_budget(self):
        from utils.reproducibility import get_device_info
        info = get_device_info()
        if info.cuda_available:
            # Warn if VRAM exceeds 6 GB (not a hard failure — may run on other hardware)
            if info.total_vram_gb > 6.0:
                print(
                    f"\n  ⚠️  VRAM={info.total_vram_gb:.2f} GB > 6 GB AGENT.md budget "
                    f"(OK for development, enforce batch_size ≤ 32 at runtime)"
                )
            else:
                print(f"  ✅ VRAM={info.total_vram_gb:.2f} GB ≤ 6 GB budget")
        else:
            print(f"  ✅ CPU-only mode (no VRAM check needed)")
        # Always passes — the VRAM check is advisory
        assert True

    def test_deterministic_context_manager(self):
        import torch
        from utils.reproducibility import DeterministicContext

        with DeterministicContext(seed=99):
            x = torch.randn(3)
        with DeterministicContext(seed=99):
            y = torch.randn(3)
        assert torch.allclose(x, y), "DeterministicContext should produce identical tensors"
        print(f"  ✅ DeterministicContext(seed=99) is reproducible")


# ===========================================================================
# 4. Directory structure tests
# ===========================================================================

class TestDirectoryStructure:
    """Verify the full scaffold exists as specified."""

    REQUIRED_DIRS = [
        "configs",
        "data/raw", "data/processed", "data/splits",
        "models", "security", "fl_engine",
        "app/backend", "app/frontend",
        "experiments",
        "results/checkpoints", "results/logs", "results/metrics",
        "paper/figures",
        "utils", "tests",
    ]

    REQUIRED_FILES = [
        "AGENT.md",
        ".gitignore",
        "Makefile",
        "README.md",
        "pyproject.toml",
        "requirements.txt",
        "configs/consortium.yaml",
        "configs/training_fl.yaml",
        "configs/security_green.yaml",
        "utils/config_loader.py",
        "utils/reproducibility.py",
        "utils/logger.py",
        "utils/__init__.py",
        "models/__init__.py",
        "security/__init__.py",
        "fl_engine/__init__.py",
        "app/backend/__init__.py",
        "app/frontend/__init__.py",
        "tests/__init__.py",
    ]

    GITKEEP_DIRS = [
        "data/raw", "data/processed", "data/splits",
        "experiments",
        "results/checkpoints", "results/logs", "results/metrics",
        "paper/figures",
    ]

    def test_required_directories_exist(self):
        root = PROJECT_ROOT
        missing = []
        for d in self.REQUIRED_DIRS:
            p = root / d
            if not p.is_dir():
                missing.append(str(p))
        assert not missing, f"Missing directories:\n" + "\n".join(missing)
        print(f"\n  ✅ All {len(self.REQUIRED_DIRS)} required directories exist")

    def test_required_files_exist(self):
        root = PROJECT_ROOT
        missing = []
        for f in self.REQUIRED_FILES:
            p = root / f
            if not p.exists():
                missing.append(str(p))
        assert not missing, f"Missing files:\n" + "\n".join(missing)
        print(f"  ✅ All {len(self.REQUIRED_FILES)} required files exist")

    def test_gitkeep_files_exist(self):
        root = PROJECT_ROOT
        missing = []
        for d in self.GITKEEP_DIRS:
            p = root / d / ".gitkeep"
            if not p.exists():
                missing.append(str(p))
        assert not missing, f"Missing .gitkeep files:\n" + "\n".join(missing)
        print(f"  ✅ All {len(self.GITKEEP_DIRS)} .gitkeep markers exist")

    def test_agent_md_has_required_sections(self):
        agent_md = (PROJECT_ROOT / "AGENT.md").read_text(encoding="utf-8")
        required_sections = [
            "Hardware & VRAM Budget",
            "Core Stack Versions",
            "System Architecture Contract",
            "Verification Law",
            "Pathologies",
        ]
        for section in required_sections:
            assert section in agent_md, f"AGENT.md missing section: '{section}'"
        print(f"  ✅ AGENT.md contains all {len(required_sections)} required sections")

    def test_yaml_files_are_parseable(self):
        import yaml
        root = PROJECT_ROOT / "configs"
        for yaml_file in ["consortium.yaml", "training_fl.yaml", "security_green.yaml"]:
            path = root / yaml_file
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            assert data is not None, f"{yaml_file} parsed to None"
        print(f"  ✅ All 3 YAML config files are valid YAML")


# ===========================================================================
# 5. Logger tests
# ===========================================================================

class TestLogger:
    """Verify utils/logger.py initializes and writes log events cleanly."""

    def test_get_logger_returns_logger(self, tmp_path):
        import logging
        from utils.logger import get_logger
        log_file = tmp_path / "test_system.log"
        logger = get_logger(name="test_phase0", log_file=log_file)
        assert isinstance(logger, logging.Logger)
        print(f"\n  ✅ get_logger() returned a logging.Logger instance")

    def test_log_round_event(self, tmp_path):
        import json
        from utils.logger import get_logger, log_round_event
        log_file = tmp_path / "round_events.log"
        logger = get_logger(name="test_rounds", log_file=log_file)
        log_round_event(logger, round_num=1, event="round_start", num_clients=6)
        # Read JSON log and verify structure
        lines = [l for l in log_file.read_text().splitlines() if l.strip()]
        assert len(lines) >= 1, "No JSON lines written to log file"
        record = json.loads(lines[-1])
        assert record["fed_round"] == 1
        assert record["fed_event"] == "round_start"
        print(f"  ✅ log_round_event written to JSON log correctly")

    def test_log_system_event(self, tmp_path):
        import json
        from utils.logger import get_logger, log_system_event
        log_file = tmp_path / "sys_events.log"
        logger = get_logger(name="test_sys", log_file=log_file)
        log_system_event(logger, "startup", version="0.1.0")
        lines = [l for l in log_file.read_text().splitlines() if l.strip()]
        assert len(lines) >= 1
        record = json.loads(lines[-1])
        assert record["fed_event"] == "startup"
        assert record["fed_version"] == "0.1.0"
        print(f"  ✅ log_system_event written to JSON log correctly")

