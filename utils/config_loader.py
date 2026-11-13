"""
FedMedVision — Configuration Loader
Merges consortium.yaml, training_fl.yaml, and security_green.yaml into a
single validated configuration object.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Root path helpers
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    """Return the absolute project root (parent of utils/)."""
    return Path(__file__).resolve().parent.parent


def _configs_dir() -> Path:
    return _project_root() / "configs"


# ---------------------------------------------------------------------------
# Raw YAML loader
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        raise ValueError(f"Configuration file is empty or invalid YAML: {path}")
    return data


# ---------------------------------------------------------------------------
# Typed sub-configs (lightweight dataclasses for IDE autocompletion)
# ---------------------------------------------------------------------------

@dataclass
class ConsortiumConfig:
    pathologies: list[str] = field(default_factory=list)
    image_size: int = 224
    dirichlet_alpha: float = 0.3
    seeds: list[int] = field(default_factory=lambda: [42, 123, 2026])
    nodes: dict[str, Any] = field(default_factory=dict)
    external_test: dict[str, Any] = field(default_factory=dict)
    _raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class TrainingConfig:
    backbone: str = "convnext_tiny"
    pretrained: bool = True
    dropout_p: float = 0.2
    fedbn_local_norm: bool = True
    num_rounds: int = 25
    local_epochs: int = 2
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-4
    use_amp: bool = True
    max_batch_size: int = 32
    max_num_workers: int = 2
    fedprox_mu: float = 0.01
    fedproto_lambda: float = 0.1
    semi_supervised: dict[str, Any] = field(default_factory=dict)
    num_classes: int = 5
    _raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class SecurityGreenConfig:
    differential_privacy: dict[str, Any] = field(default_factory=dict)
    secure_aggregation: dict[str, Any] = field(default_factory=dict)
    byzantine_defense: dict[str, Any] = field(default_factory=dict)
    green_ai: dict[str, Any] = field(default_factory=dict)
    calibration: dict[str, Any] = field(default_factory=dict)
    _raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class FedMedVisionConfig:
    consortium: ConsortiumConfig = field(default_factory=ConsortiumConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    security: SecurityGreenConfig = field(default_factory=SecurityGreenConfig)
    project_root: Path = field(default_factory=_project_root)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_consortium(raw: dict[str, Any]) -> ConsortiumConfig:
    pathologies = raw.get("pathologies", [])
    if not pathologies:
        raise ValueError("consortium.yaml must specify at least one pathology.")
    nodes = raw.get("nodes", {})
    if len(nodes) != 6:
        raise ValueError(
            f"consortium.yaml must define exactly 6 nodes; found {len(nodes)}."
        )
    for node_id, node_cfg in nodes.items():
        required_keys = {"name", "role", "dataset", "sample_count", "simulated_latency_ms"}
        missing = required_keys - set(node_cfg.keys())
        if missing:
            raise ValueError(
                f"Node '{node_id}' is missing required keys: {missing}"
            )
    return ConsortiumConfig(
        pathologies=pathologies,
        image_size=raw.get("image_size", 224),
        dirichlet_alpha=raw.get("dirichlet_alpha", 0.3),
        seeds=raw.get("seeds", [42, 123, 2026]),
        nodes=nodes,
        external_test=raw.get("external_test", {}),
        _raw=raw,
    )


def _validate_training(raw: dict[str, Any]) -> TrainingConfig:
    batch_size = raw.get("batch_size", 16)
    max_batch_size = raw.get("max_batch_size", 32)
    if batch_size > max_batch_size:
        raise ValueError(
            f"batch_size ({batch_size}) exceeds AGENT.md max_batch_size ({max_batch_size})."
        )
    num_workers = raw.get("max_num_workers", 2)
    if num_workers > 2:
        raise ValueError(
            f"max_num_workers ({num_workers}) exceeds AGENT.md limit of 2."
        )
    return TrainingConfig(
        backbone=raw.get("backbone", "convnext_tiny"),
        pretrained=raw.get("pretrained", True),
        dropout_p=raw.get("dropout_p", 0.2),
        fedbn_local_norm=raw.get("fedbn_local_norm", True),
        num_rounds=raw.get("num_rounds", 25),
        local_epochs=raw.get("local_epochs", 2),
        batch_size=batch_size,
        lr=raw.get("lr", 3e-4),
        weight_decay=raw.get("weight_decay", 1e-4),
        use_amp=raw.get("use_amp", True),
        max_batch_size=max_batch_size,
        max_num_workers=num_workers,
        fedprox_mu=raw.get("fedprox_mu", 0.01),
        fedproto_lambda=raw.get("fedproto_lambda", 0.1),
        semi_supervised=raw.get("semi_supervised", {}),
        num_classes=raw.get("num_classes", 5),
        _raw=raw,
    )


def _validate_security(raw: dict[str, Any]) -> SecurityGreenConfig:
    dp = raw.get("differential_privacy", {})
    epsilon_sweep = dp.get("epsilon_sweep", [])
    if dp.get("enabled", False) and not epsilon_sweep:
        raise ValueError(
            "differential_privacy is enabled but epsilon_sweep is empty."
        )
    return SecurityGreenConfig(
        differential_privacy=dp,
        secure_aggregation=raw.get("secure_aggregation", {}),
        byzantine_defense=raw.get("byzantine_defense", {}),
        green_ai=raw.get("green_ai", {}),
        calibration=raw.get("calibration", {}),
        _raw=raw,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(configs_dir: Path | str | None = None) -> FedMedVisionConfig:
    """
    Load and validate all three FedMedVision YAML configuration files.

    Args:
        configs_dir: Path to the configs/ directory. Defaults to the standard
                     location relative to this file.

    Returns:
        FedMedVisionConfig: Merged, validated configuration object.

    Raises:
        FileNotFoundError: If any YAML file is missing.
        ValueError: If any validation check fails.
    """
    if configs_dir is None:
        configs_dir = _configs_dir()
    configs_dir = Path(configs_dir)

    consortium_raw = _load_yaml(configs_dir / "consortium.yaml")
    training_raw = _load_yaml(configs_dir / "training_fl.yaml")
    security_raw = _load_yaml(configs_dir / "security_green.yaml")

    consortium = _validate_consortium(consortium_raw)
    training = _validate_training(training_raw)
    security = _validate_security(security_raw)

    return FedMedVisionConfig(
        consortium=consortium,
        training=training,
        security=security,
        project_root=_project_root(),
    )


def get_results_dir(cfg: FedMedVisionConfig, subdir: str = "") -> Path:
    """Return the results directory, optionally with a subdirectory."""
    path = cfg.project_root / "results" / subdir if subdir else cfg.project_root / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


if __name__ == "__main__":
    cfg = load_config()
    print(f"✅ Config loaded successfully.")
    print(f"   Pathologies : {cfg.consortium.pathologies}")
    print(f"   Backbone    : {cfg.training.backbone}")
    print(f"   DP enabled  : {cfg.security.differential_privacy.get('enabled')}")
    print(f"   Project root: {cfg.project_root}")
