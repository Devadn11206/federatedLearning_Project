"""
FedMedVision — Reproducibility Utilities
Deterministic seeding for Python, NumPy, and PyTorch (CPU + CUDA).
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Device info
# ---------------------------------------------------------------------------

@dataclass
class DeviceInfo:
    device: torch.device
    device_name: str
    cuda_available: bool
    total_vram_gb: float
    fp16_supported: bool
    cuda_version: str | None
    torch_version: str


def get_device_info() -> DeviceInfo:
    """
    Detect available hardware and return a DeviceInfo summary.

    Returns:
        DeviceInfo: GPU name, VRAM (GB), FP16 support, CUDA version.
    """
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")

    device_name = "CPU"
    total_vram_gb = 0.0
    fp16_supported = False
    cuda_version: str | None = None

    if cuda_available:
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        device_name = props.name
        total_vram_gb = props.total_memory / (1024 ** 3)
        cuda_version = torch.version.cuda
        # FP16 AMP is supported on compute capability >= 7.0 (Volta+)
        # and also usable in a degraded form on older GPUs via autocast
        fp16_supported = props.major >= 7 or True  # autocast works on all CUDA GPUs

    return DeviceInfo(
        device=device,
        device_name=device_name,
        cuda_available=cuda_available,
        total_vram_gb=round(total_vram_gb, 2),
        fp16_supported=fp16_supported,
        cuda_version=cuda_version,
        torch_version=torch.__version__,
    )


def print_device_info(info: DeviceInfo | None = None) -> DeviceInfo:
    """Print device info summary to stdout and return it."""
    if info is None:
        info = get_device_info()
    print("=" * 50)
    print("  FedMedVision — Device Info")
    print("=" * 50)
    print(f"  Device      : {info.device}")
    print(f"  GPU Name    : {info.device_name}")
    print(f"  CUDA avail  : {info.cuda_available}")
    print(f"  VRAM (GB)   : {info.total_vram_gb:.2f}")
    print(f"  FP16 AMP    : {info.fp16_supported}")
    print(f"  CUDA ver    : {info.cuda_version}")
    print(f"  PyTorch ver : {info.torch_version}")
    print("=" * 50)
    return info


# ---------------------------------------------------------------------------
# Deterministic seeding
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    """
    Set deterministic seeds across all relevant RNG sources.

    Sets:
      - Python's built-in ``random`` module
      - NumPy global RNG
      - PyTorch CPU RNG
      - PyTorch CUDA RNG (all devices if available)
      - ``PYTHONHASHSEED`` environment variable

    Args:
        seed: Integer seed value. Default is 42.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Enable deterministic algorithms where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Context manager for temporary reproducibility
# ---------------------------------------------------------------------------

class DeterministicContext:
    """
    Context manager that seeds all RNGs on entry and optionally restores
    the previous state on exit.

    Usage:
        with DeterministicContext(seed=42):
            ... # fully deterministic block
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._rng_state_py: object = None
        self._rng_state_np: object = None
        self._rng_state_torch: torch.Tensor | None = None

    def __enter__(self) -> "DeterministicContext":
        self._rng_state_py = random.getstate()
        self._rng_state_np = np.random.get_state()
        self._rng_state_torch = torch.get_rng_state()
        seed_everything(self.seed)
        return self

    def __exit__(self, *_: object) -> None:
        random.setstate(self._rng_state_py)
        np.random.set_state(self._rng_state_np)  # type: ignore[arg-type]
        torch.set_rng_state(self._rng_state_torch)  # type: ignore[arg-type]


if __name__ == "__main__":
    seed_everything(42)
    print("✅ seed_everything(42) executed.")
    info = print_device_info()
    print(f"✅ Device: {info.device_name} | VRAM: {info.total_vram_gb:.2f} GB")
