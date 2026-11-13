"""
FedMedVision — Structured Logger
JSON + console (Rich) logging for federated round events and system telemetry.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Optional Rich support (graceful fallback to stdlib if not installed)
# ---------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.logging import RichHandler
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Default log directory
# ---------------------------------------------------------------------------

def _default_log_path() -> Path:
    root = Path(__file__).resolve().parent.parent
    log_dir = root / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "system.log"


# ---------------------------------------------------------------------------
# JSON log handler
# ---------------------------------------------------------------------------

class _JsonFileHandler(logging.FileHandler):
    """Writes each log record as a single JSON line to the log file."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            payload: dict[str, Any] = {
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            # Include any extra fields attached to the record
            for key, val in record.__dict__.items():
                if key.startswith("_fed_") or key.startswith("fed_"):
                    payload[key] = val

            line = json.dumps(payload, default=str)
            self.stream.write(line + "\n")
            self.stream.flush()
        except Exception:  # noqa: BLE001
            self.handleError(record)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def get_logger(
    name: str = "fedmedvision",
    log_file: Path | str | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Build a structured logger writing JSON lines to *log_file* and
    formatted output to the console (using Rich if available).

    Args:
        name:     Logger name (default: ``"fedmedvision"``).
        log_file: Path to the JSONL log file. Defaults to
                  ``results/logs/system.log``.
        level:    Logging level (default: ``logging.INFO``).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    # --- Console handler (Rich or plain) ---
    if _RICH_AVAILABLE:
        console_handler = RichHandler(
            rich_tracebacks=True,
            show_path=False,
            markup=True,
        )
        console_handler.setLevel(level)
        fmt = logging.Formatter("%(message)s", datefmt="[%X]")
        console_handler.setFormatter(fmt)
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        console_handler.setFormatter(fmt)

    logger.addHandler(console_handler)

    # --- JSON file handler ---
    log_path = Path(log_file) if log_file else _default_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    json_handler = _JsonFileHandler(str(log_path), mode="a", encoding="utf-8")
    json_handler.setLevel(level)
    logger.addHandler(json_handler)

    return logger


# ---------------------------------------------------------------------------
# Convenience: federated-round structured logging helpers
# ---------------------------------------------------------------------------

def log_round_event(
    logger: logging.Logger,
    round_num: int,
    event: str,
    **kwargs: Any,
) -> None:
    """
    Log a federated learning round event with structured metadata.

    Args:
        logger:    Logger instance from :func:`get_logger`.
        round_num: Current federated round number.
        event:     Short event label (e.g., ``"round_start"``, ``"aggregation"``).
        **kwargs:  Additional key-value pairs to include in the JSON record.
    """
    extra: dict[str, Any] = {
        "fed_round": round_num,
        "fed_event": event,
        **{f"fed_{k}": v for k, v in kwargs.items()},
    }
    logger.info(
        f"[Round {round_num:03d}] {event}",
        extra=extra,
    )


def log_system_event(
    logger: logging.Logger,
    event: str,
    **kwargs: Any,
) -> None:
    """
    Log a system-level event (startup, shutdown, GPU info, etc.).

    Args:
        logger: Logger instance from :func:`get_logger`.
        event:  Short event label (e.g., ``"startup"``, ``"gpu_info"``).
        **kwargs: Additional key-value pairs.
    """
    extra: dict[str, Any] = {
        "fed_event": event,
        **{f"fed_{k}": v for k, v in kwargs.items()},
    }
    logger.info(f"[SYSTEM] {event}", extra=extra)


if __name__ == "__main__":
    _log = get_logger()
    log_system_event(_log, "startup", version="0.1.0")
    log_round_event(_log, round_num=1, event="round_start", num_clients=6)
    print("✅ Logger initialized and test events written.")
