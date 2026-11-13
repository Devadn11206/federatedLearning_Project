"""
FedMedVision — EU AI Act Cryptographic Audit Ledger
Maintains an immutable, append-only SHA-256 hash-chained ledger:
- Records round events, participating hospital nodes, DP epsilon/delta budget,
  Byzantine defense alerts, and global checkpoint SHA-256 digest
- Provides verify_chain_integrity() returning (is_valid, tampered_line)
"""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from utils.config_loader import load_config
from utils.logger import get_logger

logger = get_logger("fedmedvision.security.audit")


class EUAIActAuditLedger:
    """
    Cryptographic audit ledger for EU AI Act High-Risk Medical AI Compliance (Articles 12 & 15).
    """

    GENESIS_HASH = "0000000000000000000000000000000000000000000000000000000000000000"

    def __init__(self, ledger_file: str | Path | None = None) -> None:
        config = load_config()
        self.ledger_file = Path(
            ledger_file or (config.project_root / "results" / "audit" / "eu_ai_act_ledger.jsonl")
        )
        self.ledger_file.parent.mkdir(parents=True, exist_ok=True)

    def _compute_sha256(self, data: str | bytes) -> str:
        if isinstance(data, str):
            data = data.encode("utf-8")
        return hashlib.sha256(data).hexdigest()

    def get_latest_hash(self) -> str:
        if not self.ledger_file.exists() or self.ledger_file.stat().st_size == 0:
            return self.GENESIS_HASH

        last_line = ""
        with self.ledger_file.open("r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    last_line = line_str

        if not last_line:
            return self.GENESIS_HASH

        try:
            entry = json.loads(last_line)
            return entry.get("record_hash", self.GENESIS_HASH)
        except Exception:
            return self.GENESIS_HASH

    def append_audit_entry(
        self,
        round_id: int,
        event_type: str,
        participating_nodes: list[str],
        privacy_budget: dict[str, Any],
        byzantine_status: dict[str, Any],
        green_metrics: dict[str, Any],
        model_checkpoint_path: str | Path | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        """
        Appends a hash-chained tamper-evident entry to the audit ledger.
        """
        prev_hash = self.get_latest_hash()
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Compute checkpoint SHA-256 if file provided
        checkpoint_hash = None
        if model_checkpoint_path:
            ckpt_p = Path(model_checkpoint_path)
            if ckpt_p.exists():
                with ckpt_p.open("rb") as f:
                    checkpoint_hash = hashlib.sha256(f.read()).hexdigest()

        payload = {
            "timestamp": timestamp,
            "round_id": round_id,
            "event_type": event_type,
            "participating_nodes": participating_nodes,
            "privacy_budget": privacy_budget,
            "byzantine_status": byzantine_status,
            "green_metrics": green_metrics,
            "checkpoint_sha256": checkpoint_hash,
            "notes": notes,
            "prev_hash": prev_hash,
        }

        # Canonical serialized payload string
        payload_serialized = json.dumps(payload, sort_keys=True)
        record_hash = self._compute_sha256(payload_serialized)
        payload["record_hash"] = record_hash

        with self.ledger_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

        logger.info(f"Audit Ledger: Appended record {record_hash[:12]} (Round {round_id}, Prev {prev_hash[:12]})")
        return payload

    def read_ledger(self) -> list[dict[str, Any]]:
        """Read all parsed JSON entries from the ledger."""
        if not self.ledger_file.exists():
            return []
        records = []
        with open(self.ledger_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records

    def verify_chain_integrity(self) -> tuple[bool, int | None]:
        """
        Verifies the cryptographic chain integrity.
        Returns:
            (True, None) if completely valid.
            (False, line_index) if tampering is detected.
        """
        if not self.ledger_file.exists():
            return True, None

        expected_prev_hash = self.GENESIS_HASH

        with self.ledger_file.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line_str = line.strip()
                if not line_str:
                    continue

                try:
                    record = json.loads(line_str)
                except Exception:
                    return False, line_idx

                # Check prev_hash link
                actual_prev_hash = record.get("prev_hash")
                if actual_prev_hash != expected_prev_hash:
                    logger.error(f"Audit Chain Broken at line {line_idx}: expected prev {expected_prev_hash[:10]}, got {actual_prev_hash[:10]}")
                    return False, line_idx

                # Check record_hash matches payload
                claimed_hash = record.get("record_hash")
                payload_copy = {k: v for k, v in record.items() if k != "record_hash"}
                recomputed_hash = self._compute_sha256(json.dumps(payload_copy, sort_keys=True))

                if claimed_hash != recomputed_hash:
                    logger.error(f"Audit Record Altered at line {line_idx}: claimed {claimed_hash[:10]}, computed {recomputed_hash[:10]}")
                    return False, line_idx

                expected_prev_hash = claimed_hash

        return True, None

