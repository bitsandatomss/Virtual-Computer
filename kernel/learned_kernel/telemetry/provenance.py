"""
telemetry/provenance.py — Tamper-evident audit trail.

Every record is chained:  record_hash = SHA256(prev_hash ‖ canonical(record)).
Deleting, reordering, or editing any historical record breaks every hash that
follows it, and `verify_chain()` pinpoints the first corrupted sequence
number. The previous revision wrote unchained JSON lines with truncated
64-bit state hashes — trivially forgeable despite the "cryptographic" claim.
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from typing import List, Optional, Tuple

from ..policy.schemas import PolicyAction, KernelIntermediateRepresentation


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


class ProvenanceLogger:
    """Append-only JSONL audit log with a per-record hash chain."""

    def __init__(self, log_file: str = "learned_kernel_audit.jsonl", fsync: bool = False):
        self.log_file = log_file
        self.fsync = fsync
        self._seq = 0
        self._prev_hash = "GENESIS"
        self._recover_tail()

    def _recover_tail(self) -> None:
        """Resume an existing chain (last seq + hash) if the file exists."""
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    self._seq = int(rec.get("seq_no", self._seq + 1))
                    self._prev_hash = rec.get("record_hash", self._prev_hash)
        except FileNotFoundError:
            pass

    def _hash_state(self, kstate: KernelIntermediateRepresentation) -> str:
        return hashlib.sha256(kstate.model_dump_json().encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------ #

    def record_decision(
        self,
        kstate: KernelIntermediateRepresentation,
        action: Optional[PolicyAction],
        policy_id: str,
        policy_version: str,
        is_validated: bool,
        validator_error: str = "",
        reward: float = 0.0,
    ) -> str:
        self._seq += 1
        record = {
            "seq_no": self._seq,
            "timestamp": time.time(),
            "kernel_state_hash": self._hash_state(kstate),
            "policy_id": policy_id,
            "policy_version": policy_version,
            "action_proposed": action.model_dump() if action else None,
            "validator_approved": is_validated,
            "validator_error": validator_error,
            "observed_reward": reward,
            "hardware": platform.platform(),
            "prev_record_hash": self._prev_hash,
        }
        record_hash = hashlib.sha256(_canonical(record).encode()).hexdigest()
        record["record_hash"] = record_hash

        line = json.dumps(record)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            if self.fsync:
                f.flush()
                import os
                os.fsync(f.fileno())

        self._prev_hash = record_hash
        return record_hash

    # ------------------------------------------------------------------ #

    @staticmethod
    def verify_chain(log_file: str) -> Tuple[bool, int, str]:
        """Verify the whole chain. Returns (ok, first_bad_seq, reason)."""
        expected_prev = "GENESIS"
        expected_seq = 0
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    expected_seq += 1
                    if rec.get("seq_no") != expected_seq:
                        return False, expected_seq, "sequence gap/reorder"
                    if rec.get("prev_record_hash") != expected_prev:
                        return False, expected_seq, "broken prev-hash link"
                    stored = rec.get("record_hash")
                    body = {k: v for k, v in rec.items() if k != "record_hash"}
                    recomputed = hashlib.sha256(_canonical(body).encode()).hexdigest()
                    if stored != recomputed:
                        return False, expected_seq, "record content tampered"
                    expected_prev = stored
        except FileNotFoundError:
            return False, 0, "log file missing"
        return True, -1, "chain intact"

    @staticmethod
    def read_records(log_file: str) -> List[dict]:
        out: List[dict] = []
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        out.append(json.loads(line))
        except FileNotFoundError:
            pass
        return out
