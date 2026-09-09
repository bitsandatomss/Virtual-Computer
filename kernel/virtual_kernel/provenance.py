"""provenance.py — research-grade audit: run manifests + model cards.

learned_kernel.telemetry.provenance covers per-decision audit. This module
covers the EXPERIMENT layer: every benchmark/experiment writes a RunManifest
(config hash, dataset hash, model hashes, results) into a hash-chained
manifest log, and can emit a Model Card for a trained ensemble. Any number
in any report traces back to code + config + data.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import learned_kernel

CODE_VERSION = getattr(learned_kernel, "__version__", "0.0.0")


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


@dataclass
class RunManifest:
    name: str
    config_hash: str
    dataset_hash: str
    model_hashes: List[str]
    results: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    code_version: str = CODE_VERSION
    seq_no: int = 0
    prev_hash: str = "GENESIS"
    record_hash: str = ""

    def seal(self) -> str:
        body = {k: v for k, v in asdict(self).items() if k != "record_hash"}
        self.record_hash = hashlib.sha256(_canonical(body).encode()).hexdigest()
        return self.record_hash

    def to_dict(self) -> dict:
        return asdict(self)


class ManifestLog:
    """Append-only hash-chained log of run manifests."""

    def __init__(self, path: str = "reports/manifests.jsonl"):
        self.path = path
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        self._seq, self._prev = self._recover()

    def _recover(self) -> Tuple[int, str]:
        seq, prev = 0, "GENESIS"
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    seq = int(rec.get("seq_no", seq + 1))
                    prev = rec.get("record_hash", prev)
        except FileNotFoundError:
            pass
        return seq, prev

    def append(self, manifest: RunManifest) -> RunManifest:
        self._seq += 1
        manifest.seq_no = self._seq
        manifest.prev_hash = self._prev
        manifest.seal()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(_canonical(manifest.to_dict()) + "\n")
        self._prev = manifest.record_hash
        return manifest

    @staticmethod
    def verify(path: str) -> Tuple[bool, int, str]:
        expected_prev, expected_seq = "GENESIS", 0
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    expected_seq += 1
                    if rec.get("seq_no") != expected_seq:
                        return False, expected_seq, "sequence gap/reorder"
                    if rec.get("prev_hash") != expected_prev:
                        return False, expected_seq, "broken prev-hash link"
                    stored = rec.get("record_hash")
                    body = {k: v for k, v in rec.items() if k != "record_hash"}
                    if stored != hashlib.sha256(_canonical(body).encode()).hexdigest():
                        return False, expected_seq, "record content tampered"
                    expected_prev = stored
        except FileNotFoundError:
            return False, 0, "log file missing"
        return True, -1, "chain intact"


def ensemble_hash(ensemble) -> str:
    h = hashlib.sha256()
    for m in ensemble.members:
        for k in ("E", "W1", "W2", "W3", "D", "V"):
            h.update(memoryview(getattr(m, k).tobytes()))
    return h.hexdigest()[:16]


def write_model_card(path: str, ensemble, config, dataset_hash: str,
                     eval_summary: Dict[str, Any]) -> str:
    card = f"""# Virtual Kernel — Model Card

- ensemble members: {len(ensemble)}
- member seeds: {[m.seed for m in ensemble.members]}
- config hash: {config.hash}
- dataset hash: {dataset_hash}
- code version: {CODE_VERSION}

## Intended use
Counterfactual search over scheduler `target_latency_us` settings. Proposals
only; the deterministic `PolicyValidator` remains the actuation authority.

## Evaluation summary
```json
{_canonical(eval_summary)}
```

## Limitations
- Scheduler vertical slice only (VM/network dynamics not learned).
- Simulator-trained: real-kernel replay/shadow validation still required.
- Uncertainty is ensemble disagreement: uncalibrated outside the training
  workload distribution by construction.
"""
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(card)
    return path
