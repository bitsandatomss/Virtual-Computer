"""Versioned evidence log: the six stages as literal event types.

context.txt T7: ``observe → learn → represent → branch → intervene →
validate``. Every run of the virtual compiler appends one JSON record per
stage transition, so any result can be replayed stage by stage — the same
audit discipline as ``gcc-ai.repair-journal.v1`` and
``gcc-ai.policy-lab.v1`` in the oracle layer.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "virtual-compiler.evidence.v1"

STAGES = ("observe", "learn", "represent", "branch", "intervene", "validate")


class EvidenceLog:
    def __init__(self, path: str | Path | None = None,
                 session: str | None = None) -> None:
        self.path = Path(path) if path else None
        self.session = session or uuid.uuid4().hex[:8]
        self.records: list[dict[str, Any]] = []
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, stage: str, event: str, **payload: Any) -> dict[str, Any]:
        if stage not in STAGES:
            raise ValueError(f"unknown evidence stage: {stage!r}")
        rec = {
            "schema": SCHEMA,
            "session": self.session,
            "ts": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "event": event,
            "payload": payload,
        }
        self.records.append(rec)
        if self.path is not None:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
        return rec

    def __len__(self) -> int:
        return len(self.records)

    @classmethod
    def read(cls, path: str | Path) -> list[dict[str, Any]]:
        out = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out
