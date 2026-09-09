"""Versioned JSONL audit trail for repair runs.

The architecture document requires every automatic edit to be traceable
to either GCC or a named model candidate.  Console output disappears;
this module gives each repair run an append-only, machine-readable event
stream covering configuration, evidence sources, gate outcomes,
promotions, rollbacks, and the final status.  A ``None`` path disables
the writer completely so default behavior stays unchanged.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

SCHEMA = "gcc-ai.repair-journal.v1"


class JournalWriter:
    def __init__(self, path: str | Path | None) -> None:
        self._path = Path(path) if path is not None else None

    @property
    def enabled(self) -> bool:
        return self._path is not None

    def emit(self, event: str, **fields: object) -> None:
        if self._path is None:
            return
        record = {
            "schema": SCHEMA,
            "event": event,
            "timestamp": time.time(),
            **fields,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
