"""Compilation state: the persistent latent node ``z`` (context.txt T9).

``z_t = E(x_t, c)`` — observed source ``x`` plus build context ``c`` —
with pure in-memory transitions ``F(z, a)`` implemented by
`environment.VirtualCompiler`. Feature computation lives in
`features.py`; this module owns identity, persistence, and the DAG node.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .features import FEATURE_KEYS, extract_features, feature_vector

__all__ = ["CompilationState", "source_hash", "extract_features",
           "FEATURE_KEYS", "feature_vector"]


def source_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


@dataclass
class CompilationState:
    """One node in the virtual compilation DAG."""

    source_text: str
    flags: tuple[str, ...] = ("-O2",)
    policy_text: str | None = None
    label: str = "root"
    parent: str | None = None
    action: str = "init"
    # oracle observations (filled by validate(); None == not yet measured)
    build_ok: bool | None = None
    binary_size: int | None = None
    binary_sha256: str | None = None
    runtime_ms: float | None = None
    telemetry: dict[str, float] = field(default_factory=dict)
    validated: bool = False
    # kept-binary + equivalence (differential gate vs parent binary)
    binary_path: str | None = None
    equivalent: bool | None = None

    def __post_init__(self) -> None:
        self.flags = tuple(self.flags)

    @property
    def state_id(self) -> str:
        h = hashlib.sha256()
        h.update(self.source_text.encode("utf-8", errors="replace"))
        h.update(b"\x00" + "|".join(self.flags).encode())
        h.update(b"\x00" + (self.policy_text or "").encode())
        return h.hexdigest()[:12]

    @property
    def features(self) -> dict[str, float]:
        feats = extract_features(self.source_text, self.flags,
                                 self.policy_text)
        feats.update(self.telemetry)  # real IR facts override cheap ones
        return feats

    def load_source_file(self, path: str | Path) -> "CompilationState":
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        self.source_text = text
        self.validated = False
        self.build_ok = None
        return self

    # -- persistence ----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "source_text": self.source_text,
            "flags": list(self.flags),
            "policy_text": self.policy_text,
            "label": self.label,
            "parent": self.parent,
            "action": self.action,
            "build_ok": self.build_ok,
            "binary_size": self.binary_size,
            "binary_sha256": self.binary_sha256,
            "runtime_ms": self.runtime_ms,
            "telemetry": dict(self.telemetry),
            "validated": self.validated,
            "binary_path": self.binary_path,
            "equivalent": self.equivalent,
        }

    @classmethod
    def from_dict(cls, rec: dict[str, Any]) -> "CompilationState":
        return cls(
            source_text=rec["source_text"],
            flags=tuple(rec.get("flags", ("-O2",))),
            policy_text=rec.get("policy_text"),
            label=rec.get("label", "root"),
            parent=rec.get("parent"),
            action=rec.get("action", "init"),
            build_ok=rec.get("build_ok"),
            binary_size=rec.get("binary_size"),
            binary_sha256=rec.get("binary_sha256"),
            runtime_ms=rec.get("runtime_ms"),
            telemetry=dict(rec.get("telemetry", {})),
            validated=bool(rec.get("validated", False)),
            binary_path=rec.get("binary_path"),
            equivalent=rec.get("equivalent"),
        )
