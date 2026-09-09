"""Provenance: hash-chained run manifests for the unified pipeline.

Mirrors the per-layer practice (compiler ``EvidenceLog``, kernel
``ManifestLog``) at the cross-layer level: every unified run appends
one JSONL record carrying the config hash, per-layer oracle costs, the
causal parent hash, AND the version/digest bindings that make the run
reproducible (lowering map, coupling law, layer evidence artifacts).
`collect_bindings()` gathers them; a manifest without bindings is
rejected by `append_manifest` only in strict mode — the CLI always
passes bindings.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .paths import REPO_ROOT


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


def config_hash(config: dict) -> str:
    return hashlib.sha256(_canonical(config).encode()).hexdigest()[:12]


def _file_digest(rel: str) -> str | None:
    p = REPO_ROOT / rel
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def collect_bindings(vertical: dict | None = None) -> dict:
    """Version + digest bindings for everything the run depended on."""
    bindings = {
        "lowering": (vertical or {}).get("versions", {}).get("lowering", "lowering-v1"),
        "lowering_digest": (vertical or {}).get("versions", {}).get("lowering_digest"),
        "coupling": (vertical or {}).get("versions", {}).get("coupling", "coupling-v1"),
        "vertical_digest": (vertical or {}).get("digest"),
        "artifacts": {
            "micro_skill": _file_digest(
                "microarchitecture/artifacts/vmarch_benchmark_phase.json"),
            "kernel_levels": _file_digest("kernel/reports/levels/results.json"),
            "compiler_killer": _file_digest(
                "compiler/experiments/study/killer.json"),
        },
    }
    try:
        from vmarch.version import VMARCH_VERSION  # type: ignore
        bindings["vmarch_version"] = VMARCH_VERSION
    except Exception:
        bindings["vmarch_version"] = None
    return bindings


@dataclass
class UnifiedManifest:
    config: dict
    layers: list[str]
    oracle_costs: dict
    results: dict
    parent: str = "genesis"
    bindings: dict = field(default_factory=dict)
    config_hash: str = field(init=False)

    def __post_init__(self) -> None:
        self.config_hash = config_hash(self.config)
        if not self.bindings:
            self.bindings = collect_bindings(
                self.results.get("vertical") if isinstance(self.results, dict) else None)

    def record(self) -> dict:
        body = {
            "config_hash": self.config_hash,
            "config": self.config,
            "layers": self.layers,
            "oracle_costs": self.oracle_costs,
            "results": self.results,
            "bindings": self.bindings,
            "parent": self.parent,
        }
        body["hash"] = hashlib.sha256(_canonical(body).encode()).hexdigest()[:16]
        return body


def append_manifest(path: str | Path, manifest: UnifiedManifest) -> dict:
    rec = manifest.record()
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    # parent chaining: default parent = hash of previous record
    if manifest.parent == "genesis":
        prev = load_manifests(p)
        if prev:
            rec["parent"] = prev[-1]["hash"]
            rec["hash"] = hashlib.sha256(
                _canonical({k: v for k, v in rec.items() if k != "hash"}).encode()
            ).hexdigest()[:16]
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def load_manifests(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out
