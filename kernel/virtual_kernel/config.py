"""config.py — reproducible experiment configuration for the Virtual Kernel.

Every benchmark, experiment and training run resolves a VKConfig first; the
config hash is recorded in the run manifest (provenance.py) so any reported
number can be traced back to the exact settings that produced it.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import List, Tuple

LATENCY_ARMS: Tuple[int, ...] = (1000, 2000, 4000, 6000, 12000, 24000)


@dataclass
class VKConfig:
    """Single source of truth for one Virtual Kernel experiment."""

    name: str = "vk-default"
    seed: int = 7000
    # surrogate
    n_members: int = 5
    n_episodes: int = 24
    steps_per_episode: int = 40
    epochs: int = 300
    lr: float = 0.02
    n_cpus: int = 2
    dt_s: float = 0.5
    # planning-relevant training: unrolled multi-step loss weight/horizon.
    # Default 0.0 (single-step only): mixed training was implemented,
    # gradient-verified, and tested as a fix for counterfactual ranking, but
    # measured NO improvement on the L3 gate (see THEORY.md "negative
    # result"). The infrastructure stays for future work; the default stays
    # with what the gates validate.
    unroll_weight: float = 0.0
    unroll_horizon: int = 3
    # planning
    arms: Tuple[int, ...] = LATENCY_ARMS
    horizon: int = 4
    beam_width: int = 6
    mcts_sims: int = 200
    mcts_c: float = 1.4
    uncertainty_penalty: float = 0.5
    # oracle budget (the killer-benchmark constraint)
    oracle_budget: int = 20
    divergence_threshold: float = 0.15
    # zero-shot splits: workload seeds seen vs held out
    train_seeds: List[int] = field(default_factory=lambda: list(range(7000, 7024)))
    test_seeds: List[int] = field(default_factory=lambda: list(range(9000, 9012)))
    report_dir: str = "reports"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["arms"] = list(self.arms)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "VKConfig":
        d = dict(d)
        if "arms" in d:
            d["arms"] = tuple(d["arms"])
        return cls(**d)

    @property
    def hash(self) -> str:
        canon = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canon.encode()).hexdigest()[:16]


def fast_config(name: str = "vk-fast", **overrides) -> VKConfig:
    """Tiny config for tests/smoke runs. Never use for reported numbers."""
    base = VKConfig(name=name, n_members=2, n_episodes=4, steps_per_episode=10,
                    epochs=25, horizon=2, beam_width=3, mcts_sims=30,
                    oracle_budget=6,
                    train_seeds=list(range(7000, 7004)),
                    test_seeds=list(range(9000, 9002)))
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


PRESET_DEFAULT = VKConfig()
