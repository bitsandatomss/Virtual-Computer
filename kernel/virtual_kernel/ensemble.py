"""ensemble.py — uncertainty via deep-ensemble disagreement.

context.txt differentiator: "Uncertainty — know where the surrogate is
unreliable" (line 269) + "surrogate critic — identifies where the learned
model may be wrong" (line 578).

Each member trains from a different seed (different init + data shuffle).
Prediction mean = virtual state; variance = uncertainty.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np

from learned_kernel.policy.schemas import KernelIntermediateRepresentation, PolicyAction

from .dynamics import DynamicsDataset, TransitionModel, collect_dataset


class EnsembleDynamics:
    def __init__(self, members: List[TransitionModel]):
        assert members, "ensemble needs >=1 member"
        self.members = members
        self.dataset: DynamicsDataset | None = None

    @classmethod
    def train(cls, n_members: int = 5, n_episodes: int = 24,
              steps_per_episode: int = 40, base_seed: int = 7000,
              epochs: int = 300, lr: float = 0.02, n_cpus: int = 2,
              verbose: bool = False,
              dataset: DynamicsDataset | None = None,
              episodes=None, unroll_weight: float = 0.0,
              horizon: int = 3) -> "EnsembleDynamics":
        ds: DynamicsDataset = dataset if dataset is not None else collect_dataset(
            n_episodes=n_episodes, steps_per_episode=steps_per_episode,
            n_cpus=n_cpus, base_seed=base_seed)
        windows = None
        if episodes is not None and unroll_weight:
            from .datasets import windows_from_episodes
            windows = windows_from_episodes(episodes, horizon)
        members = []
        for i in range(n_members):
            m = TransitionModel(seed=1000 + i)
            m.fit(ds.feats, ds.acts, ds.targets,
                  epochs=epochs, lr=lr, verbose=verbose and i == 0,
                  windows=windows, unroll_weight=unroll_weight)
            members.append(m)
        obj = cls(members)
        obj.dataset = ds
        return obj

    @classmethod
    def train_config(cls, config, dataset: DynamicsDataset | None = None,
                     verbose: bool = False, episodes=None) -> "EnsembleDynamics":
        """Train from a VKConfig (provenance-friendly entry point)."""
        return cls.train(n_members=config.n_members,
                         n_episodes=config.n_episodes,
                         steps_per_episode=config.steps_per_episode,
                         base_seed=config.seed,
                         epochs=config.epochs, lr=config.lr,
                         n_cpus=config.n_cpus, verbose=verbose,
                         dataset=dataset, episodes=episodes,
                         unroll_weight=getattr(config, "unroll_weight", 0.0),
                         horizon=getattr(config, "unroll_horizon", 3))

    def __len__(self) -> int:
        return len(self.members)

    # -- persistence -- #
    def save(self, path: str) -> str:
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"format": "vk-ensemble-v1",
                       "members": [m.to_dict() for m in self.members]}, f)
        return path

    @classmethod
    def load(cls, path: str) -> "EnsembleDynamics":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls([TransitionModel.from_dict(m) for m in d["members"]])

    def hashes(self) -> List[str]:
        return [m.param_hash() for m in self.members]

    def evaluate(self, dataset: DynamicsDataset) -> Dict:
        """Mean member single-step error + disagreement on a transition set."""
        errs, disag = [], []
        for i in range(len(dataset)):
            f, a, y = dataset.feats[i], np.array([dataset.acts[i]]), dataset.targets[i]
            outs = np.array([m.decode(m.transition(m.encode(f), a), a)
                             for m in self.members])
            errs.append(float(np.mean((outs.mean(axis=0) - y) ** 2)))
            disag.append(float(outs.std(axis=0).mean()))
        return {"mse": float(np.mean(errs)), "mean_disagreement": float(np.mean(disag)),
                "n": len(dataset)}

    def calibration(self, dataset: DynamicsDataset,
                    sample: int = 200) -> Dict:
        """Uncertainty-error alignment on a transition set (metrics.calibration)."""
        from .metrics import calibration as _cal
        from .dynamics import encode_kir  # noqa: F401
        idx = np.random.default_rng(0).choice(len(dataset),
                                              size=min(sample, len(dataset)),
                                              replace=False)
        u, e = [], []
        for i in idx:
            f, a, y = dataset.feats[i], np.array([dataset.acts[i]]), dataset.targets[i]
            outs = np.array([m.decode(m.transition(m.encode(f), a), a)
                             for m in self.members])
            u.append(float(outs.std(axis=0).mean()))
            e.append(float(np.mean((outs.mean(axis=0) - y) ** 2)))
        return _cal(u, e)

    def predict_all_obs(self, kir: KernelIntermediateRepresentation,
                        action: Optional[PolicyAction]) -> np.ndarray:
        return np.array([m.predict_obs(kir, action) for m in self.members])

    def predict_mean_obs(self, kir: KernelIntermediateRepresentation,
                         action: Optional[PolicyAction]) -> np.ndarray:
        return self.predict_all_obs(kir, action).mean(axis=0)

    def uncertainty(self, kir: KernelIntermediateRepresentation,
                    action: Optional[PolicyAction]) -> float:
        """Mean per-output std across members — scalar unreliability score."""
        all_o = self.predict_all_obs(kir, action)
        return float(all_o.std(axis=0).mean())

    def predict_kir(self, kir: KernelIntermediateRepresentation,
                    action: Optional[PolicyAction], member: str = "mean",
                    dt_s: float = 0.5) -> KernelIntermediateRepresentation:
        if member == "mean":
            # average latent-space decode: use member closest to mean obs
            all_o = self.predict_all_obs(kir, action)
            mean = all_o.mean(axis=0)
            best = int(np.argmin(((all_o - mean) ** 2).sum(axis=1)))
            return self.members[best].predict_kir(kir, action, dt_s=dt_s,
                                                  n_cpus=len(kir.scheduler.cpus))
        return self.members[int(member)].predict_kir(
            kir, action, dt_s=dt_s, n_cpus=len(kir.scheduler.cpus))
