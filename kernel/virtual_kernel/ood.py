"""ood.py — trust requirement #3: out-of-distribution detection.

context.txt (lines 592-594): "Can it recognize that you've entered an
unfamiliar regime?" — the virtual-lab failure mode is not inaccuracy but
CONFIDENT error outside the training distribution (the explosion example,
lines 567-577).

Design (deliberately simple, dependency-free, auditable):
- Fit on training transition features: per-dimension mean/std + centroid.
- novelty(x) = max(per-dim |z-score|, euclidean distance to centroid in
  standardized space). Large = far from anything seen during training.
- Threshold = 99th percentile of training novelty (1% train flag rate by
  construction). Separation between train and test novelty distributions
  measures whether the detector can tell regimes apart at all.
- regime coverage helper: which observable demand bands were seen in
  training vs requested at test time (links to datasets.demand_band).

This is a detector, not a fix: downstream consumers (MCTS penalty,
adversarial veto, L4 gate) must refuse or down-weight decisions in flagged
regions. Detection without a policy is decoration.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


class OODDetector:
    def __init__(self, quantile: float = 0.99):
        self.quantile = quantile
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.threshold_: float = float("inf")

    def fit(self, train_feats: np.ndarray) -> "OODDetector":
        X = np.asarray(train_feats, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ < 1e-9] = 1e-9
        self.threshold_ = float(np.quantile(self.novelty(X), self.quantile))
        return self

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None, "fit() first"
        return (np.asarray(X, dtype=np.float64) - self.mean_) / self.std_

    def novelty(self, X) -> np.ndarray:
        """Novelty score per row; higher = further from training manifold."""
        Z = self._standardize(np.atleast_2d(np.asarray(X, dtype=np.float64)))
        per_dim = np.abs(Z).max(axis=1)
        radial = np.sqrt((Z ** 2).sum(axis=1))
        return np.maximum(per_dim, radial / np.sqrt(Z.shape[1]))

    def flagged(self, X) -> np.ndarray:
        return self.novelty(X) > self.threshold_

    def report(self, train_feats: np.ndarray,
               test_feats: np.ndarray) -> Dict:
        """Separation statistics between known and novel regimes."""
        n_tr = self.novelty(train_feats)
        n_te = self.novelty(test_feats)
        # P(novelty_test > novelty_train): 0.5 = indistinguishable, 1.0 = clean split
        sep = float((n_te[:, None] > n_tr[None, :]).mean())
        return {"threshold": self.threshold_,
                "train_flag_rate": float((n_tr > self.threshold_).mean()),
                "test_flag_rate": float((n_te > self.threshold_).mean()),
                "train_novelty_median": float(np.median(n_tr)),
                "test_novelty_median": float(np.median(n_te)),
                "separation": sep,
                "n_train": len(n_tr), "n_test": len(n_te)}


def regime_coverage(train_regimes: Sequence[str],
                    test_regimes: Sequence[str]) -> Dict:
    """Which observable demand bands at test time were never seen in training."""
    tr, te = set(train_regimes), set(test_regimes)
    return {"train_bands": sorted(tr), "test_bands": sorted(te),
            "unseen_bands": sorted(te - tr),
            "coverage": len(te & tr) / max(len(te), 1)}
