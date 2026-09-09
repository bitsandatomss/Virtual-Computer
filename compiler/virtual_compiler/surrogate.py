"""Learned surrogate: ``(state, action) -> predicted outcome`` + uncertainty.

Design follows context.txt: the surrogate never replaces the oracle, it
*amortizes* it. Expensive gcc builds/runs are the truth oracle with a
finite budget ``B``; cheap surrogate calls are unlimited. Uncertainty
(distance to observed experience + outcome variance) drives active
learning: the environment asks "which real build should I run next?".

Model: distance-weighted kNN over normalized latent vectors. Deliberately
small, stdlib-only, JSON-serializable — a stronger model (stumps, GP,
GNN over CFG) plugs into the same ``predict/train/save/load`` seam,
mirroring how ``gcc-ai-synthesize`` plugs into the policy-lab evidence.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .state import FEATURE_KEYS, CompilationState, feature_vector


def _norm(rows: list[list[float]]) -> tuple[list[float], list[float]]:
    if not rows:
        return [0.0] * len(FEATURE_KEYS), [1.0] * len(FEATURE_KEYS)
    n = len(FEATURE_KEYS)
    means = [sum(r[i] for r in rows) / len(rows) for i in range(n)]
    scales = []
    for i in range(n):
        var = sum((r[i] - means[i]) ** 2 for r in rows) / len(rows)
        scales.append(math.sqrt(var) if var > 1e-12 else 1.0)
    return means, scales


@dataclass
class Prediction:
    ok_prob: float          # P(build succeeds)
    runtime_ms: float | None
    size_bytes: float | None
    uncertainty: float      # 0..1, 1 == pure guess
    neighbors: int


class Surrogate:
    """Experience-backed predictor over compilation latent states."""

    SCHEMA = "virtual-compiler.surrogate.v1"

    def __init__(self) -> None:
        self._rows: list[list[float]] = []
        self._ok: list[float] = []
        self._rt: list[float | None] = []
        self._sz: list[float | None] = []

    def __len__(self) -> int:
        return len(self._rows)

    def latent_rows(self) -> list[list[float]]:
        """Public read view of experience rows (for OOD calibration)."""
        return [list(r) for r in self._rows]

    # -- training ------------------------------------------------------
    def observe(self, state: CompilationState) -> None:
        """Record one oracle-validated state as experience."""
        if not state.validated:
            return
        self._rows.append(feature_vector(state.features))
        self._ok.append(1.0 if state.build_ok else 0.0)
        self._rt.append(state.runtime_ms)
        self._sz.append(float(state.binary_size) if state.binary_size else None)

    def train_from_corpus(self, path: str | Path) -> int:
        """Ingest ``gcc-ai.corpus.v1`` JSONL (source digests + outcomes).

        Corpus rows lack raw source, so they contribute only aggregate
        priors keyed on recorded feature aggregates when present.
        Returns number of rows ingested.
        """
        n = 0
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                feats = rec.get("feature_aggregates") or rec.get("features") or {}
                if not isinstance(feats, dict):
                    continue
                vec = [float(feats.get(k, 0.0)) for k in FEATURE_KEYS]
                outcome = rec.get("outcome") or {}
                self._rows.append(vec)
                self._ok.append(1.0 if outcome.get("build_ok", True) else 0.0)
                rt = outcome.get("runtime_ms", outcome.get("median_ms"))
                self._sz.append(outcome.get("binary_size"))
                self._rt.append(float(rt) if rt is not None else None)
                n += 1
        return n

    def train_from_policy_lab(self, path: str | Path) -> int:
        """Ingest ``gcc-ai.policy-lab.v1`` JSONL variant verdicts.

        Variant names encode the intervention (``no-<pass>`` policies,
        ``O*`` flag presets); runtimes/sizes become experience rows with
        synthetic feature deltas. Returns rows ingested.
        """
        import json as _json

        n = 0
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                if rec.get("event") not in ("variant", "verdict", None):
                    continue
                vec = [0.0] * len(FEATURE_KEYS)
                name = str(rec.get("variant") or rec.get("name") or "")
                if "O3" in name:
                    vec[FEATURE_KEYS.index("opt_level")] = 3.0
                elif "O0" in name:
                    vec[FEATURE_KEYS.index("opt_level")] = 0.0
                else:
                    vec[FEATURE_KEYS.index("opt_level")] = 2.0
                if "no-" in name:
                    vec[FEATURE_KEYS.index("policy_rules")] = 1.0
                self._rows.append(vec)
                verdict = str(rec.get("verdict", "improves"))
                self._ok.append(0.0 if "fail" in verdict else 1.0)
                rt = rec.get("median_ms", rec.get("runtime_ms"))
                self._rt.append(float(rt) if rt is not None else None)
                sz = rec.get("binary_size")
                self._sz.append(float(sz) if sz is not None else None)
                n += 1
        return n

    # -- prediction ----------------------------------------------------
    DATASET_SCHEMA = "virtual-compiler.dataset.v1"

    def train_from_dataset(self, path: str | Path) -> int:
        """Ingest a collected dataset (``cli collect`` output).

        Rows carry full feature dicts + outcomes, so unlike corpus rows
        they enter the experience store at full fidelity. Returns rows.
        """
        import json as _json

        n = 0
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                feats = rec.get("features")
                if not isinstance(feats, dict):
                    continue
                self._rows.append(
                    [float(feats.get(k, 0.0)) for k in FEATURE_KEYS])
                out = rec.get("outcome", {})
                self._ok.append(1.0 if out.get("build_ok", True) else 0.0)
                rt = out.get("runtime_ms")
                self._rt.append(float(rt) if rt is not None else None)
                sz = out.get("binary_size")
                self._sz.append(float(sz) if sz is not None else None)
                n += 1
        return n

    def predict(self, state: CompilationState, k: int = 5) -> Prediction:
        if not self._rows:
            return Prediction(ok_prob=0.5, runtime_ms=None,
                              size_bytes=None, uncertainty=1.0, neighbors=0)
        means, scales = _norm(self._rows)
        q = feature_vector(state.features)
        qn = [(q[i] - means[i]) / scales[i] for i in range(len(q))]
        dists: list[tuple[float, int]] = []
        for j, r in enumerate(self._rows):
            d = math.sqrt(sum(
                ((r[i] - means[i]) / scales[i] - qn[i]) ** 2
                for i in range(len(q))))
            dists.append((d, j))
        dists.sort()
        top = dists[:max(1, min(k, len(dists)))]
        weights = [1.0 / (1.0 + d) for d, _ in top]
        wsum = sum(weights)
        ok = sum(w * self._ok[j] for w, (_, j) in zip(weights, top)) / wsum
        rts = [(w, self._rt[j]) for w, (_, j) in zip(weights, top)
               if self._rt[j] is not None]
        szs = [(w, self._sz[j]) for w, (_, j) in zip(weights, top)
               if self._sz[j] is not None]
        rt = (sum(w * v for w, v in rts) / sum(w for w, _ in rts)) if rts else None
        sz = (sum(w * v for w, v in szs) / sum(w for w, _ in szs)) if szs else None
        nearest = top[0][0]
        spread = max(d for d, _ in top) - min(d for d, _ in top)
        unc = min(1.0, nearest / (nearest + 2.0) + min(0.4, spread / 8.0))
        if len(self._rows) < 3:
            unc = min(1.0, unc + 0.3)
        return Prediction(ok_prob=ok, runtime_ms=rt, size_bytes=sz,
                          uncertainty=unc, neighbors=len(top))

    def predict_relative(self, state: CompilationState,
                           baseline_ms: float | None,
                           k: int = 5) -> dict:
        """Normalized native target (CRITIQUE §6.4).

        The model emits program-relative quantities — fractional delta
        vs the program's own baseline — instead of absolute ms. Absolute
        time does not transfer across programs; relative effects might.
        """
        pred = self.predict(state, k=k)
        if baseline_ms is None or baseline_ms <= 0:
            return {"delta": None, "uncertainty": pred.uncertainty,
                    "basis": "no-baseline"}
        if pred.runtime_ms is None:
            return {"delta": None, "uncertainty": pred.uncertainty,
                    "basis": "no-estimate"}
        return {"delta": (baseline_ms - pred.runtime_ms) / baseline_ms,
                "uncertainty": pred.uncertainty, "basis": "surrogate"}

    # -- persistence ---------------------------------------------------
    def to_dict(self) -> dict:
        return {"schema": self.SCHEMA, "rows": self._rows,
                "ok": self._ok, "rt": self._rt, "sz": self._sz}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Surrogate":
        rec = json.loads(Path(path).read_text(encoding="utf-8"))
        s = cls()
        s._rows = rec.get("rows", [])
        s._ok = rec.get("ok", [])
        s._rt = rec.get("rt", [])
        s._sz = rec.get("sz", [])
        return s
