"""Out-of-distribution regime detection (VISION N3.3).

The surrogate is "spectacularly good inside its training distribution and
catastrophically wrong outside it" (:561-577) — the explosion scenario.
Distance-to-experience is therefore not a ranking nicety but the safety
instrument. This module calibrates the instrument from data: the
abstention threshold is the 95th percentile of leave-one-out
nearest-neighbor distances in experience, not a constant. Thresholds are
task-dependent (N5): *magnitude* claims abstain sooner than *ranking*
claims, because exact values break before orderings do.
"""
from __future__ import annotations

import math

TASK_RULES = {
    # task -> multiplier on the calibrated threshold before abstaining
    "ranking": 2.0,
    "magnitude": 1.0,
    "decision": 1.5,
}


def _dist(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def fit(rows: list[list[float]]) -> dict:
    """Calibrate an OOD model on experience rows.

    Returns ``{"threshold": float|None, "n": int}``. With <2 rows there
    is no geometry to calibrate — threshold None means "always abstain"
    (honest cold start, not a permissive default).
    """
    if len(rows) < 2:
        return {"threshold": None, "n": len(rows)}
    nn: list[float] = []
    for i, r in enumerate(rows):
        best = min(_dist(r, q) for j, q in enumerate(rows) if j != i)
        nn.append(best)
    nn.sort()
    idx = min(len(nn) - 1, int(0.95 * len(nn)))
    return {"threshold": nn[idx] if nn[idx] > 0 else 1e-9, "n": len(rows)}


def assess(vec: list[float], model: dict) -> dict:
    """Regime assessment of one latent vector against fitted experience."""
    from .features import FEATURE_KEYS  # noqa: F401  (schema anchor)
    threshold = model.get("threshold")
    if threshold is None:
        return {"distance": None, "threshold": None, "regime": "unknown",
                "abstain": {t: True for t in TASK_RULES},
                "reason": "no calibrated experience"}
    # distance recomputed by caller context; here vec IS the query and the
    # model carries reference rows when fitted via fit_on_surrogate
    refs = model.get("rows", [])
    dist = min((_dist(vec, r) for r in refs), default=float("inf"))
    if dist <= threshold:
        regime = "familiar"
    elif dist <= 2.0 * threshold:
        regime = "unfamiliar"
    else:
        regime = "novel"
    return {"distance": dist, "threshold": threshold, "regime": regime,
            "abstain": {t: dist > mult * threshold
                        for t, mult in TASK_RULES.items()},
            "reason": f"{regime} regime at distance {dist:.3f}"}


def fit_on_surrogate(surrogate) -> dict:
    """Calibrate on a surrogate's experience store (public rows only)."""
    rows = [list(r) for r in surrogate.latent_rows()]
    model = fit(rows)
    model["rows"] = rows
    return model
