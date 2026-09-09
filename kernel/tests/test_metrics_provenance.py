"""Tests for metrics + provenance: scoring semantics and audit chain."""
import numpy as np

from virtual_kernel import (
    ManifestLog,
    RunManifest,
    calibration,
    divergence_stats,
    hallucination_rate,
    spearman_rank,
    strength_per_oracle,
)


def test_divergence_stats_finds_step():
    s = divergence_stats([0.01, 0.05, 0.2, 0.3], 0.15)
    assert s["divergence_step"] == 2
    assert s["final"] == 0.3
    assert s["horizon"] == 4


def test_spearman_ranking():
    assert spearman_rank([1, 2, 3], [10, 20, 30]) == 1.0
    assert spearman_rank([1, 2, 3], [30, 20, 10]) == -1.0


def test_hallucination_rate():
    curves = [[0.01, 0.02], [0.5, 0.6], [0.02, 0.2]]
    assert hallucination_rate(curves, 0.15) == 2 / 3


def test_calibration_monotonic_synthetic():
    u = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
    e = [0.001, 0.002, 0.01, 0.05, 0.1, 0.2]
    c = calibration(u, e)
    assert c["monotonic"] is True
    assert c["uncertainty_error_corr"] > 0.9


def test_strength_per_oracle():
    s = strength_per_oracle(candidate_oracle_reward=-1.0,
                            best_oracle_reward=-0.5, oracle_calls=4)
    assert s["calls"] == 4
    assert s["regret_frac"] < 0  # candidate worse than best -> negative gap convention


def test_manifest_chain_roundtrip(tmp_path):
    log = ManifestLog(str(tmp_path / "m.jsonl"))
    log.append(RunManifest(name="a", config_hash="c1", dataset_hash="d1",
                           model_hashes=["m1"], results={"x": 1}))
    log.append(RunManifest(name="b", config_hash="c2", dataset_hash="d2",
                           model_hashes=["m2"], results={"x": 2}))
    ok, seq, reason = ManifestLog.verify(str(tmp_path / "m.jsonl"))
    assert ok, reason
