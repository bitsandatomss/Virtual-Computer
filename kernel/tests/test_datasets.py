"""Tests for datasets: determinism, zero-shot splits, regime labels."""
from virtual_kernel import capture_traces, dataset_hash, regime_histogram, to_transitions
from virtual_kernel.datasets import _reseed


def test_capture_deterministic():
    a = capture_traces(2, 8, base_seed=7000)
    b = capture_traces(2, 8, base_seed=7000)
    assert len(a) == len(b) == 2
    for ea, eb in zip(a, b):
        for sa, sb in zip(ea.steps, eb.steps):
            assert (sa.next_kir.scheduler.avg_latency_ms ==
                    sb.next_kir.scheduler.avg_latency_ms)


def test_holdout_seeds_disjoint():
    tr = _reseed([], [7000, 7001], 6, (2000, 6000), 2, 0.5)
    te = _reseed([], [9000, 9001], 6, (2000, 6000), 2, 0.5)
    assert {e.seed for e in tr}.isdisjoint({e.seed for e in te})
    dtr, dte = to_transitions(tr, 0.5), to_transitions(te, 0.5)
    assert len(dtr) == 12 and len(dte) == 12
    assert dataset_hash(dtr) != dataset_hash(dte)


def test_regimes_cover_bands():
    eps = capture_traces(6, 20, base_seed=7000)
    hist = regime_histogram(eps)
    assert set(hist) <= {"idle", "mid", "saturated"}
    assert sum(hist.values()) == 6 * 20
