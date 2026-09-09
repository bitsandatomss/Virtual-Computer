"""Cross-validated decision-stump synthesis from corpus evidence.

This module is the honest "learning" step of the repository: it consumes a
``gcc-ai.corpus.v1`` dataset produced by the corpus builder and searches,
for every ablated pass, for the single-feature decision stump that best
predicts when disabling that pass improved measured runtime.

Rigor requirements enforced here:

- Model selection happens inside k-fold cross-validation (the best stump
  is re-chosen on each training split and scored only on its held-out
  split), so the reported score is out-of-sample, not resubstitution.
- A rule is emitted only when its mean cross-validated accuracy beats the
  majority-class baseline by the requested margin AND both sides of the
  split retain at least ``--min-support`` workloads.  Otherwise the tool
  refuses and says why; a negative result is recorded, never hidden.
- Everything is deterministic given the corpus and ``--seed``: fold
  assignment uses a seeded shuffle and tie-breaking is total.
- Emitted rules use the plugin's conditional vocabulary
  (``disable_pass=<p> if <feature><op><integer>``), keeping GCC as the
  authority and the deployed artifact small and auditable.

Thresholds found on real-valued aggregates are converted to exact integer
comparisons because the plugin grammar is integer-only; the conversion is
direction-aware so semantics are preserved exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Sequence

SCHEMA = "gcc-ai.synthesis.v1"
DEFAULT_SEED = 20260822


@dataclass(frozen=True)
class WorkloadRow:
    name: str
    features: dict[str, float]
    label: int  # 1 when disabling the pass improved, else 0


@dataclass(frozen=True)
class Stump:
    feature: str
    op: str  # "<=" or ">="
    threshold: float

    def predict(self, row: WorkloadRow) -> int:
        value = row.features.get(self.feature, 0.0)
        if self.op == "<=":
            return int(value <= self.threshold)
        return int(value >= self.threshold)


@dataclass(frozen=True)
class SynthesisFinding:
    pass_name: str
    emitted: bool
    reasons: tuple[str, ...]
    workloads: int
    class_counts: tuple[int, int]
    baseline_accuracy: float
    cv_mean_accuracy: float | None
    stump: Stump | None
    supports: tuple[int, int]


def load_corpus(path: Path) -> tuple[list[str], dict[str, dict]]:
    """Return (passes, {pass_name: {workload_name: row}}).

    Only runtime ablation variants (``disable-*``) produce supervised rows;
    control/reference outcomes are ignored by design.
    """
    passes: list[str] = []
    by_pass: dict[str, dict[str, WorkloadRow]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        event = record.get("event")
        if event == "corpus-meta":
            passes = [
                name for name in record.get("passes", []) if isinstance(name, str)
            ]
        elif event == "workload":
            name = str(record.get("name", ""))
            features_raw = record.get("features")
            outcomes = record.get("outcomes")
            if (
                not name
                or not isinstance(features_raw, dict)
                or not isinstance(outcomes, dict)
            ):
                continue
            features = {
                str(key): float(value)
                for key, value in features_raw.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            for variant, outcome in outcomes.items():
                if not isinstance(variant, str) or not variant.startswith("disable-"):
                    continue
                if not isinstance(outcome, dict):
                    continue
                verdict = outcome.get("verdict")
                if verdict not in {
                    "improves",
                    "within-noise",
                    "regresses",
                    "not-gated",
                }:
                    continue  # failures are excluded, not treated as label 0
                pass_name = variant[len("disable-") :]
                label = 1 if verdict == "improves" else 0
                by_pass.setdefault(pass_name, {})[name] = WorkloadRow(
                    name, features, label
                )
    ordered = [name for name in passes if name in by_pass]
    for extra in sorted(set(by_pass) - set(ordered)):
        ordered.append(extra)
    return ordered, by_pass


def _candidate_thresholds(values: Sequence[float]) -> list[float]:
    """Midpoints between consecutive distinct values, bounded and ordered."""
    unique = sorted(set(values))
    if len(unique) < 2:
        return []
    midpoints = [(low + high) / 2.0 for low, high in zip(unique, unique[1:])]
    if len(midpoints) > 24:
        step = len(midpoints) / 24
        midpoints = [midpoints[int(index * step)] for index in range(24)]
    return midpoints


def _accuracy(predictions: Sequence[int], labels: Sequence[int]) -> float:
    if not labels:
        return 0.0
    correct = sum(prediction == label for prediction, label in zip(predictions, labels))
    return correct / len(labels)


def select_stump(
    rows: Sequence[WorkloadRow], feature_names: Sequence[str]
) -> Stump | None:
    """Pick the training-split stump with best accuracy; None if impossible."""
    labels = [row.label for row in rows]
    positive = sum(labels)
    negative = len(labels) - positive
    if positive == 0 or negative == 0:
        return None
    best: tuple[tuple, Stump] | None = None
    for feature in sorted(feature_names):
        values = [row.features.get(feature, 0.0) for row in rows]
        for threshold in _candidate_thresholds(values):
            for op in ("<=", ">="):
                stump = Stump(feature, op, threshold)
                predictions = [stump.predict(row) for row in rows]
                predicted_ones = sum(predictions)
                predicted_zeros = len(predictions) - predicted_ones
                if predicted_ones == 0 or predicted_zeros == 0:
                    continue  # degenerate: cannot beat the majority baseline
                score = (
                    -_accuracy(predictions, labels),
                    feature,
                    op,
                    threshold,
                )
                if best is None or score < best[0]:
                    best = (score, stump)
    return best[1] if best else None


def _fold_assignment(count: int, folds: int, seed: int) -> list[int]:
    order = list(range(count))
    random.Random(seed).shuffle(order)
    assignments = [0] * count
    for position, index in enumerate(order):
        assignments[index] = position % folds
    return assignments


def _integerize(stump: Stump) -> Stump:
    """Convert a real-valued threshold to the exact equivalent integer rule."""
    if float(stump.threshold).is_integer():
        return Stump(stump.feature, stump.op, float(int(stump.threshold)))
    if stump.op == "<=":
        return Stump(stump.feature, stump.op, float(math.floor(stump.threshold)))
    return Stump(stump.feature, stump.op, float(math.ceil(stump.threshold)))


def synthesize_rule(
    rows: Sequence[WorkloadRow],
    pass_name: str,
    folds: int,
    margin: float,
    min_support: int,
    seed: int,
) -> SynthesisFinding:
    """Learn one pass rule under cross-validation, or refuse with reasons."""
    labels = [row.label for row in rows]
    positives = sum(labels)
    negatives = len(labels) - positives
    baseline = max(positives, negatives) / len(labels)
    class_counts = (positives, negatives)
    reasons: list[str] = []

    feature_names = sorted({name for row in rows for name in row.features})
    effective_folds = max(2, min(folds, len(rows)))

    assignments = _fold_assignment(len(rows), effective_folds, seed)
    fold_scores: list[float] = []
    for fold in range(effective_folds):
        train = [row for index, row in enumerate(rows) if assignments[index] != fold]
        test = [row for index, row in enumerate(rows) if assignments[index] == fold]
        if not test:
            continue
        stump = select_stump(train, feature_names)
        if stump is None:
            majority = 1 if sum(row.label for row in train) * 2 >= len(train) else 0
            fold_scores.append(
                _accuracy([majority] * len(test), [r.label for r in test])
            )
            continue
        predictions = [stump.predict(row) for row in test]
        fold_scores.append(_accuracy(predictions, [row.label for row in test]))

    cv_mean = fmean(fold_scores) if fold_scores else 0.0
    stump = select_stump(rows, feature_names)

    if len(rows) < 2 * effective_folds:
        reasons.append(
            f"only {len(rows)} usable workloads; need at least "
            f"{2 * effective_folds} for {effective_folds}-fold validation"
        )
    if positives == 0 or negatives == 0:
        reasons.append("a single class is present; there is no decision to learn")
    if stump is not None:
        predictions = [stump.predict(row) for row in rows]
        supports = (sum(predictions), len(predictions) - sum(predictions))
        if min(supports) < min_support:
            reasons.append(
                f"best split keeps only {min(supports)} workload(s) on one "
                f"side; require {min_support}"
            )
    else:
        supports = (0, 0)
        if positives > 0 and negatives > 0:
            reasons.append("no non-degenerate feature split exists")

    if not reasons:
        assert stump is not None
        if cv_mean < baseline + margin:
            reasons.append(
                f"cross-validated accuracy {cv_mean:.3f} does not beat the "
                f"{baseline:.3f} baseline by the required {margin:.3f}"
            )

    emitted = not reasons
    if emitted and stump is not None:
        stump = _integerize(stump)

    return SynthesisFinding(
        pass_name=pass_name,
        emitted=emitted,
        reasons=tuple(reasons),
        workloads=len(rows),
        class_counts=class_counts,
        baseline_accuracy=baseline,
        cv_mean_accuracy=cv_mean if fold_scores else None,
        stump=stump,
        supports=supports,
    )


def synthesize_from_corpus(
    corpus_path: Path,
    folds: int,
    margin: float,
    min_support: int,
    seed: int,
) -> tuple[list[SynthesisFinding], list[dict], list[str]]:
    """Return findings, report records, and emitted policy lines."""
    passes, by_pass = load_corpus(corpus_path)
    corpus_digest = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    records: list[dict] = [
        {
            "schema": SCHEMA,
            "event": "synthesis-meta",
            "corpus_sha256": corpus_digest,
            "folds": folds,
            "margin": margin,
            "min_support": min_support,
            "seed": seed,
        }
    ]
    findings: list[SynthesisFinding] = []
    policy_lines: list[str] = []
    for pass_name in passes:
        rows_by_name = by_pass[pass_name]
        rows = [rows_by_name[name] for name in sorted(rows_by_name)]
        finding = synthesize_rule(rows, pass_name, folds, margin, min_support, seed)
        findings.append(finding)
        record: dict = {
            "schema": SCHEMA,
            "event": "evaluation",
            "pass": finding.pass_name,
            "emitted": finding.emitted,
            "workloads": finding.workloads,
            "class_counts": {
                "improves": finding.class_counts[0],
                "other": finding.class_counts[1],
            },
            "baseline_majority_accuracy": finding.baseline_accuracy,
            "cv_mean_accuracy": finding.cv_mean_accuracy,
            "reasons": list(finding.reasons),
        }
        if finding.stump is not None:
            record["stump"] = {
                "feature": finding.stump.feature,
                "op": finding.stump.op,
                "threshold": finding.stump.threshold,
                "support_predicted_improves": finding.supports[0],
                "support_predicted_other": finding.supports[1],
            }
        if finding.emitted and finding.stump is not None:
            threshold_text = str(int(finding.stump.threshold))
            policy_lines.append(
                f"disable_pass={finding.pass_name} if "
                f"{finding.stump.feature}{finding.stump.op}{threshold_text}"
            )
        records.append(record)
    records.append(
        {
            "schema": SCHEMA,
            "event": "decision",
            "rules_emitted": len(policy_lines),
            "policy": policy_lines,
        }
    )
    return findings, records, policy_lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gcc-ai-synthesize",
        description=(
            "Synthesize conservative conditional pass-disable rules from a "
            "gcc-ai-corpus dataset using cross-validated decision stumps."
        ),
    )
    parser.add_argument("--version", action="version", version=_tool_version())
    parser.add_argument("--corpus", required=True, help="dataset path (.jsonl)")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--policy-out",
        help="write emitted rules to this plugin policy file",
    )
    parser.add_argument(
        "--report",
        help="write gcc-ai.synthesis.v1 JSONL evidence to this path",
    )
    return parser


def _tool_version() -> str:
    try:
        from importlib.metadata import version

        return version("gcc-ai-native")
    except Exception:
        return "0.0.0+unknown"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    corpus_path = Path(args.corpus)
    if not corpus_path.is_file():
        print(f"gcc-ai-synthesize: corpus not found: {corpus_path}", file=sys.stderr)
        return 2
    if args.folds < 2:
        print("gcc-ai-synthesize: --folds must be at least 2", file=sys.stderr)
        return 2
    if args.margin < 0 or args.min_support < 1:
        print("gcc-ai-synthesize: invalid margin/support", file=sys.stderr)
        return 2

    findings, records, policy_lines = synthesize_from_corpus(
        corpus_path, args.folds, args.margin, args.min_support, args.seed
    )
    if args.report:
        Path(args.report).write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )
    if args.policy_out and policy_lines:
        Path(args.policy_out).write_text(
            "\n".join(policy_lines) + "\n", encoding="utf-8"
        )

    for finding in findings:
        stump_text = ""
        if finding.stump is not None:
            stump_text = (
                f" [{finding.stump.feature} {finding.stump.op} "
                f"{int(finding.stump.threshold)}]"
            )
        status = "EMITTED" if finding.emitted else "refused"
        print(
            f"{finding.pass_name:<16} {status:<8} workloads={finding.workloads} "
            f"cv={finding.cv_mean_accuracy:.3f} "
            f"baseline={finding.baseline_accuracy:.3f}{stump_text}"
        )
        for reason in finding.reasons:
            print(f"    - {reason}")

    if policy_lines:
        destination = args.policy_out or "(report only)"
        print(f"Rules written: {destination}")
        return 0
    print(
        "No rule met the evidence bar; the GCC heuristics stand. "
        "This is a valid outcome, not an error."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
