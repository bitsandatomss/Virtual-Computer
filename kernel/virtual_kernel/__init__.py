"""Virtual Kernel — executable, branchable surrogate of OS kernel dynamics.

Research-grade implementation of the learned-surrogate vision in context.txt
(Virtual Cell / Virtual Chess / Vesuvius + taxonomy), instantiated for the
object the taxonomy calls Virtual Computer / Virtual Algorithm:

    computation / algorithmic process  ->  learned kernel state-transition
                                            environment

Common operation across all of them: observe -> learn -> represent ->
branch -> intervene -> validate.

The deterministic PolicyValidator (learned_kernel) remains the sole safety
authority; the surrogate proposes and explores, never actuates directly.
"""
from .config import LATENCY_ARMS, VKConfig, fast_config
from .datasets import (
    Episode,
    capture_traces,
    demand_band,
    dataset_hash,
    regime_histogram,
    regime_of,
    to_transitions,
    train_holdout_split,
)
from .dynamics import (
    ACTION_DIM,
    LATENT_DIM,
    OBS_DIM,
    DynamicsDataset,
    TransitionModel,
    collect_dataset,
    decode_latent,
    encode_action,
    encode_kir,
    kir_observables,
    train_model,
)
from .ensemble import EnsembleDynamics
from .environment import BranchState, VirtualKernel
from .metrics import (
    calibration,
    divergence_stats,
    hallucination_rate,
    mae,
    mse,
    propagation_law,
    spearman_rank,
    strength_per_oracle,
    summarize_oracle_report,
    topk_hit_rate,
)
from .oracle import LadderReport, LadderRung, OracleReport, TruthOracle, run_ladder
from .planning import MCTSResult, mcts_search, robust_search, rollout_sequence_reward
from .search import BeamResult, validate_topk, virtual_screen
from .active import ActiveLoopResult, ActiveProbe, active_loop, rank_probes
from .agents import AgentFinding, LabReport, VirtualLab
from .conservation import check_transition, conservation_report, score_trajectory
from .fidelity import FidelitySpec, STANDARD_SPECS, check as fidelity_check, fidelity_report
from .levels import LEVEL_NAMES, assess_L1, assess_L2, assess_L3, assess_L4, assess_L5, certify
from .ood import OODDetector, regime_coverage
from .provenance import CODE_VERSION, ManifestLog, RunManifest, ensemble_hash, write_model_card

__version__ = "0.1.0"

__all__ = [
    "VKConfig", "fast_config", "LATENCY_ARMS",
    "Episode", "capture_traces", "demand_band", "dataset_hash",
    "regime_histogram", "regime_of", "to_transitions", "train_holdout_split",
    "TransitionModel", "DynamicsDataset", "collect_dataset", "train_model",
    "encode_kir", "encode_action", "kir_observables", "decode_latent",
    "LATENT_DIM", "OBS_DIM", "ACTION_DIM",
    "EnsembleDynamics",
    "VirtualKernel", "BranchState",
    "mse", "mae", "divergence_stats", "hallucination_rate", "spearman_rank",
    "topk_hit_rate", "calibration", "strength_per_oracle", "summarize_oracle_report",
    "propagation_law",
    "TruthOracle", "OracleReport", "LadderRung", "LadderReport", "run_ladder",
    "MCTSResult", "mcts_search", "robust_search", "rollout_sequence_reward",
    "BeamResult", "virtual_screen", "validate_topk",
    "ActiveProbe", "ActiveLoopResult", "rank_probes", "active_loop",
    "AgentFinding", "LabReport", "VirtualLab",
    "check_transition", "score_trajectory", "conservation_report",
    "FidelitySpec", "STANDARD_SPECS", "fidelity_check", "fidelity_report",
    "LEVEL_NAMES", "assess_L1", "assess_L2", "assess_L3", "assess_L4",
    "assess_L5", "certify",
    "OODDetector", "regime_coverage",
    "RunManifest", "ManifestLog", "ensemble_hash", "write_model_card", "CODE_VERSION",
]
