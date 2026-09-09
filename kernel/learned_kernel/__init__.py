"""
Learned Kernel — introspectable, verifiable, learnable OS policy objects.

Public surface:
    schemas      KIR + action models
    PolicyBase   policy ABI (+ heuristic baselines)
    LearnedLinearPolicy, OfflineTrainer
    KernelSimulator
    PolicyValidator
    LearnedKernelRuntime, KernelActuator
    WorkloadDriftManager
    ProvenanceLogger
"""
from .policy.schemas import (
    CPUMetrics,
    KernelIntermediateRepresentation,
    MemMetrics,
    NetMetrics,
    NetAction,
    PolicyAction,
    SchedulerAction,
    SchedulerState,
    VMAction,
)
from .policy.core import HeuristicLatencyPolicy, LinuxDefaultPolicy, PolicyBase
from .policy.joint_policy import JointHeuristicPolicy
from .trainer.rl_trainer import LearnedLinearPolicy, OfflineTrainer, TrainingReport
from .simulator.env import KernelSimulator, WorkloadProfile
from .validator.core import PolicyValidator, ValidatorError
from .runtime.loop import KernelActuator, LearnedKernelRuntime, StepResult
from .runtime.adaptation import WorkloadDriftManager
from .telemetry.provenance import ProvenanceLogger

__version__ = "3.0.0"

__all__ = [
    "CPUMetrics", "KernelIntermediateRepresentation", "MemMetrics", "NetMetrics",
    "NetAction", "PolicyAction", "SchedulerAction", "SchedulerState", "VMAction",
    "PolicyBase", "LinuxDefaultPolicy", "HeuristicLatencyPolicy",
    "JointHeuristicPolicy", "LearnedLinearPolicy", "OfflineTrainer",
    "TrainingReport", "KernelSimulator", "WorkloadProfile",
    "PolicyValidator", "ValidatorError", "KernelActuator",
    "LearnedKernelRuntime", "StepResult", "WorkloadDriftManager",
    "ProvenanceLogger",
]
