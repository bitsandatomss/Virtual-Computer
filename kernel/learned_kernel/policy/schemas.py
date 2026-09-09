"""
policy/schemas.py — KIR and Action ABI.

Two deliberately different constraint philosophies:

* Observation schemas (KIR) are STRICTLY range-checked at ingestion time.
  Bad telemetry must be rejected at the boundary before it can poison
  features, rewards, or drift statistics (data-quality defence).

* Action schemas are intentionally PERMISSIVE (type-correctness only).
  Numeric/semantic bounds belong exclusively to the deterministic
  Validator, which owns safety authority. If schemas clamped actions,
  the validator could never observe or reject violations, destroying
  the layered-defence property that tests rely on.
"""
from __future__ import annotations

from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────── #
# Observation (KIR) — strict ingestion-time constraints            #
# ──────────────────────────────────────────────────────────────── #

class CPUMetrics(BaseModel):
    utilization: float = Field(..., ge=0.0, le=1.0,
                               description="CPU utilization fraction [0, 1]")
    runnable_tasks: int = Field(..., ge=0,
                                description="Tasks on this CPU's runqueue")
    irq_time: float = Field(0.0, ge=0.0, le=1.0,
                            description="Fraction of time servicing interrupts [0, 1]")


class SchedulerState(BaseModel):
    cpus: Dict[int, CPUMetrics] = Field(..., description="Metrics mapped by CPU core ID")
    total_context_switches: int = Field(..., ge=0,
                                        description="Cumulative context-switch counter")
    avg_latency_ms: float = Field(..., ge=0.0,
                                  description="Average scheduler latency observed")


class MemMetrics(BaseModel):
    pressure: float = Field(..., ge=0.0, le=1.0,
                            description="Memory pressure indicator [0, 1]")
    reclaim_rate_mb_s: float = Field(..., ge=0.0,
                                     description="Pages reclaimed per second (MB)")
    cache_hit_rate: float = Field(..., ge=0.0, le=1.0,
                                  description="Page cache hit rate [0, 1]")


class NetMetrics(BaseModel):
    rx_queue_len: int = Field(..., ge=0, description="Receive queue depth backlog")
    tx_queue_len: int = Field(..., ge=0, description="Transmit queue depth backlog")
    tcp_retransmits: int = Field(..., ge=0, description="Active TCP segments re-transmitted")


class KernelIntermediateRepresentation(BaseModel):
    """
    KIR: Kernel Intermediate Representation.

    A structured, validated snapshot of multiple OS subsystems. Every field
    carries ingestion-time range constraints so downstream consumers can
    trust basic data quality without re-checking.
    """
    timestamp: float = Field(..., gt=0.0, description="Unix timestamp of observation")
    scheduler: SchedulerState
    memory: Optional[MemMetrics] = None
    network: Optional[NetMetrics] = None


# ──────────────────────────────────────────────────────────────── #
# Actions — permissive by design; the Validator is the authority   #
# ──────────────────────────────────────────────────────────────── #

class SchedulerAction(BaseModel):
    target_latency_us: Optional[int] = Field(
        None, description="CFS target latency (kernel.sched_latency_ns / 1000)")
    wakeup_granularity_us: Optional[int] = Field(
        None, description="CFS wakeup granularity")
    migration_cost_ns: Optional[int] = Field(
        None, description="Estimated cost at which task migration is worthwhile")


class VMAction(BaseModel):
    swappiness: Optional[int] = Field(None, description="vm.swappiness (kernel accepts 0-200)")
    dirty_ratio: Optional[int] = Field(None, description="vm.dirty_ratio percentage")
    dirty_background_ratio: Optional[int] = Field(None, description="vm.dirty_background_ratio percentage")


class NetAction(BaseModel):
    # Plain str on purpose: new kernels add algorithms over time. The set of
    # values safe to WRITE is owned by the Validator allowlist, not the schema.
    congestion_control: Optional[str] = Field(
        None, description="net.ipv4.tcp_congestion_control algorithm name")
    queue_limit: Optional[int] = Field(None, description="net.core.netdev_max_backlog")


class PolicyAction(BaseModel):
    """Bounded policy decision covering coordinated subsystems."""
    policy_id: str = Field(..., min_length=1, max_length=64,
                           pattern=r"^[A-Za-z0-9._-]+$",
                           description="Identifier for the policy")
    scheduler: Optional[SchedulerAction] = None
    vm: Optional[VMAction] = None
    network: Optional[NetAction] = None


class PolicyMetadata(BaseModel):
    policy_id: str
    policy_version: str
    state_hash: str
    constraints_checked: bool
