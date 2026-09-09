"""agents.py — agent harness operating INSIDE the virtual kernel.

context.txt lines 178-216: don't have "an AI scientist"; put agents inside
the environment sharing one virtual cell/kernel:

  Research agent     proposes hypotheses
  Perturbation agent chooses interventions (virtual_screen)
  Pathway analyst    interprets the response (latency/util deltas)
  Adversarial agent  surrogate critic: finds refutations / high-uncertainty claims
  Design agent       searches combinations of sequential perturbations
  Validation agent   selects experiments for oracle (external) validation

All agents operate on the same VirtualKernel; findings update the search.
Deterministic + dependency-free so tests stay hermetic.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from learned_kernel.policy.schemas import PolicyAction

from .active import rank_probes
from .environment import VirtualKernel
from .search import validate_topk, virtual_screen


@dataclass
class AgentFinding:
    agent: str
    claim: str
    detail: dict = field(default_factory=dict)
    confidence: float = 0.5
    round: int = 0


@dataclass
class LabReport:
    findings: List[AgentFinding]
    best_latencies: List[int]
    best_predicted_reward: float
    best_oracle_reward: Optional[float]
    virtual_worlds: int
    oracle_calls: int
    debate_rounds: int = 0
    revised: bool = False

    def summary(self) -> dict:
        return {"n_findings": len(self.findings),
                "best_latencies": self.best_latencies,
                "best_predicted_reward": self.best_predicted_reward,
                "best_oracle_reward": self.best_oracle_reward,
                "virtual_worlds": self.virtual_worlds,
                "oracle_calls": self.oracle_calls,
                "debate_rounds": self.debate_rounds,
                "revised": self.revised}

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


class VirtualLab:
    """Orchestrates specialist agents over one shared VirtualKernel.

    Debate protocol (adversarial review): after the first proposal round, the
    adversarial agent challenges the leading candidate on uncertainty and
    robustness grounds; the design agent must either revise the candidate or
    record a rebuttal with evidence. Rounds are bounded and logged.
    """

    def __init__(self, vkernel: VirtualKernel, oracle=None,
                 uncertainty_veto: float = 0.05):
        self.vk = vkernel
        self.oracle = oracle
        self.uncertainty_veto = uncertainty_veto

    def run(self, horizon: int = 3, beam_width: int = 6,
            validate_k: int = 1, debate_rounds: int = 1) -> LabReport:
        findings: List[AgentFinding] = []
        kir0 = self.vk.observe()

        # Research agent: hypotheses = latency regimes worth testing
        findings.append(AgentFinding(
            "research",
            "tight windows suit saturated load; loose windows suit idle load",
            {"horizon": horizon}))

        # Perturbation + Design agents: virtual screening over sequences
        results = virtual_screen(self.vk.ensemble, kir0, horizon=horizon,
                                 beam_width=beam_width)
        worlds = getattr(virtual_screen, "last_virtual_worlds", 0)
        findings.append(AgentFinding(
            "perturbation",
            f"screened {worlds} virtual worlds; top={results[0].latencies}",
            {"predicted_reward": results[0].predicted_reward}))
        findings.append(AgentFinding(
            "design",
            f"sequential combo {results[0].latencies} beats single-shot baselines virtually",
            {}))

        # Pathway analyst: interpret best trajectory branch
        self.vk.branch("analysis")
        traj = []
        for lat in results[0].latencies:
            from learned_kernel.policy.schemas import SchedulerAction
            act = PolicyAction(policy_id="lab",
                               scheduler=SchedulerAction(target_latency_us=lat))
            traj.append(self.vk.intervene(act, branch="analysis"))
        lat0 = kir0.scheduler.avg_latency_ms
        lat1 = traj[-1].scheduler.avg_latency_ms if traj else lat0
        findings.append(AgentFinding(
            "analyst",
            f"latency {lat0:.2f}ms -> {lat1:.2f}ms over {len(traj)} virtual steps",
            {"delta_ms": lat1 - lat0}))

        # Adversarial / critic agent: where is the surrogate least trustworthy?
        probes = rank_probes(self.vk.ensemble, kir0)
        findings.append(AgentFinding(
            "adversarial",
            f"highest-uncertainty probe is {probes[0].target_latency_us}us "
            f"(u={probes[0].uncertainty:.4f}); claims there need oracle checks",
            {"top_probe_us": probes[0].target_latency_us,
             "uncertainty": probes[0].uncertainty}))

        # Validation agent: ground top-k in the exact oracle
        oracle_calls = 0
        best_oracle = None
        if self.oracle is not None:
            validate_topk(results, self.oracle, kir0, k=validate_k)
            oracle_calls = sum(r.oracle_calls for r in results[:validate_k])
            best_oracle = results[0].oracle_reward
            findings.append(AgentFinding(
                "validation",
                f"oracle-checked top-{validate_k}: reward={best_oracle}",
                {"oracle_calls": oracle_calls,
                 "per_oracle_strength": (best_oracle or 0.0) / max(oracle_calls, 1)},
                confidence=0.9))

        revised = False
        for rd in range(1, debate_rounds + 1):
            challenge = self._challenge(results[0].latencies, kir0, horizon)
            findings.append(AgentFinding(
                "adversarial",
                f"round-{rd} challenge: {challenge['verdict']}",
                challenge, confidence=challenge.get("confidence", 0.5), round=rd))
            if challenge["veto"]:
                alt = [r for r in results[1:] if r.latencies != results[0].latencies]
                if alt:
                    results.insert(0, alt[0])
                    revised = True
                    findings.append(AgentFinding(
                        "design",
                        f"round-{rd} revision: adopted {alt[0].latencies} "
                        f"(pred={alt[0].predicted_reward:.4f})",
                        {"superseded": results[1].latencies}, confidence=0.6,
                        round=rd))
            else:
                findings.append(AgentFinding(
                    "design",
                    f"round-{rd} rebuttal: holding {results[0].latencies} "
                    f"({challenge['reason']})",
                    {}, confidence=0.7, round=rd))

        return LabReport(findings=findings, best_latencies=results[0].latencies,
                         best_predicted_reward=results[0].predicted_reward,
                         best_oracle_reward=best_oracle,
                         virtual_worlds=worlds, oracle_calls=oracle_calls,
                         debate_rounds=debate_rounds, revised=revised)

    def _challenge(self, latencies: List[int], kir0, horizon: int) -> dict:
        """Surrogate-critic: veto the leader if its trajectory is uncertain."""
        from learned_kernel.policy.schemas import SchedulerAction
        kir = kir0.model_copy(deep=True)
        worst_u, worst_step = 0.0, -1
        for i, lat in enumerate(latencies):
            act = PolicyAction(policy_id="critic",
                               scheduler=SchedulerAction(target_latency_us=lat))
            u = self.vk.ensemble.uncertainty(kir, act)
            if u > worst_u:
                worst_u, worst_step = u, i
            kir = self.vk.ensemble.predict_kir(kir, act)
        veto = bool(worst_u > self.uncertainty_veto)
        return {"verdict": ("VETO trajectory too uncertain "
                            f"(max u={worst_u:.4f} at step {worst_step})" if veto
                            else f"ACCEPT max trajectory uncertainty {worst_u:.4f} "
                            "within tolerance"),
                "reason": (f"max-u {worst_u:.4f} exceeds veto {self.uncertainty_veto}"
                           if veto else "uncertainty bounded along virtual path"),
                "max_uncertainty": worst_u, "worst_step": worst_step,
                "veto": veto, "confidence": min(0.95, 0.5 + worst_u * 10)}
