import tempfile
import unittest
from pathlib import Path

from learned_kernel.policy.schemas import CPUMetrics, SchedulerState
from learned_kernel.policy.schemas import KernelIntermediateRepresentation, PolicyAction, SchedulerAction
from learned_kernel.telemetry.provenance import ProvenanceLogger


def kir(ts=1.0):
    return KernelIntermediateRepresentation(
        timestamp=ts,
        scheduler=SchedulerState(
            cpus={0: CPUMetrics(utilization=0.5, runnable_tasks=1, irq_time=0.0)},
            total_context_switches=10,
            avg_latency_ms=2.0),
    )


class ProvenanceChainTests(unittest.TestCase):
    def _write(self, path, n=5):
        log = ProvenanceLogger(str(path))
        for i in range(n):
            log.record_decision(
                kstate=kir(ts=i + 1.0),
                action=PolicyAction(policy_id="t", scheduler=SchedulerAction(target_latency_us=6000)),
                policy_id="t-pol", policy_version="1",
                is_validated=True, reward=float(i))
        return log

    def test_chain_verifies_when_untampered(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "audit.jsonl"
            self._write(path)
            ok, bad_seq, reason = ProvenanceLogger.verify_chain(str(path))
            self.assertTrue(ok, f"{bad_seq}: {reason}")
            self.assertEqual(reason, "chain intact")

    def test_content_tampering_detected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "audit.jsonl"
            self._write(path)
            lines = path.read_text().splitlines()
            rec = __import__("json").loads(lines[2])
            rec["observed_reward"] = 99999.0          # forge history
            lines[2] = __import__("json").dumps(rec)
            path.write_text("\n".join(lines) + "\n")
            ok, bad_seq, reason = ProvenanceLogger.verify_chain(str(path))
            self.assertFalse(ok)
            self.assertEqual(bad_seq, 3)

    def test_deletion_detected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "audit.jsonl"
            self._write(path)
            lines = path.read_text().splitlines()
            del lines[1]                               # remove a middle record
            path.write_text("\n".join(lines) + "\n")
            ok, bad_seq, _ = ProvenanceLogger.verify_chain(str(path))
            self.assertFalse(ok)
            self.assertEqual(bad_seq, 2)

    def test_reorder_detected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "audit.jsonl"
            self._write(path)
            lines = path.read_text().splitlines()
            lines[0], lines[1] = lines[1], lines[0]
            path.write_text("\n".join(lines) + "\n")
            ok, _, reason = ProvenanceLogger.verify_chain(str(path))
            self.assertFalse(ok)

    def test_full_state_hash_width(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "audit.jsonl"
            self._write(path, n=1)
            rec = __import__("json").loads(path.read_text().splitlines()[0])
            self.assertEqual(len(rec["kernel_state_hash"]), 64)
            self.assertNotEqual(rec["hardware"], "generic-x86_64-sim")

    def test_resume_appends_to_existing_chain(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "audit.jsonl"
            self._write(path, n=3)
            self._write(path, n=2)                     # new logger resumes chain
            ok, _, reason = ProvenanceLogger.verify_chain(str(path))
            self.assertTrue(ok, reason)
            records = ProvenanceLogger.read_records(str(path))
            self.assertEqual([r["seq_no"] for r in records], [1, 2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
