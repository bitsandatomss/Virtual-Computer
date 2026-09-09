from __future__ import annotations

import json
from pathlib import Path

from self_compiler.ai_engine import AIEngine
from self_compiler.modes import RepairLoop


def _events(journal_path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in journal_path.read_text(encoding="utf-8").splitlines()
    ]


def test_successful_offline_repair_is_fully_journaled(tmp_path):
    source = tmp_path / "broken.c"
    source.write_text(
        "int main(void) {\n    int value = 1\n    return value - 1;\n}\n",
        encoding="utf-8",
    )
    journal_path = tmp_path / "journal.jsonl"
    loop = RepairLoop(
        str(source),
        max_attempts=1,
        ai=AIEngine(offline=True),
        journal_path=str(journal_path),
    )
    assert loop.run("repair") is True

    events = _events(journal_path)
    names = [event["event"] for event in events]
    assert names[0] == "run-start"
    assert names[-1] == "run-end"
    assert events[-1]["status"] == "success"
    assert "ai-requested" in names
    ai_event = next(event for event in events if event["event"] == "ai-requested")
    assert ai_event["offline"] is True
    assert "promotion" in names
    assert all(event["schema"] == "gcc-ai.repair-journal.v1" for event in events)


def test_exhausted_repair_records_final_status(tmp_path):
    source = tmp_path / "broken.c"
    original = "int main(void) { this is invalid }\n"
    source.write_text(original, encoding="utf-8")
    journal_path = tmp_path / "journal.jsonl"

    class DoNothingAI:
        model_id = "stub"
        offline = True

        def propose_patch(self, source, diagnostics, mode):
            return source  # never makes progress

    loop = RepairLoop(
        str(source),
        max_attempts=2,
        ai=DoNothingAI(),
        journal_path=str(journal_path),
    )
    assert loop.run("repair") is False
    events = _events(journal_path)
    assert events[-1]["event"] == "run-end"
    assert events[-1]["status"] == "exhausted"
    requested = [e for e in events if e["event"] == "ai-requested"]
    assert [e["attempt"] for e in requested] == [1, 2]


def test_journal_disabled_by_default_writes_nothing(tmp_path):
    source = tmp_path / "valid.c"
    source.write_text("int main(void) { return 0; }\n", encoding="utf-8")
    loop = RepairLoop(str(source), max_attempts=1, ai=AIEngine(offline=True))
    assert loop.journal.enabled is False
    assert loop.run("repair") is True
    assert not list(tmp_path.glob("journal.jsonl"))
