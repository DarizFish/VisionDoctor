from __future__ import annotations

import json
from pathlib import Path

from visiondoctor.reliability import run_reliability_gate
from visiondoctor.workflow import DemoRunResult


def test_reliability_gate_persists_incremental_summary(
    demo_result: DemoRunResult, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr("visiondoctor.reliability.run_demo", lambda *args, **kwargs: demo_result)

    summary = run_reliability_gate(
        tmp_path / "qualification",
        runs=2,
        sandbox_mode="local",
        robot_backend="dataset",
    )

    assert summary["all_passed"]
    assert summary["passed_runs"] == 2
    persisted = json.loads(
        (tmp_path / "qualification" / "reliability_summary.json").read_text(encoding="utf-8")
    )
    assert persisted == summary
