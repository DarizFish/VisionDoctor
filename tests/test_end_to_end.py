from __future__ import annotations

import json
import subprocess
from pathlib import Path

from visiondoctor.schemas import Decision, HumanAction, WorkflowState
from visiondoctor.workflow import DemoRunResult


def _metric(report, name: str):
    return next(metric for metric in report.metric_results if metric.name == name)


def _policy(report, name: str):
    return next(check for check in report.policy_checks if check.name == name)


def test_full_dataset_loop_gates_model_generated_root_fix(
    demo_result: DemoRunResult,
) -> None:
    assert demo_result.state is WorkflowState.AWAITING_HUMAN_APPROVAL
    assert demo_result.baseline_validation.decision is Decision.PASS
    assert demo_result.faulty_validation.decision is Decision.REJECTED
    assert demo_result.diagnosis.confirmed
    assert demo_result.diagnosis.model == "test-protocol-double"
    assert len(demo_result.candidate_validations) == 1

    root_fix = demo_result.candidate_validations[0]
    assert root_fix.decision is Decision.PASS
    assert root_fix.human_action is HumanAction.AWAITING_APPROVAL
    assert len(root_fix.passed_cases) == 50
    assert all(metric.passed for metric in root_fix.metric_results if metric.mandatory)
    gazebo_rgbd = _metric(root_fix, "gazebo_rgbd_scene_count")
    assert gazebo_rgbd.passed
    assert gazebo_rgbd.value == 0.0
    assert gazebo_rgbd.threshold == 0.0
    assert _policy(root_fix, "expected_change_set").passed
    assert "2 changed" in _policy(root_fix, "changed_file_count").details


def test_run_artifacts_are_auditable_and_source_repo_is_unchanged(
    demo_result: DemoRunResult,
) -> None:
    run_root = Path(demo_result.run_root)
    candidate_id = demo_result.selected_candidate.candidate_id
    required = {
        "incident.json",
        "evidence_bundle.json",
        "diagnosis_report.json",
        f"{candidate_id}.diff",
        f"{candidate_id}_validation.json",
        "trace.jsonl",
        "release_decision.json",
        "postmortem.md",
    }
    assert required <= {path.name for path in run_root.rglob("*") if path.is_file()}
    release = json.loads((run_root / "release_decision.json").read_text(encoding="utf-8"))
    assert release["decision"] == "AWAITING_HUMAN_APPROVAL"
    assert release["automatic_merge"] is False
    root_patch = (run_root / f"{candidate_id}.diff").read_text(encoding="utf-8")
    assert "tests/test_transform_order_regression.py" in root_patch

    repository = Path(demo_result.incident.repository.path)
    head = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
    assert head == demo_result.incident.faulty_commit
    assert status == ""
