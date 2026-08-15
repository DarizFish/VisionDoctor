from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from visiondoctor.adapters.gazebo import GazeboAdapter
from visiondoctor.cli import build_parser
from visiondoctor.demo.scenario import run_demo
from visiondoctor.schemas import Decision, FailureCategory, WorkflowState


def test_core_import_does_not_import_ros_or_gazebo() -> None:
    status = GazeboAdapter.availability()

    assert isinstance(status.available, bool)
    assert status.reason
    assert "rclpy" not in sys.modules
    assert "moveit_msgs" not in sys.modules


def test_cli_exposes_gazebo_rgbd_contract() -> None:
    args = build_parser().parse_args(
        ["gazebo-rgbd-contract", "--output-dir", "capture", "--case-id", "live"]
    )

    assert args.output_dir == Path("capture")
    assert args.case_id == "live"


def test_structured_moveit_failure_is_classified_as_transient_infrastructure() -> None:
    category, retryable = GazeboAdapter._classify_fixed_motion_failure(
        {
            "success": False,
            "error": "MoveIt failed VALIDATION_POSE: error 99999",
            "failure_stage": "VALIDATION_POSE",
            "moveit_error_code": 99999,
            "steps": [
                {"name": "HOME", "moveit_error_code": 1},
                {"name": "VALIDATION_POSE", "moveit_error_code": 99999},
            ],
        },
        returncode=1,
    )

    assert category is FailureCategory.PLANNING_EXECUTION_TRANSIENT
    assert retryable is True


@pytest.mark.integration
def test_real_gazebo_rgbd_capture_contract(tmp_path: Path) -> None:
    status = GazeboAdapter.availability()
    if not status.available or status.runtime != "docker":
        pytest.skip(status.reason)

    result = GazeboAdapter.run_rgbd_capture_contract(
        Path(__file__).resolve().parents[1],
        tmp_path / "gazebo-rgbd",
        case_id="live-rgbd",
    )

    assert result.success, result.stderr or result.container_logs[-4000:]
    assert not result.infrastructure_error
    assert result.payload["backend"] == "gazebo_rgbd"
    assert result.payload["runtime_uid"] == 1000
    assert result.payload["reference_source"] == "gazebo_truth"
    assert result.payload["depth_valid_ratio"] >= 0.60
    assert result.payload["message_stamp_spread_s"] <= 0.1
    assert result.payload["rgbd_translation_error_m"] <= 0.005
    assert result.payload["rgbd_rotation_error_rad"] <= 0.01745
    output = tmp_path / "gazebo-rgbd"
    assert (output / "rgb.png").is_file()
    assert (output / "depth.npy").is_file()
    assert (output / "gazebo_rgbd_capture.json").is_file()
    assert (output / "gazebo_rgbd_capture.log").is_file()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["case_id"] == "live-rgbd"
    assert manifest["source"] == "gazebo"
    assert "reference_t_base_object" not in manifest
    reference = json.loads((output / "qa_reference.json").read_text(encoding="utf-8"))
    assert reference["provider"] == "gz model runtime state"


@pytest.mark.integration
def test_real_gazebo_moveit_ur5e_fixed_motion_contract(
    tmp_path: Path, model_gateway_factory
) -> None:
    status = GazeboAdapter.availability()
    if not status.available or status.runtime != "docker":
        pytest.skip(status.reason)

    result = run_demo(
        tmp_path / "gazebo-demo",
        sandbox_mode="docker",
        robot_backend="gazebo",
        model_gateway=model_gateway_factory(),
    )
    gate = result.external_gate_results[0]
    run_root = Path(result.run_root)
    evidence = json.loads((run_root / "evidence_bundle.json").read_text(encoding="utf-8"))
    references = json.loads(
        (run_root / "trusted_qa" / "references.json").read_text(encoding="utf-8")
    )

    assert sum(case["source"] == "gazebo" for case in evidence["cases"]) == 1
    assert sum(reference["source_type"] == "gazebo_truth" for reference in references) == 1
    if result.state is WorkflowState.AWAITING_TECHNICAL_REVIEW:
        validation = result.candidate_validations[-1]
        trace = (run_root / "trace.jsonl").read_text(encoding="utf-8")
        review = json.loads((run_root / "technical_review.json").read_text(encoding="utf-8"))

        assert gate.failure_category is FailureCategory.PLANNING_EXECUTION_TRANSIENT
        assert gate.retry_exhausted is True
        assert gate.attempt_count == 2
        assert validation.decision is Decision.NEEDS_REVIEW
        assert validation.rollback_result == "NOT_REQUIRED"
        assert review["patch_retry_requested"] is False
        assert "PATCH_REJECTED" not in trace
        assert "candidate_rejected" not in trace
        return

    assert result.state is WorkflowState.AWAITING_HUMAN_APPROVAL
    assert gate.passed, gate.details or gate.logs[-4000:]
    assert gate.payload["backend"] == "gazebo"
    assert gate.payload["robot"] == "UR5e"
    assert gate.payload["runtime_uid"] == 1000
    assert gate.payload["motion_sequence"] == [
        "HOME",
        "OBSERVATION_POSE",
        "VALIDATION_POSE",
        "HOME",
    ]
    assert gate.payload["tcp_translation_error_m"] <= 0.005
    assert gate.payload["tcp_rotation_error_rad"] <= 0.01745
    assert all(step["moveit_error_code"] == 1 for step in gate.payload["steps"])
    validation = result.candidate_validations[-1]
    rgbd_metric = next(
        item for item in validation.metric_results if item.name == "gazebo_rgbd_scene_count"
    )
    assert rgbd_metric.mandatory and rgbd_metric.passed
    assert rgbd_metric.value == 1.0
    assert rgbd_metric.threshold == 1.0
    metric = next(
        item for item in validation.metric_results if item.name == "gazebo_moveit_fixed_motion"
    )
    assert metric.mandatory and metric.passed
    candidate_id = result.selected_candidate.candidate_id
    assert (run_root / f"{candidate_id}_external_gate.json").is_file()
    assert (run_root / f"{candidate_id}_gazebo.log").is_file()
