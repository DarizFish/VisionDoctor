from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from harness.gazebo.build_demo_project import build
from harness.gazebo.controller import PickCellController
from harness.gazebo.geometry import compose, pose, pose_error
from harness.gazebo.models import GraspClassification, classify_grasp


def test_fault_and_normal_outcomes_have_distinct_classification() -> None:
    normal = classify_grasp(
        0.002,
        0.01,
        position_tolerance_m=0.006,
        rotation_tolerance_rad=0.02,
        motion_completed=True,
    )
    faulty = classify_grasp(
        0.22,
        0.26,
        position_tolerance_m=0.006,
        rotation_tolerance_rad=0.02,
        motion_completed=True,
    )
    assert normal.classification is GraspClassification.SUCCESS
    assert faulty.classification is GraspClassification.FAILURE


def test_demo_bundle_contains_normal_and_faulty_history(tmp_path: Path) -> None:
    workspace = tmp_path / "faulty"
    bundle = tmp_path / "pick-a17.bundle"
    build(workspace, bundle)
    log = subprocess.run(
        ["git", "-C", str(workspace), "log", "--oneline", "main"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert len(log.stdout.splitlines()) == 2
    verify = subprocess.run(
        ["git", "bundle", "verify", str(bundle)],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "main" in verify.stdout


def test_normal_revision_recovers_while_faulty_head_misses_pick_pose(tmp_path: Path) -> None:
    workspace = tmp_path / "faulty"
    build(workspace, tmp_path / "pick-a17.bundle")
    algorithm_input = {
        "run_id": "test-run",
        "part_id": "A",
        "detection_id": "test-detection",
        "detected_part_in_camera": {
            "position": [1.781695155, -0.024170628, -0.085942750],
            "quaternion_xyzw": [
                0.820718088,
                0.176804984,
                -0.347379300,
                0.417719330,
            ],
        },
    }

    def replay(revision: str) -> dict[str, object]:
        subprocess.run(["git", "-C", str(workspace), "checkout", "--quiet", revision], check=True)
        input_path = workspace / "input.json"
        output_path = workspace / "output.json"
        log_path = workspace / "app.jsonl"
        input_path.write_text(json.dumps(algorithm_input), encoding="utf-8")
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(workspace)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pick_demo.replay",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--log",
                str(log_path),
            ],
            cwd=workspace,
            env=environment,
            check=True,
        )
        return json.loads(output_path.read_text(encoding="utf-8"))

    faulty = replay("main")
    normal = replay("HEAD~1")
    tool = pose(
        {
            "position": [0.000, 0.000, 0.035],
            "quaternion_xyzw": [0.0, 0.0, 0.0099998, 0.99995],
        }
    )
    desired_tcp = pose(
        {
            "position": [0.500000000, -0.100000000, 0.148000000],
            "quaternion_xyzw": [0.000000000, 1.000000000, 0.000000000, 0.000000000],
        }
    )
    normal_error = pose_error(desired_tcp, compose(pose(normal["commanded_flange_base"]), tool))
    faulty_error = pose_error(desired_tcp, compose(pose(faulty["commanded_flange_base"]), tool))
    assert classify_grasp(
        *normal_error,
        position_tolerance_m=0.006,
        rotation_tolerance_rad=0.02,
        motion_completed=True,
    ).success
    assert not classify_grasp(
        *faulty_error,
        position_tolerance_m=0.006,
        rotation_tolerance_rad=0.02,
        motion_completed=True,
    ).success


def test_bundle_export_has_no_private_scoring_material(tmp_path: Path) -> None:
    controller = PickCellController(runtime_root=tmp_path / "runtime")
    run_id = "run-test"
    run_root = controller.runtime_root / "runs" / run_id
    (run_root / "parts" / "A").mkdir(parents=True)
    (run_root / "parts" / "A" / "observation.json").write_text(
        json.dumps({"kind": "camera_observation"}), encoding="utf-8"
    )
    manifest_path = controller._write_bundle(
        run_root,
        run_id,
        {"commit": "deadbeef"},
        [{"at": "2026-09-02T00:00:00+00:00", "event": "capture", "part_id": "A"}],
        [
            {
                "part_id": "A",
                "classification": "missed_pick_pose",
                "success": False,
                "position_error_m": 0.2,
                "rotation_error_rad": 0.1,
            }
        ],
    )
    encoded = manifest_path.read_text(encoding="utf-8").lower()
    for forbidden in ("private", "truth", "score", "answer", "patch", "expected"):
        assert forbidden not in encoded
    artifact = json.loads(encoded)["artifacts"][0]
    assert artifact["captured_at"]
    assert artifact["clock_domain"] == "utc_file_mtime"
    export = controller.export_bundle(run_id, tmp_path / "export")
    assert (export / "bundle.json").is_file()
    assert (export / "parts" / "A" / "observation.json").is_file()


def test_bundle_export_stops_when_a_declared_artifact_digest_changes(tmp_path: Path) -> None:
    controller = PickCellController(runtime_root=tmp_path / "runtime")
    run_id = "run-tampered"
    run_root = controller.runtime_root / "runs" / run_id
    run_root.mkdir(parents=True)
    artifact_path = run_root / "observation.json"
    artifact_path.write_text(json.dumps({"kind": "camera_observation"}), encoding="utf-8")
    controller._write_bundle(run_root, run_id, {"commit": "deadbeef"}, [], [])
    artifact_path.write_text(json.dumps({"kind": "changed"}), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        controller.export_bundle(run_id, tmp_path / "export")


def test_bundle_rejects_private_data_even_under_a_benign_filename(tmp_path: Path) -> None:
    controller = PickCellController(runtime_root=tmp_path / "runtime")
    run_root = controller.runtime_root / "runs" / "run-private"
    run_root.mkdir(parents=True)
    (run_root / "observation.json").write_text(
        json.dumps({"private_target": [0.1, 0.2, 0.3]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="private data"):
        controller._write_bundle(
            run_root,
            "run-private",
            {"commit": "deadbeef"},
            [],
            [],
        )


def test_controller_status_is_local_state_without_a_running_container(
    monkeypatch, tmp_path: Path
) -> None:
    controller = PickCellController(runtime_root=tmp_path / "runtime")
    monkeypatch.setattr(controller, "_container_running", lambda: False)
    monkeypatch.setattr(controller, "_image_ready", lambda: False)
    status = controller.status()
    assert status["running"] is False
    assert status["gazebo_gui_running"] is False
    assert status["default_workspace"].endswith("projects\\topdown-clearance-final-faulty")
    assert status["selected_workspace"].endswith("projects\\topdown-clearance-final-faulty")
    assert status["image"] == "visiondoctor/ros-gazebo:jazzy-v1"


def test_controller_remembers_console_workspace_across_sessions(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    controller = PickCellController(runtime_root=runtime_root)
    selected = controller.remember_workspace(tmp_path / "candidate-project")

    fresh_controller = PickCellController(runtime_root=runtime_root)

    assert selected == (tmp_path / "candidate-project").resolve()
    assert fresh_controller.selected_workspace == selected
    assert fresh_controller.status()["selected_workspace"] == str(selected)


def test_harness_python_has_no_product_import_or_api_route() -> None:
    root = Path(__file__).resolve().parents[1] / "harness" / "gazebo"
    source = "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*.py"))
    assert "import visiondoctor" not in source
    assert "from visiondoctor" not in source
    assert "/api/v1/" not in source
