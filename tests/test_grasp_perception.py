from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from harness.gazebo.build_demo_project import TEMPLATE_ROOT
from harness.gazebo.controller import PickCellController
from harness.gazebo.demo_project_template.pick_demo.geometry import compose
from harness.gazebo.demo_project_template.pick_demo.perception import detect

FIXTURE = Path(__file__).parent / "fixtures/grasp-rgbd"


@pytest.fixture
def captured(tmp_path: Path) -> Path:
    for name in ("rgb.png", "camera-info.json", "capture.json"):
        shutil.copy2(FIXTURE / name, tmp_path / name)
    np.save(tmp_path / "depth.npy", np.load(FIXTURE / "depth.npz")["depth"])
    return tmp_path


def test_actual_capture_localizes_both_parts_without_supplied_positions(captured: Path) -> None:
    calibration = yaml.safe_load((TEMPLATE_ROOT / "config/cell_calibration.yaml").read_text())
    for part, center in (("A", [0.5, -0.1, 0.1]), ("B", [0.5, -0.3, 0.1])):
        result = detect(captured, part, "test", TEMPLATE_ROOT)
        assert result["status"] == "detected"
        measured = compose(calibration["camera_to_base"], result["detected_part_in_camera"])
        assert np.linalg.norm(np.asarray(measured["position"]) - center) < 0.004
        assert result["roi_xyxy"][0] > 250  # Robot highlights are not in the workpiece region.
        assert result["source_hashes"]["depth.npy"]
        assert "orientation supplied" in result["pose_scope"]


def test_pixel_translation_changes_estimated_position(captured: Path) -> None:
    before = detect(captured, "A", "test", TEMPLATE_ROOT)
    rgb = np.asarray(Image.open(captured / "rgb.png"))
    depth = np.load(captured / "depth.npy")
    Image.fromarray(np.roll(rgb, 12, axis=1)).save(captured / "rgb.png")
    np.save(captured / "depth.npy", np.roll(depth, 12, axis=1))
    after = detect(captured, "A", "test", TEMPLATE_ROOT)
    difference = (np.asarray(after["detected_part_in_camera"]["position"])
                  - before["detected_part_in_camera"]["position"])
    assert np.linalg.norm(difference) > 0.02
    assert before["source_hashes"] != after["source_hashes"]


def test_target_depth_loss_produces_no_fabricated_pose(captured: Path) -> None:
    before = detect(captured, "A", "test", TEMPLATE_ROOT)
    x0, y0, x1, y1 = before["roi_xyxy"]
    depth = np.load(captured / "depth.npy")
    depth[y0:y1, x0:x1] = np.nan
    np.save(captured / "depth.npy", depth)
    after = detect(captured, "A", "test", TEMPLATE_ROOT)
    assert after["status"] == "insufficient_target_depth"
    assert "detected_part_in_camera" not in after
    assert after["target_depth_valid_ratio"] == 0


def test_rgb_loss_and_unknown_camera_contract_are_distinct(captured: Path) -> None:
    Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8)).save(captured / "rgb.png")
    assert detect(captured, "A", "test", TEMPLATE_ROOT)["status"] == "target_color_not_found"
    info = json.loads((captured / "camera-info.json").read_text())
    del info["depth_unit"]
    (captured / "camera-info.json").write_text(json.dumps(info))
    assert detect(captured, "A", "test", TEMPLATE_ROOT)["status"] == "unsupported_camera_contract"


def test_depth_fault_cycle_exports_actual_inputs_without_motion_or_fault_labels(
    captured: Path, tmp_path: Path, monkeypatch,
) -> None:
    controller = PickCellController(runtime_root=tmp_path / "runtime")
    project = tmp_path / "project"
    shutil.copytree(TEMPLATE_ROOT, project)
    monkeypatch.setattr(controller, "_require_running", lambda: None)
    monkeypatch.setattr(controller, "_warm_cell", lambda: None)

    def capture(destination: Path, **_) -> dict:
        destination.mkdir(parents=True)
        for name in ("rgb.png", "depth.npy", "camera-info.json", "capture.json"):
            shutil.copy2(captured / name, destination / name)
        return json.loads((destination / "capture.json").read_text())

    def unexpected_motion(*args):
        pytest.fail("perception failure must not produce a robot motion")

    monkeypatch.setattr(controller, "capture", capture)
    monkeypatch.setattr(controller, "_execute_motion", unexpected_motion)
    result = controller.run_cycle(project, sensor_fault="target_depth_dropout")
    assert not result["overall_success"]
    assert all(row["classification"] == "perception_unavailable" for row in result["results"])
    manifest = json.loads(Path(result["bundle_path"]).read_text())
    assert all("scenario-labels" not in row["path"] for row in manifest["artifacts"])
    export = controller.export_bundle(result["run_id"], tmp_path / "exports")
    detection = json.loads((export / "parts/A/algorithm/input.json").read_text())
    assert detection["status"] == "insufficient_target_depth"
    assert "detected_part_in_camera" not in detection
    public = [file.read_text() for file in export.rglob("*.json")]
    assert not any("sensor_fault" in text for text in public)
