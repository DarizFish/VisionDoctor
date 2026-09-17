from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from harness.gazebo.controller import PickCellController
from harness.gazebo.scenarios import make_insufficient_control
from harness.gazebo.sensor_faults import apply_sensor_fault
from visiondoctor.case import Case
from visiondoctor.environment import FileBundleAdapter
from visiondoctor.investigation import Toolbox
from visiondoctor.investigation.view import build_view

FIXTURE = Path(__file__).parent / "fixtures/grasp-rgbd"


def capture_at(root: Path) -> Path:
    capture = root / "parts/A/capture"
    capture.mkdir(parents=True)
    for name in ("rgb.png", "camera-info.json", "capture.json"):
        shutil.copy2(FIXTURE / name, capture / name)
    np.save(capture / "depth.npy", np.load(FIXTURE / "depth.npz")["depth"])
    meta = json.loads((capture / "capture.json").read_text())
    meta["phase"] = "before_command"
    (capture / "capture.json").write_text(json.dumps(meta))
    return capture


def test_capture_phase_reaches_model_and_unavailable_vision_is_not_examined(tmp_path: Path) -> None:
    cell = PickCellController(runtime_root=tmp_path)
    root = tmp_path / "runs/test"
    capture_at(root)
    cell._write_bundle(root, "test", {}, [], [])
    adapter = FileBundleAdapter(root)
    bundle = adapter.collect()
    case = Case("C", "阶段")
    case.admit(bundle)
    rgb = next(e for e in case.evidence if e.reference.endswith("/rgb.png"))
    assert rgb.phase == "before_command"
    view = build_view(case, bundle)
    assert next(e for e in view["evidence_catalogue"]
                if e["evidence_id"] == rgb.evidence_id)["phase"] == "before_command"
    read = Toolbox(case, adapter, bundle).read_evidence([rgb.evidence_id])[0]
    assert "unavailable" in read
    assert rgb.evidence_id not in case.examined

    class Vision:
        def assess(self, path, **kwargs):
            assert "before_command" in kwargs["user_context"]
            assert "请核对目标身份" in kwargs["user_context"]
            return {"visible": "工件在台面上", "limits": "没有动作阶段画面"}

    observed = Toolbox(case, adapter, bundle, vision=Vision()).read_evidence(
        [rgb.evidence_id], question="请核对目标身份",
    )[0]
    assert observed["phase"] == "before_command"
    assert observed["question"] == "请核对目标身份"
    assert rgb.evidence_id in case.examined


def test_raw_depth_separates_sensor_data_from_downstream_corruption(tmp_path: Path) -> None:
    capture = capture_at(tmp_path)
    original = (capture / "depth.npy").read_bytes()
    apply_sensor_fault(capture, "target_depth_dropout")
    assert (capture / "raw-depth.npy").read_bytes() == original
    before = np.load(capture / "raw-depth.npy")
    after = np.load(capture / "depth.npy")
    assert np.count_nonzero(np.isfinite(before) & ~np.isfinite(after)) > 100
    metadata = json.loads((capture / "capture.json").read_text())
    assert metadata["depth_streams"]["sensor_output"] == "raw-depth.npy"
    assert "sensor_fault" not in metadata


def test_insufficient_control_has_no_internal_failure_label_or_usable_depth(tmp_path: Path) -> None:
    cell = PickCellController(runtime_root=tmp_path / "runtime")
    root = cell.runtime_root / "runs/source"
    capture_at(root)
    cell._write_bundle(root, "source", {}, [], [
        {"part_id": "A", "classification": "perception_unavailable", "success": False},
    ])
    result = make_insufficient_control(root, cell)
    bundle = FileBundleAdapter(Path(result["bundle"]))
    manifest = bundle.collect()
    assert len(manifest.artifacts) == 3
    assert manifest.results[0].classification == "operator_reported_failure"
    for artifact in manifest.artifacts:
        payload = bundle.read_artifact(artifact.path)
        if artifact.media_type == "application/json":
            assert b"perception_unavailable" not in payload and b"depth_valid_ratio" not in payload
        assert not artifact.path.endswith(".npy")
    case = Case("C", "证据不足")
    case.admit(manifest)
    refs = {e.reference: e.evidence_id for e in case.evidence}
    rgb = refs["parts/A/capture/rgb.png"]
    camera = refs["parts/A/capture/camera-info.json"]
    with pytest.raises(ValueError, match="not a NumPy depth array"):
        Toolbox(case, bundle, manifest).measure_rgbd_region(
            rgb_evidence_id=rgb, depth_evidence_id=camera,
            camera_evidence_id=camera, roi_xyxy=[10, 10, 20, 20],
        )
    assert not case.examined


def test_missing_motion_measurement_cannot_be_replaced_by_the_command(tmp_path: Path) -> None:
    cell = PickCellController(runtime_root=tmp_path)
    outcome = cell._assess(tmp_path, "A", {
        "commanded_flange_base": {"position": [0, 0, 9], "quaternion_xyzw": [0, 0, 0, 1]},
    }, {"success": False, "error": "IK failed"})
    assert not outcome.success
    assert outcome.position_error_m is None and outcome.rotation_error_rad is None
