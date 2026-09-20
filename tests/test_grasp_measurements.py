from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from visiondoctor.case import CaseService, Evidence
from visiondoctor.investigation import Toolbox
from visiondoctor.investigation.measurements import check_alignment, measure_region


def camera() -> dict:
    return {
        "width": 100, "height": 80, "frame_id": "camera",
        "intrinsics": [100, 0, 50, 0, 100, 40, 0, 0, 1], "distortion": [0] * 5,
        "rgb_depth_registered": True, "depth_unit": "m", "depth_measurement": "optical_z",
        "pose_axes": "optical_x_right_y_down_z_forward",
    }


def test_target_depth_hole_is_not_hidden_by_healthy_global_ratio() -> None:
    rgb = np.full((80, 100, 3), 128, dtype=np.uint8)
    depth = np.full((80, 100), 2.0)
    depth[35:45, 45:55] = np.nan
    measured = measure_region(rgb, depth, camera(), [45, 35, 55, 45])
    assert measured["full_depth"]["valid_ratio"] > 0.98
    assert measured["roi_depth"]["valid_ratio"] == 0
    assert "surface_median_optical_xyz_m" not in measured["roi_depth"]
    assert measured["projection"] is None


@pytest.mark.parametrize("missing", ["rgb_depth_registered", "depth_unit", "distortion"])
def test_missing_metric_prerequisites_do_not_become_implicit_assumptions(missing: str) -> None:
    metadata = camera()
    del metadata[missing]
    measured = measure_region(np.ones((80, 100, 3)), np.ones((80, 100)), metadata,
                              [45, 35, 55, 45])
    assert measured["roi_depth"] is None or (
        "surface_median_optical_xyz_m" not in measured["roi_depth"]
    )
    assert measured["projection"] is None


def test_body_and_optical_projection_agree_without_conflating_surface_with_center() -> None:
    rgb, depth = np.ones((80, 100, 3)), np.full((80, 100), 1.95)
    optical = {"frame_id": "camera", "detected_part_in_camera": {"position": [0.2, 0, 2]}}
    first = measure_region(rgb, depth, camera(), [55, 35, 65, 45], optical)
    body = {"frame_id": "camera", "detected_part_in_camera": {"position": [2, -0.2, 0]}}
    second = measure_region(rgb, depth, {**camera(), "pose_axes": "body_x_forward_y_left_z_up"},
                            [55, 35, 65, 45], body)
    assert first["projection"] == second["projection"]
    assert first["projection"]["detected_center_pixel_xy"] == [60, 40]
    assert first["projection"]["center_minus_surface_median_depth_m"] == pytest.approx(0.05)
    assert any("object geometry" in item for item in first["limitations"])
    with pytest.raises(ValueError, match="outside"):
        measure_region(rgb, depth, camera(), [-1, 0, 5, 5])


def test_alignment_distinguishes_wrong_capture_from_unsynchronized_clocks() -> None:
    capture = {"capture_id": "new", "frame_id": "cam", "clock_domain": "sim",
               "sensor_stamps_s": {"rgb": 9.0, "depth": 9.01}}
    detection = {"capture_id": "old", "frame_id": "cam", "clock_domain": "sim",
                 "source_stamp_s": 8.0, "part_id": "A", "detection_id": "d1"}
    command = {"capture_id": "old", "part_id": "B", "detection_id": "d1"}
    result = check_alignment(capture, detection, command)
    states = {row["name"]: row["status"] for row in result["identity_checks"]}
    assert states["capture_detection.capture_id"] == "mismatch"
    assert states["detection_command.part_id"] == "mismatch"
    assert states["detection_command.run_id"] == "unknown"
    assert result["detection_source_minus_capture_rgb_s"] == -1
    mixed = check_alignment(capture, {**detection, "clock_domain": "utc"}, command)
    assert not mixed["capture_detection_time_comparable"]
    assert mixed["detection_source_minus_capture_rgb_s"] is None
    assert mixed["depth_minus_rgb_stamp_s"] == pytest.approx(0.01)


def test_computed_region_evidence_survives_reload_and_remains_readable(tmp_path: Path) -> None:
    service = CaseService(tmp_path / "cases")
    case_id = service.create("目标区域深度缺失")
    record = service.record(case_id)
    Image.fromarray(np.full((80, 100, 3), 128, dtype=np.uint8)).save(tmp_path / "rgb.png")
    np.save(tmp_path / "depth.npy", np.ones((80, 100)))
    (tmp_path / "camera.json").write_text(json.dumps(camera()), encoding="utf-8")
    ids = []
    for name, media in (("rgb.png", "image/png"), ("depth.npy", "application/octet-stream"),
                        ("camera.json", "application/json")):
        ev = record.case.add_evidence(Evidence(
            evidence_id=record.case.next_evidence_id(), bundle_id="manual", reference=name,
            media_type=media, captured_at=datetime.now(UTC), clock_domain="host", summary=name,
        ))
        record.uploads[ev.evidence_id] = tmp_path / name
        ids.append(ev.evidence_id)
    box = Toolbox(record.case, None, None, uploads=record.uploads)
    result = box.measure_rgbd_region(rgb_evidence_id=ids[0], depth_evidence_id=ids[1],
                                    camera_evidence_id=ids[2], roi_xyxy=[45, 35, 55, 45])
    service.keep(case_id)
    restored = CaseService(service.root).record(case_id)
    read = Toolbox(restored.case, None, None).read_evidence([result["evidence_id"]])[0]
    data = json.loads(read["text"])
    assert data["sources"] == ids
    assert data["roi_depth"]["valid_ratio"] == 1


def test_shared_host_does_not_align_simulation_and_utc() -> None:
    from tests.test_case_walkthrough import _case

    case, _, _ = _case()
    bundle = case.observations[0]
    assert len(bundle.clock_domains) > 1 and bundle.clock.source
    assert not bundle.cross_source_comparable
