from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from visiondoctor.geometry import rotation_error_rad, translation_error_m
from visiondoctor.vision import DeterministicRgbdPoseEstimator
from visiondoctor.workflow import DemoRunResult


def test_rgbd_marker_pose_is_measured_not_copied_from_label(
    demo_result: DemoRunResult,
) -> None:
    estimator = DeterministicRgbdPoseEstimator()
    translation_errors: list[float] = []
    rotation_errors: list[float] = []
    for case in demo_result.incident.case_set:
        manifest_path = Path(case.manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        estimated = estimator.estimate(
            manifest_path.parent / manifest["rgb_path"],
            manifest_path.parent / manifest["depth_path"],
            np.asarray(manifest["camera_matrix"], dtype=float),
        ).as_array()
        reference = json.loads(Path(case.reference_path).read_text(encoding="utf-8"))
        t_base_camera = np.asarray(manifest["t_base_camera"]["matrix"], dtype=float)
        t_base_object = np.asarray(reference["reference_t_base_object"]["matrix"], dtype=float)
        label = np.linalg.inv(t_base_camera) @ t_base_object
        translation_errors.append(translation_error_m(estimated, label))
        rotation_errors.append(rotation_error_rad(estimated, label))

    assert max(translation_errors) <= 0.005
    assert max(rotation_errors) <= demo_result.incident.acceptance_criteria.mean_rotation_error_rad
    assert any(error > 1e-6 for error in translation_errors)


def test_rgbd_estimator_rejects_missing_origin_marker(
    demo_result: DemoRunResult, tmp_path: Path
) -> None:
    case = demo_result.incident.case_set[0]
    manifest_path = Path(case.manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_rgb = manifest_path.parent / manifest["rgb_path"]
    source_depth = manifest_path.parent / manifest["depth_path"]
    rgb_path = tmp_path / "rgb.png"
    depth_path = tmp_path / "depth.npy"
    shutil.copy2(source_depth, depth_path)
    with Image.open(source_rgb) as image:
        rgb = np.asarray(image.convert("RGB")).copy()
    green = (rgb[..., 1] > 180) & (rgb[..., 0] < 120) & (rgb[..., 2] < 180)
    rgb[green] = [25, 31, 42]
    Image.fromarray(rgb).save(rgb_path)

    with pytest.raises(ValueError, match="origin"):
        DeterministicRgbdPoseEstimator().estimate(
            rgb_path,
            depth_path,
            np.asarray(manifest["camera_matrix"], dtype=float),
        )
