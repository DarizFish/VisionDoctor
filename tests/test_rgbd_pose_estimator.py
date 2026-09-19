"""The RGB-D marker pose is measured from pixels and depth, never declared."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from visiondoctor.geometry import make_transform, rotation_error_rad, translation_error_m
from visiondoctor.vision import DeterministicRgbdPoseEstimator

CAMERA_MATRIX = np.array(
    [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]], dtype=float
)
BACKGROUND = (25, 31, 42)
ORIGIN_RGB = (0, 255, 0)
X_AXIS_RGB = (255, 0, 0)
Y_AXIS_RGB = (0, 0, 255)


def _marker_frame(angle_rad: float) -> np.ndarray:
    """An orthonormal marker frame rotated about the camera z axis."""

    cos, sin = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    rotation = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return make_transform(rotation, np.array([0.02, -0.03, 0.60], dtype=float))


def _paint(
    rgb: np.ndarray, depth: np.ndarray, point: np.ndarray, color: tuple[int, int, int]
) -> None:
    pixel = CAMERA_MATRIX @ point
    u, v = int(round(pixel[0] / pixel[2])), int(round(pixel[1] / pixel[2]))
    rows, columns = np.ogrid[: rgb.shape[0], : rgb.shape[1]]
    disk = (columns - u) ** 2 + (rows - v) ** 2 <= 9
    rgb[disk] = color
    depth[disk] = point[2]


def _scene(root: Path, pose: np.ndarray, *, omit_origin: bool = False) -> tuple[Path, Path]:
    rgb = np.full((480, 640, 3), BACKGROUND, dtype=np.uint8)
    depth = np.zeros((480, 640), dtype=float)
    origin = pose[:3, 3]
    if not omit_origin:
        _paint(rgb, depth, origin, ORIGIN_RGB)
    _paint(rgb, depth, origin + pose[:3, 0] * 0.08, X_AXIS_RGB)
    _paint(rgb, depth, origin + pose[:3, 1] * 0.08, Y_AXIS_RGB)
    rgb_path, depth_path = root / "rgb.png", root / "depth.npy"
    Image.fromarray(rgb).save(rgb_path)
    np.save(depth_path, depth)
    return rgb_path, depth_path


def test_marker_pose_is_recovered_from_pixels_and_depth(tmp_path: Path) -> None:
    expected = _marker_frame(np.deg2rad(30.0))
    rgb_path, depth_path = _scene(tmp_path, expected)

    measured = DeterministicRgbdPoseEstimator().estimate(
        rgb_path, depth_path, CAMERA_MATRIX
    ).as_array()

    assert translation_error_m(measured, expected) <= 0.002
    assert rotation_error_rad(measured, expected) <= 0.02


def test_estimator_fails_closed_when_the_origin_marker_is_absent(tmp_path: Path) -> None:
    rgb_path, depth_path = _scene(tmp_path, _marker_frame(0.0), omit_origin=True)

    with pytest.raises(ValueError, match="origin"):
        DeterministicRgbdPoseEstimator().estimate(rgb_path, depth_path, CAMERA_MATRIX)
