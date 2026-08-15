from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from visiondoctor.geometry import make_transform
from visiondoctor.schemas import PoseTransform


class DeterministicRgbdPoseEstimator:
    """Recover a colored orthogonal marker frame by RGB segmentation and depth backprojection."""

    def estimate(
        self,
        rgb_path: Path,
        depth_path: Path,
        camera_matrix: np.ndarray,
    ) -> PoseTransform:
        with Image.open(rgb_path) as image:
            rgb = np.asarray(image.convert("RGB"))
        depth = np.load(depth_path, allow_pickle=False)
        if depth.shape != rgb.shape[:2]:
            raise ValueError("RGB and depth dimensions do not match")
        camera_matrix = np.asarray(camera_matrix, dtype=float)
        if camera_matrix.shape != (3, 3):
            raise ValueError("camera matrix must have shape (3, 3)")

        origin = self._backproject(rgb, depth, camera_matrix, self._origin_mask(rgb), "origin")
        x_endpoint = self._backproject(
            rgb, depth, camera_matrix, self._x_axis_mask(rgb), "x-axis"
        )
        y_endpoint = self._backproject(
            rgb, depth, camera_matrix, self._y_axis_mask(rgb), "y-axis"
        )
        x_axis = self._normalize(x_endpoint - origin, "x-axis")
        y_residual = (y_endpoint - origin) - x_axis * np.dot(y_endpoint - origin, x_axis)
        y_axis = self._normalize(y_residual, "y-axis")
        z_axis = self._normalize(np.cross(x_axis, y_axis), "z-axis")
        y_axis = self._normalize(np.cross(z_axis, x_axis), "orthogonalized y-axis")
        rotation = np.column_stack((x_axis, y_axis, z_axis))
        return PoseTransform.from_array("camera", "object", make_transform(rotation, origin))

    @staticmethod
    def _backproject(
        rgb: np.ndarray,
        depth: np.ndarray,
        camera_matrix: np.ndarray,
        mask: np.ndarray,
        marker_name: str,
    ) -> np.ndarray:
        rows, columns = np.nonzero(mask)
        if len(rows) < 5:
            raise ValueError(f"RGB marker is missing: {marker_name}")
        values = depth[rows, columns]
        valid = np.isfinite(values) & (values > 0)
        if int(valid.sum()) < 5:
            raise ValueError(f"depth is missing at marker: {marker_name}")
        u = float(np.mean(columns[valid]))
        v = float(np.mean(rows[valid]))
        z = float(np.median(values[valid]))
        fx, fy = float(camera_matrix[0, 0]), float(camera_matrix[1, 1])
        cx, cy = float(camera_matrix[0, 2]), float(camera_matrix[1, 2])
        return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z], dtype=float)

    @staticmethod
    def _normalize(vector: np.ndarray, name: str) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-9:
            raise ValueError(f"degenerate marker geometry: {name}")
        return vector / norm

    @staticmethod
    def _origin_mask(rgb: np.ndarray) -> np.ndarray:
        return (rgb[..., 1] > 180) & (rgb[..., 0] < 120) & (rgb[..., 2] < 180)

    @staticmethod
    def _x_axis_mask(rgb: np.ndarray) -> np.ndarray:
        return (rgb[..., 0] > 180) & (rgb[..., 1] < 140) & (rgb[..., 2] < 140)

    @staticmethod
    def _y_axis_mask(rgb: np.ndarray) -> np.ndarray:
        return (rgb[..., 2] > 180) & (rgb[..., 0] < 140) & (rgb[..., 1] < 180)
