"""Measurements with explicit scope: no fault labels and no global clearance."""

from __future__ import annotations

from typing import Any

import numpy as np


def depth_statistics(depth: np.ndarray) -> dict[str, Any]:
    if depth.ndim != 2 or not depth.size:
        raise ValueError("depth must be a nonempty two-dimensional array")
    values = depth[np.isfinite(depth) & (depth > 0)]
    return {
        "pixel_count": int(depth.size),
        "valid_ratio": round(float(values.size / depth.size), 6),
        "percentiles_raw": (
            dict(zip(("p05", "p50", "p95"), np.percentile(values, [5, 50, 95]).tolist(),
                     strict=True)) if values.size else None
        ),
    }


def measure_region(
    rgb: np.ndarray, depth: np.ndarray, camera: dict[str, Any],
    roi_xyxy: list[int], detection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure one caller-selected rectangle, with exclusive right/bottom edges.

    Depth validity alone needs no metric convention. Metric surface samples and
    projection are withheld unless their conventions are explicitly recorded.
    A rectangle is not a segmentation mask or an object pose measurement.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB must have three channels")
    if len(roi_xyxy) != 4 or any(type(value) is not int for value in roi_xyxy):
        raise ValueError("roi_xyxy must contain four integer pixel coordinates")
    x0, y0, x1, y1 = roi_xyxy
    height, width = rgb.shape[:2]
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("ROI is empty or outside the RGB image; it was not silently clipped")
    crop = rgb[y0:y1, x0:x1].astype(float)
    gray = crop @ np.array([0.2126, 0.7152, 0.0722])
    laplacian = (
        -4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1]
        + gray[1:-1, :-2] + gray[1:-1, 2:]
    )
    registered = camera.get("rgb_depth_registered") is True
    matching_shape = depth.shape == rgb.shape[:2]
    limitations = [
        "ROI may contain background; this is not an object segmentation.",
        "Exposure and sharpness statistics need a task-specific reference; no pass/fail threshold.",
        "Depth validity does not prove depth accuracy, calibration, or successful grasping.",
    ]
    result: dict[str, Any] = {
        "roi_xyxy": roi_xyxy,
        "roi_source": "caller-selected pixels",
        "image_shape": list(rgb.shape),
        "rgb": {
            "luma_p05_p50_p95": np.percentile(gray, [5, 50, 95]).round(3).tolist(),
            "dark_pixel_ratio_luma_le_5": round(float((gray <= 5).mean()), 6),
            "clipped_channel_pixel_ratio_ge_250": round(float((crop >= 250).any(axis=2).mean()), 6),
            "laplacian_variance": round(float(laplacian.var()), 3) if laplacian.size else None,
        },
        "full_depth": depth_statistics(depth),
        "rgb_depth_registered_declared": registered,
        "depth_shape_matches_rgb": matching_shape,
        "roi_depth": None,
        "projection": None,
        "limitations": limitations,
    }
    if not (registered and matching_shape):
        limitations.append("No registered RGB-D correspondence; depth ROI/projection withheld.")
        return result
    region_depth = depth[y0:y1, x0:x1]
    result["roi_depth"] = depth_statistics(region_depth)
    if camera.get("depth_unit") != "m" or camera.get("depth_measurement") != "optical_z":
        limitations.append("Metric projection requires meter units and optical-Z depth.")
        return result
    if camera.get("width") != width or camera.get("height") != height:
        limitations.append("Camera dimensions differ from the RGB image; projection withheld.")
        return result
    intrinsic = np.asarray(camera.get("intrinsics", []), dtype=float)
    if intrinsic.size != 9 or not np.isfinite(intrinsic).all():
        limitations.append("Missing or invalid camera intrinsics; projection withheld.")
        return result
    k = intrinsic.reshape(3, 3)
    if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]):
        limitations.append("Unsupported intrinsic matrix; projection withheld.")
        return result
    distortion = camera.get("distortion")
    if distortion is None or not np.allclose(distortion, 0):
        limitations.append("This pinhole measurement requires explicitly zero distortion.")
        return result
    valid = np.isfinite(region_depth) & (region_depth > 0)
    if not valid.any():
        limitations.append("No valid depth in the ROI; no metric surface estimate.")
        return result
    ys, xs = np.nonzero(valid)
    zs = region_depth[valid]
    rays = np.linalg.solve(k, np.stack([xs + x0, ys + y0, np.ones_like(xs)]))
    points = rays * zs
    result["roi_depth"]["surface_median_optical_xyz_m"] = np.median(points, axis=1).tolist()
    limitations.append("Surface samples are not the object's center or a six-dimensional pose.")
    if detection is None:
        return result
    if detection.get("frame_id") != camera.get("frame_id") or not camera.get("frame_id"):
        limitations.append("Detection/camera frames missing or different; projection withheld.")
        return result
    pose = detection.get("detected_part_in_camera", {})
    position = np.asarray(pose.get("position", []), dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all():
        limitations.append("No finite detected position to project.")
        return result
    convention = camera.get("pose_axes")
    if convention == "body_x_forward_y_left_z_up":
        position = np.array([-position[1], -position[2], position[0]])
    elif convention != "optical_x_right_y_down_z_forward":
        limitations.append("Unknown pose axis convention; detection projection withheld.")
        return result
    if position[2] <= 0:
        limitations.append("Declared detected position is behind the camera.")
        return result
    pixel = k @ position / position[2]
    u, v = pixel[:2].tolist()
    result["projection"] = {
        "detected_center_pixel_xy": [u, v],
        "inside_selected_roi": x0 <= u < x1 and y0 <= v < y1,
        "declared_center_optical_depth_m": float(position[2]),
        "center_minus_surface_median_depth_m": (
            float(position[2] - np.median(zs))
        ),
    }
    limitations.append("Center/surface depth gap includes object geometry, not just pose error.")
    return result


def check_alignment(
    capture: dict[str, Any], detection: dict[str, Any], command: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compare declared identities and clocks, without guessing synchronization."""
    checks: list[dict[str, Any]] = []

    def compare(name: str, left: Any, right: Any) -> None:
        checks.append({
            "name": name, "left": left, "right": right,
            "status": "unknown" if left is None or right is None else (
                "match" if left == right else "mismatch"
            ),
        })

    for key in ("capture_id", "frame_id"):
        compare(f"capture_detection.{key}", capture.get(key), detection.get(key))
    if command is not None:
        for key in ("run_id", "part_id", "detection_id", "capture_id"):
            compare(f"detection_command.{key}", detection.get(key), command.get(key))
    stamps = capture.get("sensor_stamps_s") or {}
    rgb_stamp, depth_stamp = stamps.get("rgb"), stamps.get("depth")
    sensor_delta = None
    if all(isinstance(s, (int, float)) and np.isfinite(s) for s in (rgb_stamp, depth_stamp)):
        sensor_delta = float(depth_stamp - rgb_stamp)
    capture_domain = capture.get("clock_domain")
    source_domain = detection.get("clock_domain")
    source_stamp = detection.get("source_stamp_s")
    source_delta = None
    comparable = bool(capture_domain) and capture_domain == source_domain
    if comparable and all(
        isinstance(s, (int, float)) and np.isfinite(s) for s in (source_stamp, rgb_stamp)
    ):
        source_delta = float(source_stamp - rgb_stamp)
    else:
        comparable = False
    return {
        "identity_checks": checks,
        "depth_minus_rgb_stamp_s": sensor_delta,
        "detection_source_minus_capture_rgb_s": source_delta,
        "capture_detection_time_comparable": comparable,
        "limitations": [
            "Checks compare recorded identities; equal labels do not prove the pixels used.",
            "Timestamps from different or unspecified clock domains are not subtracted.",
            "Source timestamp is acquisition time, not processing completion or command time.",
            "No age tolerance is assumed; acceptable skew depends on motion and task tolerance.",
        ],
    }
