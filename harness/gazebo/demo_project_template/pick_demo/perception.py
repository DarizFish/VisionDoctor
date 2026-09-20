"""A deliberately bounded RGB-D detector for colored parts in an upright fixture.

Position is measured from visible top-surface points. Orientation comes from
the declared fixture recipe. This is not a general six-dimensional pose model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from .geometry import compose, inverse, normalise, rotate


def _largest_region(mask: np.ndarray) -> np.ndarray:
    """Keep one connected color component; robot highlights are separate regions."""
    pixels = set(zip(*np.nonzero(mask), strict=True))
    largest: set[tuple[int, int]] = set()
    while pixels:
        seed = pixels.pop()
        region, pending = {seed}, [seed]
        while pending:
            y, x = pending.pop()
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                point = (y + dy, x + dx)
                if point in pixels:
                    pixels.remove(point)
                    region.add(point)
                    pending.append(point)
        if len(region) > len(largest):
            largest = region
    selected = np.zeros_like(mask)
    for y, x in largest:
        selected[y, x] = True
    return selected


def detect(capture_root: Path, part_id: str, run_id: str, project_root: Path) -> dict:
    settings = yaml.safe_load((project_root / "config/perception.yaml").read_text(encoding="utf-8"))
    recipe = settings["parts"][part_id]
    calibration = yaml.safe_load(
        (project_root / "config/cell_calibration.yaml").read_text(encoding="utf-8")
    )["camera_to_base"]
    capture = json.loads((capture_root / "capture.json").read_text(encoding="utf-8"))
    camera = json.loads((capture_root / "camera-info.json").read_text(encoding="utf-8"))
    rgb = np.asarray(Image.open(capture_root / "rgb.png").convert("RGB"))
    depth = np.load(capture_root / "depth.npy", allow_pickle=False)
    result = {
        "schema_version": "pick-a17-vision/v2", "run_id": run_id, "part_id": part_id,
        "detection_id": f"{capture['capture_id']}-{part_id}",
        "capture_id": capture["capture_id"], "frame_id": camera["frame_id"],
        "source_stamp_s": capture["sensor_stamps_s"]["rgb"],
        "clock_domain": capture["clock_domain"],
        "method": "RGB connected color region, registered depth, upright known-shape fit",
        "pose_scope": "Position measured; orientation supplied by fixture grasp-frame recipe.",
        "source_hashes": {
            name: hashlib.sha256((capture_root / name).read_bytes()).hexdigest()
            for name in ("rgb.png", "depth.npy", "camera-info.json", "capture.json")
        },
    }
    required = {
        "rgb_depth_registered": True, "depth_unit": "m", "depth_measurement": "optical_z",
        "pose_axes": "body_x_forward_y_left_z_up",
    }
    if any(camera.get(key) != value for key, value in required.items()):
        return {**result, "status": "unsupported_camera_contract"}
    if depth.shape != rgb.shape[:2]:
        return {**result, "status": "rgb_depth_shape_mismatch"}
    mask = _largest_region(((rgb >= recipe["rgb_min"]) & (rgb <= recipe["rgb_max"])).all(axis=2))
    ys, xs = np.nonzero(mask)
    result["color_pixel_count"] = int(len(xs))
    if len(xs) < settings["minimum_color_pixels"]:
        return {**result, "status": "target_color_not_found"}
    result["roi_xyxy"] = [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]
    ds = depth[ys, xs]
    valid = np.isfinite(ds) & (ds > 0)
    result["target_depth_valid_ratio"] = float(valid.mean())
    if valid.mean() < settings["minimum_target_depth_ratio"]:
        return {**result, "status": "insufficient_target_depth"}
    k = np.asarray(camera["intrinsics"], dtype=float).reshape(3, 3)
    rays = np.linalg.solve(k, np.stack([xs[valid], ys[valid], np.ones(int(valid.sum()))]))
    optical_points = (rays * ds[valid]).T
    body_points = np.column_stack([
        optical_points[:, 2], -optical_points[:, 0], -optical_points[:, 1],
    ])
    rotation = normalise(calibration["quaternion_xyzw"])
    base_points = np.array([rotate(point.tolist(), rotation) for point in body_points])
    base_points += np.asarray(calibration["position"])
    top_z = float(np.percentile(base_points[:, 2], 99))
    top = base_points[np.abs(base_points[:, 2] - top_z) < settings["top_surface_band_m"]]
    result["top_surface_point_count"] = len(top)
    if len(top) < settings["minimum_top_surface_points"]:
        return {**result, "status": "top_surface_not_resolved"}
    # Visible top surface must span the complete part. The model records this
    # assumption; occlusion is a competing explanation, never silently cleared.
    low, high = np.percentile(top[:, :2], [1, 99], axis=0)
    xy = (low + high) / 2
    if recipe["shape"] == "cylinder":
        side = base_points[base_points[:, 2] < top_z - 2 * settings["top_surface_band_m"], :2]
        if len(side) < settings["minimum_color_pixels"]:
            return {**result, "status": "cylinder_side_not_resolved"}
        origin = side.mean(axis=0)
        centered = side - origin
        design = np.column_stack([2 * centered, np.ones(len(side))])
        fit, _, rank, _ = np.linalg.lstsq(design, (centered**2).sum(axis=1), rcond=None)
        if rank < 3 or fit[2] + (fit[:2]**2).sum() <= 0:
            return {**result, "status": "cylinder_fit_degenerate"}
        xy = fit[:2] + origin
        result["fitted_radius_m"] = float(np.sqrt(fit[2] + (fit[:2]**2).sum()))
    center_z = float(top_z - recipe["height_m"] / 2)
    part_base = {
        "position": [float(xy[0]), float(xy[1]), center_z],
        "quaternion_xyzw": settings["fixture_grasp_frame_quaternion_xyzw"],
    }
    return {
        **result, "status": "detected",
        "visible_top_span_xy_m": (high - low).tolist(),
        "detected_part_in_camera": compose(inverse(calibration), part_base),
        "assumptions": ["upright known-height part", "unoccluded top surface",
                        "declared camera extrinsics", "fixture-prescribed orientation"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--part", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = detect(args.capture, args.part, args.run_id, Path(__file__).resolve().parents[1])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
