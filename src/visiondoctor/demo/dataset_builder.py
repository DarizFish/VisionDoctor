from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from visiondoctor.geometry.transforms import (
    make_transform,
    project_origin,
    rotation_from_euler,
    rotation_from_quaternion_xyzw,
)
from visiondoctor.schemas import PoseTransform, TestCaseRef

FIXED_SEED = 20260805
IMAGE_SIZE = (640, 480)
MARKER_AXIS_LENGTH_M = 0.08
MAIN_SCENE_COMPENSATION_M = np.array(
    [-0.00460521223, 0.0296444265, 0.0], dtype=float
)
MAIN_SCENE_HALF_ROTATION_XYZW = np.array(
    [-0.38090777, -0.05917339, 0.09367101, 0.91795072], dtype=float
)
MAIN_SCENE_BASE_CAMERA_M = np.array(
    [0.02589095, -0.08451777, 0.51145229], dtype=float
)
MAIN_SCENE_CAMERA_OBJECT_M = np.array(
    [0.03629244, -0.09808184, 0.54518074], dtype=float
)


def _pose(target: str, source: str, matrix: np.ndarray) -> dict[str, object]:
    return PoseTransform.from_array(target, source, matrix).model_dump(mode="json")


def _project(point: np.ndarray, camera_matrix: np.ndarray) -> tuple[float, float]:
    pixel = camera_matrix @ point
    return float(pixel[0] / pixel[2]), float(pixel[1] / pixel[2])


def _marker_mask(shape: tuple[int, int], pixel: tuple[float, float], radius: int) -> np.ndarray:
    u, v = (int(round(value)) for value in pixel)
    rows, columns = np.ogrid[: shape[0], : shape[1]]
    return (columns - u) ** 2 + (rows - v) ** 2 <= radius**2


def build_demo_dataset(root: Path, *, scene_count: int = 50) -> tuple[TestCaseRef, ...]:
    if scene_count < 50:
        raise ValueError("the frozen MVP requires at least 50 scenes")
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(FIXED_SEED)
    camera_matrix = np.array(
        [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]
    )
    captured_base = datetime(2026, 8, 5, 9, 0, tzinfo=UTC)
    refs: list[TestCaseRef] = []

    for index in range(scene_count):
        case_id = "scene-main" if index == 0 else f"scene-{index:03d}"
        case_dir = root / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        if index == 0:
            half_rotation = rotation_from_quaternion_xyzw(MAIN_SCENE_HALF_ROTATION_XYZW)
            t_base_camera = make_transform(half_rotation, MAIN_SCENE_BASE_CAMERA_M)
            t_camera_object = make_transform(half_rotation, MAIN_SCENE_CAMERA_OBJECT_M)
        else:
            base_rotation = rotation_from_euler(
                float(rng.uniform(-0.25, 0.25)),
                float(rng.uniform(-0.25, 0.25)),
                float(rng.uniform(-1.0, 1.0)),
            )
            object_rotation = rotation_from_euler(
                float(rng.uniform(-0.45, 0.45)),
                float(rng.uniform(-0.45, 0.45)),
                float(rng.uniform(-1.2, 1.2)),
            )
            t_base_camera = make_transform(
                base_rotation,
                np.array(
                    [
                        rng.uniform(-0.12, 0.12),
                        rng.uniform(-0.12, 0.12),
                        rng.uniform(0.15, 0.35),
                    ]
                ),
            )
            t_camera_object = make_transform(
                object_rotation,
                np.array(
                    [
                        rng.uniform(-0.16, 0.16),
                        rng.uniform(-0.11, 0.11),
                        rng.uniform(0.55, 0.9),
                    ]
                ),
            )
        reference = t_base_camera @ t_camera_object
        expected_pixel = project_origin(t_camera_object, camera_matrix)

        rgb = Image.new("RGB", IMAGE_SIZE, color=(25, 31, 42))
        draw = ImageDraw.Draw(rgb)
        origin = t_camera_object[:3, 3]
        x_endpoint = origin + t_camera_object[:3, 0] * MARKER_AXIS_LENGTH_M
        y_endpoint = origin + t_camera_object[:3, 1] * MARKER_AXIS_LENGTH_M
        marker_points = (
            ("x", _project(x_endpoint, camera_matrix), x_endpoint, (235, 80, 80)),
            ("y", _project(y_endpoint, camera_matrix), y_endpoint, (80, 130, 240)),
            ("origin", expected_pixel, origin, (60, 220, 130)),
        )
        for name, (u, v), _point, _color in marker_points:
            if not (6 <= u < IMAGE_SIZE[0] - 6 and 6 <= v < IMAGE_SIZE[1] - 6):
                raise ValueError(f"{case_id} {name} marker projects outside the image")
        radius = 4
        for _name, (u, v), _point, color in marker_points:
            draw.ellipse((u - radius, v - radius, u + radius, v + radius), fill=color)
        rgb_path = case_dir / "rgb.png"
        rgb.save(rgb_path)

        depth = np.full(
            (IMAGE_SIZE[1], IMAGE_SIZE[0]), float(t_camera_object[2, 3]), dtype=np.float32
        )
        invalid_count = 300 + index % 17
        invalid_indices = rng.choice(depth.size, size=invalid_count, replace=False)
        depth.flat[invalid_indices] = 0.0
        for _name, pixel, point, _color in marker_points:
            depth[_marker_mask(depth.shape, pixel, radius)] = float(point[2])
        depth_path = case_dir / "depth.npy"
        np.save(depth_path, depth, allow_pickle=False)

        manifest_path = case_dir / "evidence_manifest.json"
        reference_path = case_dir / "qa_reference.json"
        manifest = {
            "schema_version": 1,
            "case_id": case_id,
            "seed": FIXED_SEED + index,
            "captured_at": (captured_base + timedelta(seconds=index)).isoformat(),
            "rgb_path": rgb_path.name,
            "depth_path": depth_path.name,
            "camera_matrix": camera_matrix.tolist(),
            "expected_pixel": list(expected_pixel),
            "marker_axis_length_m": MARKER_AXIS_LENGTH_M,
            "t_base_camera": _pose("base", "camera", t_base_camera),
            "source": "dataset",
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        reference_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "case_id": case_id,
                    "captured_at": manifest["captured_at"],
                    "reference_t_base_object": _pose("base", "object", reference),
                    "source_type": "dataset_label",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        refs.append(
            TestCaseRef(
                case_id=case_id,
                manifest_path=str(manifest_path.resolve()),
                reference_path=str(reference_path.resolve()),
            )
        )

    index_path = root / "dataset_index.json"
    index_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "seed": FIXED_SEED,
                "scene_count": scene_count,
                "cases": [ref.model_dump(mode="json") for ref in refs],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return tuple(refs)
