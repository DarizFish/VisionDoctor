from __future__ import annotations

import math

import numpy as np


def make_transform(rotation: np.ndarray, translation_m: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=float)
    translation_m = np.asarray(translation_m, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")
    if translation_m.shape != (3,):
        raise ValueError("translation must have shape (3,)")
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation_m
    return transform


def compose(t_target_middle: np.ndarray, t_middle_source: np.ndarray) -> np.ndarray:
    return np.asarray(t_target_middle, dtype=float) @ np.asarray(t_middle_source, dtype=float)


def invert(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=float)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    result = np.eye(4, dtype=float)
    result[:3, :3] = rotation.T
    result[:3, 3] = -(rotation.T @ translation)
    return result


def translation_error_m(predicted: np.ndarray, reference: np.ndarray) -> float:
    delta = np.asarray(predicted, dtype=float)[:3, 3] - np.asarray(reference, dtype=float)[:3, 3]
    return float(np.linalg.norm(delta))


def rotation_error_rad(predicted: np.ndarray, reference: np.ndarray) -> float:
    predicted_rotation = np.asarray(predicted, dtype=float)[:3, :3]
    reference_rotation = np.asarray(reference, dtype=float)[:3, :3]
    relative = reference_rotation.T @ predicted_rotation
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(math.acos(cosine))


def project_origin(t_camera_object: np.ndarray, camera_matrix: np.ndarray) -> tuple[float, float]:
    point = np.asarray(t_camera_object, dtype=float)[:3, 3]
    if point[2] <= 0:
        raise ValueError("object origin must be in front of the camera")
    pixel_homogeneous = np.asarray(camera_matrix, dtype=float) @ point
    return (
        float(pixel_homogeneous[0] / pixel_homogeneous[2]),
        float(pixel_homogeneous[1] / pixel_homogeneous[2]),
    )


def rotation_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def rotation_from_quaternion_xyzw(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=float)
    if quaternion.shape != (4,):
        raise ValueError("quaternion must have shape (4,)")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("quaternion norm must be non-zero")
    x, y, z, w = quaternion / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def quaternion_xyzw_from_rotation(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")
    trace = float(np.trace(rotation))
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2
        quaternion = np.array(
            [
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
            quaternion = np.array(
                [
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                ]
            )
        elif axis == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
            quaternion = np.array(
                [
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                ]
            )
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
            quaternion = np.array(
                [
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                ]
            )
    quaternion /= np.linalg.norm(quaternion)
    return quaternion if quaternion[3] >= 0 else -quaternion
