"""Small rigid-pose helpers used by the harness-side private adjudicator."""

from __future__ import annotations

import math
from typing import Any

Pose = dict[str, list[float]]


def _quaternion_multiply(left: list[float], right: list[float]) -> list[float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return [
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ]


def _quaternion_conjugate(quaternion: list[float]) -> list[float]:
    return [-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]]


def _rotate(vector: list[float], quaternion: list[float]) -> list[float]:
    result = _quaternion_multiply(
        _quaternion_multiply(quaternion, [vector[0], vector[1], vector[2], 0.0]),
        _quaternion_conjugate(quaternion),
    )
    return result[:3]


def _normalise(quaternion: list[float]) -> list[float]:
    scale = math.sqrt(sum(value * value for value in quaternion))
    if scale <= 0:
        raise ValueError("quaternion must not be zero")
    return [value / scale for value in quaternion]


def compose(left: Pose, right: Pose) -> Pose:
    left_q = _normalise(list(left["quaternion_xyzw"]))
    right_q = _normalise(list(right["quaternion_xyzw"]))
    shifted = _rotate(list(right["position"]), left_q)
    return {
        "position": [left["position"][index] + shifted[index] for index in range(3)],
        "quaternion_xyzw": _normalise(_quaternion_multiply(left_q, right_q)),
    }


def inverse(pose: Pose) -> Pose:
    orientation = _normalise(list(pose["quaternion_xyzw"]))
    inverse_orientation = _quaternion_conjugate(orientation)
    rotated = _rotate([-value for value in pose["position"]], inverse_orientation)
    return {"position": rotated, "quaternion_xyzw": inverse_orientation}


def pose_error(expected: Pose, actual: Pose) -> tuple[float, float]:
    position_error = math.sqrt(
        sum(
            (float(expected["position"][index]) - float(actual["position"][index])) ** 2
            for index in range(3)
        )
    )
    expected_q = _normalise(list(expected["quaternion_xyzw"]))
    actual_q = _normalise(list(actual["quaternion_xyzw"]))
    dot = abs(sum(expected_q[index] * actual_q[index] for index in range(4)))
    return position_error, 2.0 * math.acos(min(1.0, max(-1.0, dot)))


def pose(value: dict[str, Any]) -> Pose:
    position = value.get("position")
    quaternion = value.get("quaternion_xyzw")
    if not isinstance(position, list) or not isinstance(quaternion, list):
        raise ValueError("pose must contain position and quaternion_xyzw lists")
    if len(position) != 3 or len(quaternion) != 4:
        raise ValueError("pose dimensions are invalid")
    return {
        "position": [float(item) for item in position],
        "quaternion_xyzw": _normalise([float(item) for item in quaternion]),
    }
