from __future__ import annotations

import math
from typing import TypedDict


class Pose(TypedDict):
    position: list[float]
    quaternion_xyzw: list[float]


def normalise(quaternion: list[float]) -> list[float]:
    length = math.sqrt(sum(component * component for component in quaternion))
    if length <= 0:
        raise ValueError("zero quaternion")
    return [component / length for component in quaternion]


def multiply(left: list[float], right: list[float]) -> list[float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return [
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ]


def conjugate(quaternion: list[float]) -> list[float]:
    return [-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]]


def rotate(vector: list[float], quaternion: list[float]) -> list[float]:
    result = multiply(multiply(quaternion, [*vector, 0.0]), conjugate(quaternion))
    return result[:3]


def compose(left: Pose, right: Pose) -> Pose:
    left_orientation = normalise(left["quaternion_xyzw"])
    right_orientation = normalise(right["quaternion_xyzw"])
    offset = rotate(right["position"], left_orientation)
    return {
        "position": [left["position"][index] + offset[index] for index in range(3)],
        "quaternion_xyzw": normalise(multiply(left_orientation, right_orientation)),
    }


def inverse(transform: Pose) -> Pose:
    orientation = normalise(transform["quaternion_xyzw"])
    inverse_orientation = conjugate(orientation)
    return {
        "position": rotate([-value for value in transform["position"]], inverse_orientation),
        "quaternion_xyzw": inverse_orientation,
    }
