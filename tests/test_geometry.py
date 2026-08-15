from __future__ import annotations

import math

import numpy as np

from visiondoctor.geometry import (
    invert,
    make_transform,
    quaternion_xyzw_from_rotation,
    rotation_error_rad,
    rotation_from_quaternion_xyzw,
    translation_error_m,
)
from visiondoctor.geometry.transforms import rotation_from_euler


def test_inverse_and_composition_are_consistent() -> None:
    transform = make_transform(rotation_from_euler(0.2, -0.1, 0.4), np.array([0.1, -0.2, 0.7]))

    np.testing.assert_allclose(transform @ invert(transform), np.eye(4), atol=1e-12)


def test_translation_and_rotation_errors_use_meters_and_radians() -> None:
    reference = np.eye(4)
    predicted = make_transform(
        rotation_from_euler(0.0, 0.0, math.radians(1.0)), np.array([0.003, 0.004, 0.0])
    )

    assert translation_error_m(predicted, reference) == 0.005
    assert math.isclose(rotation_error_rad(predicted, reference), math.radians(1), abs_tol=1e-12)


def test_quaternion_rotation_round_trip_uses_xyzw() -> None:
    quaternion = np.array([-0.6993091168, -0.1086365055, 0.1719707493, 0.6852670503])

    rotation = rotation_from_quaternion_xyzw(quaternion)
    round_trip = quaternion_xyzw_from_rotation(rotation)

    np.testing.assert_allclose(round_trip, quaternion / np.linalg.norm(quaternion), atol=1e-10)
