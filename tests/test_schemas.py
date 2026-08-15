from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from visiondoctor.schemas import PoseTransform


def test_pose_transform_accepts_valid_target_source_matrix() -> None:
    pose = PoseTransform.from_array("base", "camera", np.eye(4))

    assert pose.target_frame == "base"
    assert pose.source_frame == "camera"
    assert pose.length_unit == "m"
    assert pose.angle_unit == "rad"
    assert pose.quaternion_order == "xyzw"


@pytest.mark.parametrize(
    "matrix",
    [
        np.diag([2.0, 1.0, 1.0, 1.0]),
        np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0, 0, 1, 1]]),
        np.array(
            [[1.0, 0.0, 0.0, np.nan], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0, 0, 0, 1]]
        ),
    ],
)
def test_pose_transform_rejects_invalid_rigid_matrices(matrix: np.ndarray) -> None:
    with pytest.raises(ValidationError):
        PoseTransform.from_array("base", "camera", matrix)


def test_pose_transform_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        PoseTransform.from_array("base", "camera", np.eye(3))
