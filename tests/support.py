from __future__ import annotations

import difflib

from visiondoctor.demo.repository_builder import BASELINE_SOURCE, FAULTY_SOURCE

REGRESSION_TEST_SOURCE = """from __future__ import annotations

import unittest

import numpy as np

from pose_transformer import compose_target_pose


class TransformOrderRegressionTest(unittest.TestCase):
    def test_non_commuting_transforms_follow_target_source_order(self) -> None:
        angle = np.deg2rad(37.0)
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle), 0.0],
             [np.sin(angle), np.cos(angle), 0.0],
             [0.0, 0.0, 1.0]],
            dtype=float,
        )
        T_base_camera = np.eye(4)
        T_base_camera[:3, :3] = rotation
        T_base_camera[:3, 3] = [0.11, -0.04, 0.27]
        T_camera_object = np.eye(4)
        T_camera_object[:3, 3] = [0.07, 0.03, 0.65]

        actual = compose_target_pose(T_base_camera, T_camera_object)

        np.testing.assert_allclose(actual, T_base_camera @ T_camera_object, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
"""


def _file_patch(path: str, before: str | None, after: str) -> str:
    before_lines = [] if before is None else before.splitlines(keepends=True)
    after_lines = after.splitlines(keepends=True)
    header = [f"diff --git a/{path} b/{path}\n"]
    if before is None:
        header.append("new file mode 100644\n")
    body = difflib.unified_diff(
        before_lines,
        after_lines,
        fromfile="/dev/null" if before is None else f"a/{path}",
        tofile=f"b/{path}",
        lineterm="\n",
    )
    return "".join([*header, *body])


def root_cause_patch() -> str:
    return "".join(
        [
            _file_patch("src/pose_transformer.py", FAULTY_SOURCE, BASELINE_SOURCE),
            _file_patch("tests/test_transform_order_regression.py", None, REGRESSION_TEST_SOURCE),
        ]
    )
