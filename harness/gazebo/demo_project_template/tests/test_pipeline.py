from __future__ import annotations

from pick_demo.geometry import compose, inverse


def test_tool_pose_round_trip_is_not_an_identity_transform() -> None:
    tcp = {"position": [0.5, 0.1, 0.8], "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]}
    tool = {"position": [0.05, 0.0, 0.1], "quaternion_xyzw": [0.0, 0.0, 0.1, 0.995]}
    command = compose(tcp, tool)
    assert command["position"] != tcp["position"]
    assert compose(command, inverse(tool))["position"] == tcp
