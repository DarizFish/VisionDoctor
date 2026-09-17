from __future__ import annotations

from pick_demo.geometry import compose
from pick_demo.pipeline import flange_command_for_tcp


def test_flange_command_reconstructs_requested_tcp() -> None:
    tcp = {"position": [0.5, 0.1, 0.8], "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]}
    tool = {"position": [0.05, 0.0, 0.1], "quaternion_xyzw": [0.0, 0.0, 0.1, 0.995]}
    command = flange_command_for_tcp(tcp, tool)
    assert command["position"] != tcp["position"]
    recovered = compose(command, tool)
    errors = [abs(a - b) for a, b in zip(recovered["position"], tcp["position"], strict=True)]
    assert max(errors) < 1e-9
