from __future__ import annotations

from .geometry import Pose, compose


def derive_desired_tcp(
    detected_part_in_camera: Pose,
    camera_to_base: Pose,
    pick_offset_from_part: Pose,
) -> Pose:
    """Turn a camera-frame detection into the desired TCP pick pose."""

    part_in_base = compose(camera_to_base, detected_part_in_camera)
    return compose(part_in_base, pick_offset_from_part)


def flange_command_for_tcp(desired_tcp_in_base: Pose, tool0_to_tcp: Pose) -> Pose:
    """Return the tool0 command used by the robot controller.

    The current revision is deliberately the regression under demonstration:
    it applies the tool compensation in the wrong direction.
    """

    return compose(desired_tcp_in_base, tool0_to_tcp)
