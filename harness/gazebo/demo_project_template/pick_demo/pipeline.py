from __future__ import annotations

from .geometry import Pose, compose, inverse


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
    it repeats the reverse TCP compensation twice after applying it once.

    The top-down cell therefore reaches a stable pose roughly 70 mm above
    the part instead of driving the fault target into the low fixture.  The
    normal parent revision contains the one correct inverse transform.
    """

    compensated = compose(desired_tcp_in_base, inverse(tool0_to_tcp))
    overcompensated = compose(compensated, inverse(tool0_to_tcp))
    return compose(overcompensated, inverse(tool0_to_tcp))
