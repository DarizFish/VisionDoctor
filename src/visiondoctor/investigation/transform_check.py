"""Recompute the declared transform chain and report what it does not explain.

This is a meter, not a diagnosis.  It says how far the logged command sits from
the two conventions the tool transform could have been used under; it does not
name a segment, a fault family or a repair.  Reading the numbers is the model's
job, and so is deciding whether a reversed convention means defective code, a
config written backwards, or calibration that has drifted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from visiondoctor.geometry.transforms import (
    compose,
    invert,
    make_transform,
    quaternion_xyzw_from_rotation,
    rotation_error_rad,
    rotation_from_quaternion_xyzw,
    translation_error_m,
)

Pose = dict[str, Any]


def _matrix(pose: Pose) -> np.ndarray:
    rotation = rotation_from_quaternion_xyzw(np.asarray(pose["quaternion_xyzw"], dtype=float))
    return make_transform(rotation, np.asarray(pose["position"], dtype=float))


def _pose(matrix: np.ndarray) -> Pose:
    return {
        "position": [round(float(value), 9) for value in matrix[:3, 3]],
        "quaternion_xyzw": [
            round(float(value), 9) for value in quaternion_xyzw_from_rotation(matrix[:3, :3])
        ],
    }


@dataclass(frozen=True)
class Residual:
    position_m: float
    rotation_rad: float

    def as_dict(self) -> dict[str, float]:
        return {"position_m": self.position_m, "rotation_rad": self.rotation_rad}


def _residual(predicted: np.ndarray, reference: np.ndarray) -> Residual:
    return Residual(
        position_m=round(translation_error_m(predicted, reference), 9),
        rotation_rad=round(rotation_error_rad(predicted, reference), 9),
    )


@dataclass(frozen=True)
class DeclaredMatch:
    """How closely the observed error resembles one declared transform.

    A transform used one time too many leaves an error equal to that transform
    itself; used two times too many, equal to it composed with itself.  The same
    holds for its inverse, so both directions are reported.  Four comparisons per
    declared transform, with no opinion about which row matters.
    """

    name: str
    applied_once: Residual
    applied_twice: Residual
    inverted_once: Residual
    inverted_twice: Residual

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "error_vs_transform": self.applied_once.as_dict(),
            "error_vs_transform_twice": self.applied_twice.as_dict(),
            "error_vs_inverse": self.inverted_once.as_dict(),
            "error_vs_inverse_twice": self.inverted_twice.as_dict(),
        }


@dataclass(frozen=True)
class TransformCheck:
    recomputed_tcp: Pose
    commanded_flange: Pose
    if_tool_forward: Residual
    if_tool_inverted: Residual
    measured_tcp: Pose | None
    #: Where the TCP actually landed relative to where the chain says it should,
    #: as a vector in the robot base frame.  Only the components are reported;
    #: what a changing direction across grasps would mean is not decided here.
    error_in_base: list[float] | None
    #: The gap between what the chain intended and what the command asks for,
    #: expressed in each frame the chain passes through.  An error that is
    #: constant in one frame and not another says where it is anchored.  This is
    #: the software's own arithmetic; the robot's tracking is not in it.
    error_by_frame: dict[str, Pose]
    #: That same gap measured against every transform the system declares.
    declared_matches: tuple[DeclaredMatch, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "recomputed_tcp": self.recomputed_tcp,
            "commanded_flange": self.commanded_flange,
            "residual_if_tool_forward": self.if_tool_forward.as_dict(),
            "residual_if_tool_inverted": self.if_tool_inverted.as_dict(),
            "measured_tcp": self.measured_tcp,
            "error_in_base": self.error_in_base,
            "error_by_frame": self.error_by_frame,
            "declared_matches": [item.as_dict() for item in self.declared_matches],
        }


def check_transform_chain(
    *,
    detection: Pose,
    camera_to_base: Pose,
    pick_offset: Pose,
    tool0_to_tcp: Pose,
    commanded_flange: Pose,
    measured_flange: Pose | None = None,
) -> TransformCheck:
    """Compare the logged flange command against both uses of the tool transform.

    Both compositions are reported the same way: neither name says which one the
    convention calls for.  Whichever residual is zero is the one the running code
    applied; what that means is not for this function to say.  ``error_in_base``
    reports where the TCP landed relative to the chain's own answer, as a vector.
    """

    declared = {
        "camera_to_base": _matrix(camera_to_base),
        "pick_offset_from_part": _matrix(pick_offset),
        "tool0_to_tcp": _matrix(tool0_to_tcp),
    }
    tool = declared["tool0_to_tcp"]
    part_in_base = compose(declared["camera_to_base"], _matrix(detection))
    tcp = compose(part_in_base, declared["pick_offset_from_part"])
    command = _matrix(commanded_flange)
    measured = compose(_matrix(measured_flange), tool) if measured_flange else None
    landed = measured if measured is not None else compose(command, tool)
    #: What the command implies the TCP will be, against what the chain intended.
    error = compose(invert(tcp), compose(command, tool))
    return TransformCheck(
        recomputed_tcp=_pose(tcp),
        commanded_flange=_pose(command),
        if_tool_forward=_residual(command, compose(tcp, tool)),
        if_tool_inverted=_residual(command, compose(tcp, invert(tool))),
        measured_tcp=_pose(measured) if measured is not None else None,
        error_in_base=(
            [round(float(value), 9) for value in landed[:3, 3] - tcp[:3, 3]]
            if measured is not None
            else None
        ),
        error_by_frame={
            "robot_base": _pose(compose(compose(tcp, error), invert(tcp))),
            "desired_tcp": _pose(error),
            "camera": _pose(
                compose(
                    compose(invert(declared["camera_to_base"]), compose(tcp, error)),
                    invert(compose(invert(declared["camera_to_base"]), tcp)),
                )
            ),
        },
        declared_matches=tuple(
            DeclaredMatch(
                name=name,
                applied_once=_residual(error, transform),
                applied_twice=_residual(error, compose(transform, transform)),
                inverted_once=_residual(error, invert(transform)),
                inverted_twice=_residual(
                    error, compose(invert(transform), invert(transform))
                ),
            )
            for name, transform in declared.items()
        ),
    )
