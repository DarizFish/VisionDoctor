from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class GraspClassification(StrEnum):
    SUCCESS = "within_tolerance"
    FAILURE = "missed_pick_pose"
    EXECUTION_ERROR = "motion_execution_error"


@dataclass(frozen=True)
class GraspAssessment:
    classification: GraspClassification
    position_error_m: float
    rotation_error_rad: float

    @property
    def success(self) -> bool:
        return self.classification is GraspClassification.SUCCESS

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_grasp(
    position_error_m: float,
    rotation_error_rad: float,
    *,
    position_tolerance_m: float,
    rotation_tolerance_rad: float,
    motion_completed: bool,
) -> GraspAssessment:
    """Classify a completed TCP placement without exposing its private target."""

    if not motion_completed:
        classification = GraspClassification.EXECUTION_ERROR
    elif position_error_m <= position_tolerance_m and rotation_error_rad <= rotation_tolerance_rad:
        classification = GraspClassification.SUCCESS
    else:
        classification = GraspClassification.FAILURE
    return GraspAssessment(
        classification=classification,
        position_error_m=round(float(position_error_m), 6),
        rotation_error_rad=round(float(rotation_error_rad), 6),
    )
