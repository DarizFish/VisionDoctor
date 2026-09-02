"""The shared map a demarcation is written on.

The chain is not a fixed troubleshooting order and not the project's directory
tree.  It exists so that "where did this first go wrong" has addressable answers
that two people -- or a person and an agent -- can disagree about precisely.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, model_validator


class Segment(StrEnum):
    PART_SUPPLY = "part_supply"
    IMAGING = "imaging"
    ACQUISITION = "acquisition"
    ALGORITHM = "algorithm"
    TASK_RESULT = "task_result"
    CALIBRATION = "calibration"
    INTERFACE = "interface"
    ROBOT = "robot"


#: Every vision cell has these.
CORE_SEGMENTS: tuple[Segment, ...] = (
    Segment.PART_SUPPLY,
    Segment.IMAGING,
    Segment.ACQUISITION,
    Segment.ALGORITHM,
    Segment.TASK_RESULT,
)

#: These three arrive together, on one signal: the vision output is consumed by
#: an actuator in some coordinate frame.  No actuator, no hand-eye calibration,
#: no frame handover, no robot side.
GUIDED_SEGMENTS: tuple[Segment, ...] = (
    Segment.CALIBRATION,
    Segment.INTERFACE,
    Segment.ROBOT,
)


class SegmentStatus(StrEnum):
    UNTESTED = "untested"
    CLEARED = "cleared"
    SUSPECT = "suspect"


class SegmentFinding(BaseModel):
    """A verdict on one segment.

    Anything but UNTESTED has to name the evidence it rests on.  This is the
    single rule that keeps a demarcation from being an opinion.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    segment: Segment
    status: SegmentStatus
    note: str
    evidence_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _conclusions_cite_evidence(self) -> SegmentFinding:
        if self.status is not SegmentStatus.UNTESTED and not self.evidence_ids:
            raise ValueError(f"{self.status} on {self.segment} cites no evidence")
        return self


class Hypothesis(BaseModel):
    """An explanation still competing with others, and what it rests on."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hypothesis_id: str
    target_segment: Segment
    statement: str
    evidence_ids: tuple[str, ...] = ()


def chain_for(*, guided_motion: bool) -> tuple[Segment, ...]:
    """The segments drawn on the map for this case."""

    return CORE_SEGMENTS + (GUIDED_SEGMENTS if guided_motion else ())
