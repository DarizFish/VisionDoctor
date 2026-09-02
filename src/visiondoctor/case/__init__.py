"""Case, diagnostic chain, and the gates a conclusion has to pass."""

from .case import Case, Evidence
from .chain import (
    CORE_SEGMENTS,
    GUIDED_SEGMENTS,
    SEGMENT_NAME,
    SEGMENT_SCOPE,
    Hypothesis,
    Segment,
    SegmentFinding,
    SegmentStatus,
    chain_for,
)
from .gates import GateResult, approval_gate, diagnosis_gate
from .repair import ApprovalRecord, RepairPlan
from .service import CaseRecord, CaseService

__all__ = [
    "CORE_SEGMENTS",
    "GUIDED_SEGMENTS",
    "SEGMENT_NAME",
    "SEGMENT_SCOPE",
    "ApprovalRecord",
    "Case",
    "CaseRecord",
    "CaseService",
    "Evidence",
    "GateResult",
    "Hypothesis",
    "RepairPlan",
    "Segment",
    "SegmentFinding",
    "SegmentStatus",
    "approval_gate",
    "chain_for",
    "diagnosis_gate",
]
