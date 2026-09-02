"""Hard gates.  A gate refuses; it never repairs what it found missing."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from .case import Case
from .chain import SegmentStatus
from .repair import ApprovalRecord, RepairPlan


class GateResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    reasons: tuple[str, ...] = ()


def diagnosis_gate(case: Case) -> GateResult:
    """A demarcation may leave the case only if the evidence carries it.

    At least one segment implicated, at least one ruled out, and at least one
    explanation resting on evidence the case actually holds.
    """

    verdicts = case.demarcation()
    reasons: list[str] = []
    if SegmentStatus.SUSPECT not in verdicts.values():
        reasons.append("no segment is implicated")
    if SegmentStatus.CLEARED not in verdicts.values():
        reasons.append("no segment has been ruled out")
    if not any(item.evidence_ids for item in case.hypotheses):
        reasons.append("no hypothesis rests on evidence")
    return GateResult(passed=not reasons, reasons=tuple(reasons))


def approval_gate(plan: RepairPlan, approval: ApprovalRecord | None) -> GateResult:
    """Nothing is applied without a human decision about this exact plan."""

    if approval is None:
        return GateResult(passed=False, reasons=("no one has decided on this plan",))
    reasons: list[str] = []
    if approval.plan_id != plan.plan_id:
        reasons.append("the decision names a different plan")
    if approval.frozen_hash != plan.frozen_hash:
        reasons.append("the plan changed after it was reviewed")
    if not approval.approved:
        reasons.append(f"{approval.approver} declined the plan")
    return GateResult(passed=not reasons, reasons=tuple(reasons))
