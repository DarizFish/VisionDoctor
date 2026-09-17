"""Hard gates.  A gate refuses; it never repairs what it found missing."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from .case import Case
from .chain import SegmentFinding, SegmentStatus
from .repair import ApprovalRecord, RepairPlan


class GateResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    reasons: tuple[str, ...] = ()


def software_localizations(case: Case) -> list[SegmentFinding]:
    """Suspects on software-carried targets that the running program's own records support.

    This is the software layer's demarcation: a module whose inputs, outputs,
    configuration or version depart from what they should be.  Pictures alone or
    source alone do not make one, and neither does a node the host's structural
    diagnosis has already exonerated for the run under diagnosis.
    """

    from .graph import consuming_node, is_software_target
    from .isolation import node_standing

    standing = node_standing(case)
    return [
        finding
        for finding in case.findings
        if finding.status is SegmentStatus.SUSPECT
        and finding.target_id
        and is_software_target(finding.target_id)
        and standing.get(consuming_node(finding.target_id) or "") != "exonerated"
        and any(case.layer_of(name) == "software" for name in finding.evidence_ids)
    ]


def source_layer_gate(case: Case) -> GateResult:
    """Source is read beneath a located software fault, and only at the version that ran."""

    reasons: list[str] = []
    binding = case.project
    if binding is None or not binding.source_readable:
        reasons.append("本案没有源码接入，诊断停在软件层")
    else:
        recorded = {
            str(item.project_revision.get("commit") or "") for item in case.observations
        } - {""}
        if not recorded:
            reasons.append("观察包没有记录运行版本，无法确认源码就是运行的那一份")
        elif not any(binding.revision.startswith(commit) or commit.startswith(binding.revision)
                     for commit in recorded):
            reasons.append("绑定的源码版本与观察包记录的运行版本不一致")
    if not software_localizations(case):
        reasons.append("软件层还没有用运行记录把异常定位到承载软件的节点")
    return GateResult(passed=not reasons, reasons=tuple(reasons))


def repair_gate(case: Case, hypothesis_id: str) -> GateResult:
    """A patch answers one source-patch hypothesis on a target the software layer located."""

    verdict = source_layer_gate(case)
    reasons = list(verdict.reasons)
    hypothesis = next(
        (item for item in case.hypotheses if item.hypothesis_id == hypothesis_id), None
    )
    located = {item.target_id for item in software_localizations(case)}
    if hypothesis is None:
        reasons.append(f"案件中没有已提交的假设 {hypothesis_id}")
    else:
        if hypothesis.remedy != "source_patch":
            reasons.append(f"{hypothesis_id} 的处理方式不是源码补丁")
        if hypothesis.target_id not in located:
            reasons.append(f"{hypothesis_id} 指向的对象没有软件层定位")
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
