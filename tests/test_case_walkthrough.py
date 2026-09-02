"""The investigation path, walked once on a real observation.

The bundle under ``fixtures/pick-a17-observation`` is the text half of an actual
cell run: the logged command composes the tool transform forward where the
TCP-to-flange convention needs its inverse, and the chain check measures exactly
that.  Nothing here reads a fault ground truth.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from visiondoctor.case import (
    ApprovalRecord,
    Case,
    Hypothesis,
    RepairPlan,
    Segment,
    SegmentFinding,
    SegmentStatus,
    approval_gate,
    diagnosis_gate,
)
from visiondoctor.environment import FileBundleAdapter
from visiondoctor.investigation import Investigation, Toolbox, build_view

BUNDLE = Path(__file__).parent / "fixtures" / "pick-a17-observation"


def _case() -> tuple[Case, FileBundleAdapter, dict[str, str]]:
    adapter = FileBundleAdapter(BUNDLE)
    case = Case("CASE-A17", "A17 工位抓偏", guided_motion=True)
    admitted = case.admit(adapter.collect())
    return case, adapter, {item.reference: item.evidence_id for item in admitted}


def test_a_failed_pick_is_demarcated_from_what_the_host_delivered() -> None:
    case, adapter, reference = _case()
    bundle = case.observations[0]
    assert not bundle.succeeded
    assert len(case.segments) == 8
    assert len(case.evidence_ids) == len(bundle.artifacts) + len(bundle.results)

    view = build_view(case, bundle)
    assert len(view["evidence_catalogue"]) == len(case.evidence)
    assert all(entry["status"] == "untested" for entry in view["chain"])

    turn = Investigation(case, Toolbox(case, adapter, bundle), "机器人抓偏了，是哪一段的问题")
    delivered = turn.invoke(
        "read_evidence",
        evidence_ids=[
            reference["parts/A/algorithm/output.json"],
            reference["parts/B/algorithm/output.json"],
        ],
    )
    assert all("commanded_flange_base" in item["text"] for item in delivered)

    check = turn.invoke(
        "check_transform_chain",
        detection_evidence_id=reference["parts/A/algorithm/input.json"],
        calibration_evidence_id=reference["project/cell_calibration.yaml"],
        tool_evidence_id=reference["project/tool_profile.yaml"],
        command_evidence_id=reference["parts/A/algorithm/output.json"],
        motion_evidence_id=reference["parts/A/motion.json"],
    )
    # The command sits exactly on the forward composition and far from the inverse.
    assert check["residual_if_tool_forward"]["position_m"] == 0.0
    assert check["residual_if_tool_inverted"]["position_m"] > 0.015

    record = turn.commit(
        findings=(
            SegmentFinding(
                segment=Segment.ALGORITHM,
                status=SegmentStatus.CLEARED,
                note="重算的期望 TCP 与记录的检测输入一致",
                evidence_ids=(reference["parts/A/algorithm/input.json"], check["evidence_id"]),
            ),
            SegmentFinding(
                segment=Segment.INTERFACE,
                status=SegmentStatus.SUSPECT,
                note="指令位姿对正向复合零残差，对取逆复合差出任务容差",
                evidence_ids=(check["evidence_id"],),
            ),
        ),
        hypotheses=(
            Hypothesis(
                hypothesis_id="H1",
                target_segment=Segment.INTERFACE,
                statement="TCP 到法兰的工具补偿方向用反了",
                evidence_ids=(check["evidence_id"],),
            ),
        ),
        next_step="申请查看变换相关源码",
    )
    assert [call.name for call in record.calls] == ["read_evidence", "check_transform_chain"]
    assert diagnosis_gate(case).passed
    assert case.demarcation()[Segment.IMAGING] is SegmentStatus.UNTESTED

    plan = RepairPlan(
        plan_id="PLAN-1",
        case_id=case.case_id,
        target_segment=Segment.INTERFACE,
        hypothesis_id="H1",
        project_revision=bundle.project_revision["commit"],
        diff="compose(desired_tcp_in_base, inverse(tool0_to_tcp))",
    )
    assert not approval_gate(plan, None).passed
    approval = ApprovalRecord(
        plan_id=plan.plan_id,
        frozen_hash=plan.frozen_hash,
        approved=True,
        approver="operator",
        decided_at=bundle.created_at,
    )
    assert approval_gate(plan, approval).passed
    assert not approval_gate(plan.model_copy(update={"diff": "别的东西"}), approval).passed


def test_a_conclusion_cannot_cite_evidence_the_host_never_delivered() -> None:
    case, adapter, reference = _case()
    turn = Investigation(case, Toolbox(case, adapter, case.observations[0]), "先给个结论")
    with pytest.raises(ValueError, match="never delivered"):
        turn.commit(
            findings=(
                SegmentFinding(
                    segment=Segment.INTERFACE,
                    status=SegmentStatus.SUSPECT,
                    note="没有读过就下的结论",
                    evidence_ids=(reference["parts/B/motion.json"],),
                ),
            ),
            next_step="无",
        )
