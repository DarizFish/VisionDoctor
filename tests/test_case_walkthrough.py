"""The demonstration path, walked once end to end.

The bundle under ``fixtures/pick-a17-faulty`` was produced by running the real
faulty demo project; its logged command composes the tool transform directly
where the TCP-to-flange convention requires the inverse.  That discrepancy is
visible from the bundle alone -- no fault ground truth is involved.
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

BUNDLE = Path(__file__).parent / "fixtures" / "pick-a17-faulty"


def test_a_failed_pick_is_demarcated_from_the_evidence_it_carries() -> None:
    adapter = FileBundleAdapter(BUNDLE)
    bundle = adapter.collect()
    assert not bundle.succeeded
    assert bundle.cross_source_comparable

    case = Case("CASE-A17", "A17 工位抓偏")
    case.extend_for_guided_motion()
    assert len(case.segments) == 8

    admitted = case.admit(bundle)
    assert len(admitted) == len(bundle.artifacts) + len(bundle.results)
    reference = {item.reference: item.evidence_id for item in admitted}
    detection = reference["parts/A/algorithm/input.json"]
    command = reference["parts/A/algorithm/output.json"]
    tool = reference["project/tool_profile.yaml"]

    assert adapter.read_artifact("parts/A/algorithm/output.json")

    case.record(
        SegmentFinding(
            segment=Segment.ALGORITHM,
            status=SegmentStatus.CLEARED,
            note="标定链路重算出的期望 TCP 与检测输入一致，算法段没有偏离",
            evidence_ids=(detection, command),
        )
    )
    case.record(
        SegmentFinding(
            segment=Segment.INTERFACE,
            status=SegmentStatus.SUSPECT,
            note="指令法兰位姿等于 TCP 直接复合工具变换；改用逆变换才自洽",
            evidence_ids=(command, tool),
        )
    )
    case.propose(
        Hypothesis(
            hypothesis_id="H1",
            target_segment=Segment.INTERFACE,
            statement="TCP 到法兰的工具补偿方向用反了",
            evidence_ids=(command, tool),
        )
    )

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
    substituted = plan.model_copy(update={"diff": "something else entirely"})
    assert not approval_gate(substituted, approval).passed


def test_a_conclusion_cannot_cite_evidence_the_case_never_took_in() -> None:
    case = Case("CASE-A17", "A17 工位抓偏")
    with pytest.raises(ValueError, match="holds no evidence"):
        case.record(
            SegmentFinding(
                segment=Segment.INTERFACE,
                status=SegmentStatus.SUSPECT,
                note="没有来源的结论",
                evidence_ids=("EV-999",),
            )
        )
