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
    software_localizations,
    source_layer_gate,
)
from visiondoctor.environment import FileBundleAdapter
from visiondoctor.investigation import Investigation, Toolbox, build_view
from visiondoctor.repair import ProjectBinding, recheck, replay

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
    assert {Segment.PLANNING, Segment.GRASPING}.issubset(case.segments)
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
                target_id="tool_command",
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
    # The check is computed from what the program read and wrote: a software-layer
    # localization.  With no source bound, the diagnosis ends there.
    assert case.layer_of(check["evidence_id"]) == "software"
    assert [item.target_id for item in software_localizations(case)] == ["tool_command"]
    assert "没有源码接入" in source_layer_gate(case).reasons[0]
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


def _tiny_project(root: Path) -> str:
    """A two-commit project whose replay entry prints a number the patch changes."""

    import subprocess

    root.mkdir(parents=True)
    run = lambda *args: subprocess.run(  # noqa: E731 - local shorthand
        ["git", "-C", str(root), *args], check=True, capture_output=True
    )
    run("init", "--initial-branch=main")
    run("config", "user.email", "cell@example.invalid")
    run("config", "user.name", "Cell")
    (root / "compute.py").write_text(
        'import json, sys\n'
        'value = {"offset": 1}\n'
        'json.dump(value, open(sys.argv[2], "w"))\n',
        encoding="utf-8",
    )
    run("add", "-A")
    run("commit", "-m", "baseline")
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def test_a_candidate_is_replayed_in_isolation_on_recorded_input(tmp_path: Path) -> None:
    import sys

    repository = tmp_path / "project"
    revision = _tiny_project(repository)
    binding = ProjectBinding(
        repository=repository,
        revision=revision,
        replay_command=(sys.executable, "compute.py", "{input}", "{output}"),
    )
    patch = (
        "diff --git a/compute.py b/compute.py\n"
        "--- a/compute.py\n"
        "+++ b/compute.py\n"
        "@@ -1,3 +1,3 @@\n"
        " import json, sys\n"
        '-value = {"offset": 1}\n'
        '+value = {"offset": 2}\n'
        ' json.dump(value, open(sys.argv[2], "w"))\n'
    )
    outcome = replay(
        binding=binding,
        patch_text=patch,
        candidate_id="CAND-1",
        sandbox_root=tmp_path / "sandbox",
        inputs={"A": b"{}"},
    )
    assert outcome.changed_files == ("compute.py",)
    assert outcome.replays[0]["output"] == {"offset": 2}
    # The worktree is gone; the bound repository never saw the candidate.
    assert not (repository / ".replay").exists()
    assert (repository / "compute.py").read_text(encoding="utf-8").count('"offset": 1') == 1


def test_only_evidence_collected_after_the_change_can_say_the_site_recovered() -> None:
    from datetime import timedelta

    before = FileBundleAdapter(BUNDLE).collect()
    passing = before.model_copy(
        update={
            "run_id": "run-after",
            "created_at": before.created_at + timedelta(hours=1),
            "observation_started_at": before.created_at + timedelta(minutes=45),
            "results": tuple(
                item.model_copy(update={"success": True, "classification": "within_tolerance"})
                for item in before.results
            ),
        }
    )
    applied = before.created_at + timedelta(minutes=30)
    assert recheck(before=before, after=passing, applied_at=None).scope == "not_a_recheck"
    assert recheck(before=before, after=passing, applied_at=applied).scope == "site_recovered"
    partial = passing.model_copy(update={"results": passing.results[:1]})
    assert recheck(before=before, after=partial, applied_at=applied).scope == "not_a_recheck"
    late_export = passing.model_copy(update={"observation_started_at": before.created_at})
    assert recheck(before=before, after=late_export, applied_at=applied).scope == "not_a_recheck"
    still_bad = before.model_copy(
        update={"run_id": "run-again", "created_at": before.created_at + timedelta(hours=1),
                "observation_started_at": before.created_at + timedelta(minutes=45)}
    )
    assert recheck(before=before, after=still_bad, applied_at=applied).scope == "site_still_failing"
