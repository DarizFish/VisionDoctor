"""Behavioral checks for graph reasoning, not a test of model accuracy."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.support import ProtocolDoubleGateway, _turn
from tests.test_case_walkthrough import _case, _tiny_project
from visiondoctor.case import CaseService, Hypothesis, Segment, SegmentFinding, SegmentStatus
from visiondoctor.case.graph import EDGES, NODES, graph_view
from visiondoctor.investigation import Investigation, Toolbox, investigate, tools_for
from visiondoctor.knowledge import catalogue, read_knowledge
from visiondoctor.llm import AssistantTurn
from visiondoctor.repair import ProjectBinding
from visiondoctor.web.graph import graph_dot


def _answer(payload: dict) -> AssistantTurn:
    content = json.dumps(payload, ensure_ascii=False)
    return AssistantTurn(
        content=content, tool_calls=(), finish_reason="stop",
        raw_message={"role": "assistant", "content": content},
    )


def test_knowledge_and_dependencies_do_not_count_as_observed_evidence() -> None:
    case, adapter, _ = _case()
    turn = Investigation(case, Toolbox(case, adapter, case.observations[0]), "为什么抓不住")
    guide = turn.invoke("read_domain_knowledge", knowledge_ids=["planning", "contact"])
    edge = turn.invoke("inspect_grasp_graph", target_id="motion_contact")
    assert guide["kind"] == edge["kind"] == "domain_guidance"
    assert not case.examined
    assert all(not call.delivered for call in turn.calls)
    with pytest.raises(ValueError, match="holds no evidence"):
        case.record(SegmentFinding(
            segment=Segment.GRASPING, target_id="motion_contact", status=SegmentStatus.CLEARED,
            note="读过知识所以没有接触故障", evidence_ids=("contact",),
        ))
    domains = {item["id"] for item in catalogue()}
    assert domains == {
        "task", "optics", "timing", "perception", "geometry", "planning",
        "execution", "contact", "software", "measurement",
    }
    for target in (*NODES, *EDGES):
        assert set(target.knowledge_ids) <= domains
    for edge in EDGES:
        assert {edge.source, edge.target} <= {node.id for node in NODES}
    with pytest.raises(ValueError, match="unknown knowledge"):
        read_knowledge(["the_fault_answer"])


def test_partial_check_does_not_clear_neighbours_and_can_be_withdrawn() -> None:
    case, adapter, references = _case()
    evidence = references["parts/A/motion.json"]
    toolbox = Toolbox(case, adapter, case.observations[0])
    turn = Investigation(case, toolbox, "检查执行与接触")
    turn.invoke("read_evidence", evidence_ids=[evidence])
    old = turn.commit(findings=(SegmentFinding(
        segment=Segment.ROBOT, target_id="command_motion", status=SegmentStatus.CLEARED,
        note="该次记录中的法兰跟踪差较小", checked_scope="A 的最终法兰位置",
        limitations="没有夹持证据，也未检查整个轨迹", evidence_ids=(evidence,),
    ),), next_step="检查工件是否被夹住")
    graph = graph_view(case.findings)
    states = {item["id"]: item["status"] for item in [*graph["nodes"], *graph["edges"]]}
    assert states["command_motion"] == "cleared"
    assert states["robot"] == states["motion_contact"] == states["contact"] == "untested"
    second = Investigation(case, toolbox, "发现采样没有对应同一条命令")
    second.commit(findings=(old.findings[0].model_copy(update={
        "status": SegmentStatus.UNTESTED,
        "note": "命令对应关系尚未确认，撤回先前排除",
    }),), next_step="核对命令 ID")
    assert case.demarcation()[Segment.ROBOT] is SegmentStatus.UNTESTED
    assert old.findings[0].status is SegmentStatus.CLEARED


def _bound_case(tmp_path: Path, *, source_readable: bool = True):
    import sys

    fixture, adapter, references = _case()
    repository = tmp_path / "project"
    revision = _tiny_project(repository)
    bundle = adapter.collect().model_copy(update={"project_revision": {"commit": revision}})
    case = type(fixture)(fixture.case_id, fixture.title, guided_motion=True)
    case.admit(bundle)
    case.bind_project(ProjectBinding(
        repository, revision, (sys.executable, "compute.py", "{input}", "{output}"),
        source_readable=source_readable,
    ))
    return case, adapter, bundle, references


def _locate_command_fault(case, adapter, bundle, references, remedy: str) -> str:
    output = references["parts/A/algorithm/output.json"]
    turn = Investigation(case, Toolbox(case, adapter, bundle), "指令位置偏了")
    turn.invoke("read_evidence", evidence_ids=[output])
    turn.commit(
        findings=(SegmentFinding(
            segment=Segment.INTERFACE, target_id="tool_command", status=SegmentStatus.SUSPECT,
            note="记录的法兰指令偏离期望", evidence_ids=(output,),
        ),),
        hypotheses=(Hypothesis(
            hypothesis_id="H1", target_segment=Segment.INTERFACE, target_id="tool_command",
            statement="指令计算多加了一次工具补偿", evidence_ids=(output,), remedy=remedy,
        ),),
        next_step="交给程序负责人或进入源码层",
    )
    return output


def test_source_layer_opens_only_beneath_a_located_software_fault(tmp_path: Path) -> None:
    case, adapter, bundle, references = _bound_case(tmp_path)
    toolbox = Toolbox(case, adapter, bundle, sandbox_root=tmp_path / "sandbox")
    names = {item["function"]["name"] for item in tools_for(case)}
    assert "replay_running_version" in names
    assert not {"list_source", "read_source", "propose_repair"} & names
    with pytest.raises(PermissionError, match="源码层未开放"):
        toolbox.read_source("compute.py", "H1")

    output = _locate_command_fault(case, adapter, bundle, references, "handoff")
    names = {item["function"]["name"] for item in tools_for(case)}
    assert {"list_source", "read_source", "propose_repair"} <= names
    with pytest.raises(PermissionError, match="已定位"):
        toolbox.read_source("compute.py", "H9")
    read = toolbox.read_source("compute.py", "H1")
    assert read["layer"] == "source" and '"offset": 1' in read["text"]
    with pytest.raises(PermissionError, match="不是源码补丁") as refused:
        toolbox.propose_repair(hypothesis_id="H1", path="compute.py", new_text="", rationale="")
    # The gate reads remedy mid-turn but a hypothesis is only written when the turn
    # concludes, so the refusal has to name the way out of that.
    assert "下一轮再提补丁" in str(refused.value)

    # With a second objection standing, rewriting remedy would not get the patch
    # through either, so the refusal must not promise that it would.
    case.propose(Hypothesis(
        hypothesis_id="H2", target_segment=Segment.ALGORITHM, target_id="perception",
        statement="感知位姿也可能偏了", evidence_ids=(output,), remedy="handoff",
    ))
    with pytest.raises(PermissionError, match="没有软件层定位") as unlocated:
        toolbox.propose_repair(hypothesis_id="H2", path="compute.py", new_text="", rationale="")
    assert "下一轮再提补丁" not in str(unlocated.value)

    # Source alone neither clears nor implicates anything.
    with pytest.raises(ValueError, match="source alone"):
        case.record(SegmentFinding(
            segment=Segment.ALGORITHM, target_id="perception", status=SegmentStatus.CLEARED,
            note="代码看起来没问题", evidence_ids=(read["evidence_id"],),
        ))
    from visiondoctor.investigation.agent import _findings

    downgraded = _findings({"findings": [{
        "target_id": "perception", "status": "cleared", "note": "代码看起来没问题",
        "evidence_ids": [read["evidence_id"]],
    }]}, case)[0]
    assert downgraded.status is SegmentStatus.UNTESTED
    assert "只引用了源码" in downgraded.limitations


def test_a_refused_patch_is_reported_to_the_model_rather_than_ending_the_turn(
    tmp_path: Path,
) -> None:
    """The refusal has to come back as a tool result.

    Letting it escape ended the turn and left the call in the transcript with no
    result, which the next turn read as though the patch had been submitted.
    """

    case, adapter, bundle, references = _bound_case(tmp_path)
    output = _locate_command_fault(case, adapter, bundle, references, "handoff")
    gateway = ProtocolDoubleGateway([
        _turn("propose_repair", {
            "hypothesis_id": "H1", "path": "compute.py",
            "new_text": "print('patched')\n", "rationale": "少一次工具补偿",
        }, 1),
        _answer({
            "findings": [{
                "target_id": "tool_command", "status": "suspect",
                "evidence_ids": [output], "note": "记录的法兰指令偏离期望",
                "checked_scope": "A 的指令与期望对照", "limitations": "尚未读源码",
            }],
            "hypotheses": [{
                "hypothesis_id": "H2", "target_id": "tool_command",
                "statement": "指令计算多加了一次工具补偿", "evidence_ids": [output],
                "prediction": "改回一次补偿后偏差消失", "next_check": "提交源码补丁并复跑",
                "remedy": "source_patch",
            }],
            "next_step": "按源码补丁重新提交",
        }),
    ])
    turn = investigate(
        case=case, adapter=adapter, bundle=bundle, prompt="源码已连接，请修复",
        gateway=gateway, sandbox_root=tmp_path / "sandbox",
    )

    assert turn.hypotheses[0].remedy == "source_patch"
    assert not case.repair_plans          # the gate still refused the patch
    refusals = [message for message in case.transcript
                if message.get("role") == "tool" and "不能提交源码补丁" in message["content"]]
    assert len(refusals) == 1


def test_released_program_is_replayed_without_ever_opening_source(tmp_path: Path) -> None:
    case, adapter, bundle, references = _bound_case(tmp_path, source_readable=False)
    toolbox = Toolbox(case, adapter, bundle, sandbox_root=tmp_path / "sandbox")
    replayed = toolbox.replay_running_version()
    assert replayed["layer"] == "software"
    assert {row["input"] for row in replayed["replays"]} == {
        "parts-A-algorithm-input.json", "parts-B-algorithm-input.json",
    }
    assert not any(row["reproduced"] for row in replayed["replays"])
    _locate_command_fault(case, adapter, bundle, references, "handoff")
    names = {item["function"]["name"] for item in tools_for(case)}
    assert "replay_running_version" in names and "read_source" not in names
    assert not any(item.layer == "source" for item in case.evidence)


def test_model_turn_uses_non_geometry_knowledge_and_survives_reload(tmp_path: Path) -> None:
    service = CaseService(tmp_path / "cases")
    case_id = service.create("到位但没有抓住")
    fixture, adapter, references = _case()
    record = service.record(case_id)
    record.case.admit(adapter.collect())
    record.adapter, record.bundle = adapter, fixture.observations[0]
    record.observation_dirs.append(str(adapter.root))
    ev = references["parts/A/motion.json"]
    gateway = ProtocolDoubleGateway([
        _turn("inspect_grasp_graph", {"target_id": "motion_contact"}, 1),
        _turn("read_domain_knowledge", {"knowledge_ids": ["contact", "measurement"]}, 2),
        _turn("read_evidence", {"evidence_ids": [ev]}, 3),
        _answer({
            "findings": [{
                "target_id": "contact", "status": "untested", "evidence_ids": [],
                "note": "现有记录没有工件保持观测", "checked_scope": "查看运动记录",
                "limitations": "到位不能证明接触或保持",
            }],
            "hypotheses": [{
                "hypothesis_id": "H1", "target_id": "motion_contact", "statement": "可能未形成接触",
                "evidence_ids": [ev], "prediction": "末端到位但闭合后工件仍留在原位",
                "next_check": "补充闭合与抬起阶段视频", "remedy": "more_evidence",
            }],
            "next_step": "补充接触与抬起观测后区分空夹和滑移，暂不提出代码补丁。",
        }),
    ])
    turn = investigate(
        case=record.case, adapter=adapter, bundle=record.bundle,
        prompt="机器人到位但没有抓住", gateway=gateway,
    )
    record.turns.append(turn)
    service.keep(case_id)
    loaded = CaseService(service.root)
    view = loaded.view(case_id)
    assert view["hypotheses"][0]["target_id"] == "motion_contact"
    assert view["hypotheses"][0]["next_check"] == "补充闭合与抬起阶段视频"
    assert not view["gates"]["source_layer"]["passed"]
    assert not view["plans"]
    assert "graph" in view and "contact" in graph_dot(view["graph"])
    assert "read_domain_knowledge" in {
        call.name for call in loaded.record(case_id).turns[0].calls
    }


def test_invalid_graph_target_cannot_partially_commit_a_turn() -> None:
    case, adapter, references = _case()
    turn = Investigation(case, Toolbox(case, adapter, case.observations[0]), "核对")
    ev = references["parts/A/motion.json"]
    turn.invoke("read_evidence", evidence_ids=[ev])
    with pytest.raises(ValueError, match="unknown grasp graph target"):
        turn.commit(
            findings=(SegmentFinding(
                segment=Segment.ROBOT, target_id="robot", status=SegmentStatus.CLEARED,
                note="检查", evidence_ids=(ev,),
            ),),
            hypotheses=(Hypothesis(
                hypothesis_id="H1", target_segment=Segment.ROBOT, target_id="invented_sensor",
                statement="虚构目标", evidence_ids=(ev,),
            ),), next_step="结束",
        )
    assert not case.findings and not case.hypotheses


def test_numeric_check_separates_total_compensation_from_extra_error() -> None:
    from visiondoctor.investigation import check_transform_chain

    def pose(position):
        return {"position": position, "quaternion_xyzw": [0, 0, 0, 1]}

    result = check_transform_chain(
        detection=pose([1, 2, 3]), camera_to_base=pose([0, 0, 0]),
        pick_offset=pose([0, 0, 0]), tool0_to_tcp=pose([0, 0, 0.04]),
        commanded_flange=pose([1, 2, 2.88]),
    ).as_dict()
    assert result["flange_if_tool_inverted"]["position"] == [1, 2, 2.96]
    assert result["command_minus_inverted_flange_m"] == [0, 0, -0.08]
    assert result["residual_if_tool_inverted"]["position_m"] == 0.08


def test_exhausted_tool_budget_switches_to_conclusion_and_keeps_evidence(monkeypatch) -> None:
    import visiondoctor.investigation.agent as agent_module
    import visiondoctor.investigation.turn as turn_module

    monkeypatch.setattr(agent_module, "CALL_BUDGET", 2)
    monkeypatch.setattr(turn_module, "CALL_BUDGET", 2)
    case, adapter, references = _case()
    ev = references["parts/A/motion.json"]

    class BudgetGateway(ProtocolDoubleGateway):
        calls = 0

        def complete(self, messages, tools):
            self.calls += 1
            if self.calls == 3:
                assert not tools
                assert "预算已用尽" in messages[-1]["content"]
            return super().complete(messages, tools)

    gateway = BudgetGateway([
        _turn("read_evidence", {"evidence_ids": [ev]}, 1),
        _turn("inspect_grasp_graph", {"target_id": "contact"}, 2),
        _answer({"findings": [], "hypotheses": [], "next_step": "接触仍未知，需要闭合后视频。"}),
    ])
    result = investigate(case=case, adapter=adapter, bundle=case.observations[0],
                         prompt="查一下", gateway=gateway)
    assert len(result.calls) == 2
    assert ev in case.examined
    assert any(item["role"] == "tool" for item in case.transcript)


def test_invalid_conclusion_target_can_be_corrected_without_losing_the_turn():
    case, adapter, _ = _case()
    gateway = ProtocolDoubleGateway([
        _answer({"findings": [], "hypotheses": [
            {"target_segment": "task", "statement": "缺少证据"},
        ], "next_step": "请求补证"}),
        _answer({"findings": [], "hypotheses": [
            {"target_id": "task", "statement": "缺少证据", "remedy": "more_evidence"},
        ], "next_step": "请求补证"}),
    ])
    result = investigate(case=case, adapter=adapter, bundle=case.observations[0],
                         prompt="检查", gateway=gateway)
    assert result.hypotheses[0].target_id == "task"
    assert len(case.hypotheses) == 1 and not result.calls
    assert any("尚未提交" in str(m.get("content")) for m in case.transcript)


def test_two_verdicts_on_one_object_are_sent_back_instead_of_overwritten():
    case, adapter, references = _case()
    ev = references["parts/A/motion.json"]
    verdict = {"target_id": "command_motion", "note": "跟踪记录", "evidence_ids": [ev]}
    gateway = ProtocolDoubleGateway([
        _turn("read_evidence", {"evidence_ids": [ev]}, 1),
        _answer({"findings": [{**verdict, "status": "suspect"}, {**verdict, "status": "cleared"}],
                 "hypotheses": [], "next_step": "核对"}),
        _answer({"findings": [{**verdict, "status": "suspect"}], "hypotheses": [],
                 "next_step": "核对"}),
    ])
    investigate(case=case, adapter=adapter, bundle=case.observations[0],
                prompt="检查", gateway=gateway)
    assert [item.status for item in case.findings] == [SegmentStatus.SUSPECT]
    assert any("两条判定" in str(m.get("content")) for m in case.transcript)
