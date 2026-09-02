"""What the model is shown, and what it is allowed to ask for.

The view carries the catalogue but never the contents: an artifact enters the
reasoning only by being asked for, so that citing it means something.  The
prompt states the rules of demarcation and nothing about this particular cell --
naming the answer here would make the demarcation a recital.
"""

from __future__ import annotations

from typing import Any

from visiondoctor.case import SEGMENT_SCOPE, Case, diagnosis_gate
from visiondoctor.environment import ObservationBundle

SYSTEM_PROMPT = """\
你在对一次机器视觉抓取失败做定界。你看到的只有一次观察产生的证据包，没有别的信息来源。

诊断链路把系统分成若干段，每段的 scope 写明了它负责什么；判定要落在职责相符的那一段上。
你要对段给出判定，每段只能是三种之一：
- untested：还没查
- cleared：查过，这段没有偏离
- suspect：查过，这段可疑

规则：
1. cleared 和 suspect 都必须引用 evidence_id，而且只能引用你**真的调工具读过**的证据。
   没读过就引用会被拒绝，整轮作废。
2. 不要求每段都有判定。查不动的段就留 untested。
3. 证据不足以区分几种解释时，提出你还需要什么证据，而不是硬给结论。
4. 你现在看不到项目源码。定界通过之前不会给你。
5. check_transform_chain 只返回重算位姿和残差。两个残差字段名不含对错含义，哪一种才是
   本系统应当采用的约定，要你自己判断。它不会告诉你哪段有问题，也不会告诉你
   故障属于哪一类——那是你要判断的。同一组残差可能来自代码缺陷、配置写反或标定漂移，
   这三者的修复方式完全不同，你必须说清楚是哪一种以及依据。

6. 诊断门未通过时你看不到源码。门要求：至少一段 suspect、至少一段 cleared、至少一个有证据
   支撑的假设。通过之后 list_source / read_source / propose_repair 才会出现在工具里。
7. 提交候选修复时，补丁会在隔离工作树里用本次运行**真实记录的输入**复跑一遍。复跑通过只说明
   代码现在算出了链路本来的意图，**不等于现场已经恢复**——现场恢复只能由改动应用后新采集的
   证据授予。

只返回一个 JSON 对象：
{
  "findings": [{"segment": "...", "status": "cleared|suspect|untested",
                "note": "简体中文，说明依据", "evidence_ids": ["EV-001"]}],
  "hypotheses": [{"hypothesis_id": "H1", "target_segment": "...",
                  "statement": "简体中文", "evidence_ids": ["EV-001"]}],
  "next_step": "简体中文，说明下一步该做什么"
}
"""

TOOLS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "function": {
            "name": "read_evidence",
            "description": (
                "读取证据，一次最多 8 件。文本与 JSON 直接返回内容；"
                "图片返回逐图观察；深度图返回统计摘要。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 8,
                    }
                },
                "required": ["evidence_ids"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_transform_chain",
            "description": (
                "用你指定的证据重算声明的变换链，返回期望 TCP、指令位姿，"
                "以及指令相对于两种工具复合方式各自的残差。只有测量，没有结论。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "detection_evidence_id": {"type": "string"},
                    "calibration_evidence_id": {"type": "string"},
                    "tool_evidence_id": {"type": "string"},
                    "command_evidence_id": {"type": "string"},
                    "motion_evidence_id": {"type": "string"},
                },
                "required": [
                    "detection_evidence_id",
                    "calibration_evidence_id",
                    "tool_evidence_id",
                    "command_evidence_id",
                ],
                "additionalProperties": False,
            },
        },
    },
)


SOURCE_TOOLS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "function": {
            "name": "list_source",
            "description": "列出绑定提交下的源码文件名。",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_source",
            "description": (
                "读取绑定提交下的一个源码文件——"
                "是当时运行的那份，不是当前工作树。"
            ),
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_repair",
            "description": (
                "提交一个候选修复：给出文件路径和该文件修改后的完整内容。"
                "它会在隔离工作树里用本次真实输入复跑并返回结果，绑定的仓库不会被改动。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target_segment": {"type": "string"},
                    "hypothesis_id": {"type": "string"},
                    "path": {"type": "string"},
                    "new_text": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": [
                    "target_segment",
                    "hypothesis_id",
                    "path",
                    "new_text",
                    "rationale",
                ],
                "additionalProperties": False,
            },
        },
    },
)


def tools_for(case: Case) -> tuple[dict[str, Any], ...]:
    """Source becomes visible only once the demarcation carries itself."""

    if case.project is not None and diagnosis_gate(case).passed:
        return TOOLS + SOURCE_TOOLS
    return TOOLS


def build_view(case: Case, bundle: ObservationBundle) -> dict[str, Any]:
    """Case state, the observation's own account of itself, and the catalogue."""

    verdict = diagnosis_gate(case)
    return {
        "case": {
            "case_id": case.case_id,
            "title": case.title,
            "project_bound": case.project is not None,
            "diagnosis_gate": {"passed": verdict.passed, "missing": list(verdict.reasons)},
            "source_visible": bool(case.project is not None and verdict.passed),
        },
        "observation": {
            "run_id": bundle.run_id,
            "collected_at": bundle.created_at.isoformat(),
            "project_revision": bundle.project_revision,
            "task_results": [result.model_dump(mode="json") for result in bundle.results],
            "timeline": [event.model_dump(mode="json") for event in bundle.timeline],
            "cross_source_comparable": bundle.cross_source_comparable,
        },
        "chain": [
            {
                "segment": segment.value,
                "status": status.value,
                "scope": SEGMENT_SCOPE[segment],
            }
            for segment, status in case.demarcation().items()
        ],
        "evidence_catalogue": [
            {
                "evidence_id": item.evidence_id,
                "reference": item.reference,
                "media_type": item.media_type,
                "captured_at": item.captured_at.isoformat(),
                "clock_domain": item.clock_domain,
            }
            for item in case.evidence
        ],
    }
