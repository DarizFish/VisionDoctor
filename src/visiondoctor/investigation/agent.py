"""Drive one investigation turn with a real model.

The model decides what to look at; the host executes every look, records it and
hands back only what it actually read.  When the model stops calling tools it
must produce the turn's conclusions, which are accepted only if the evidence
they cite came back through the host.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from visiondoctor.case import Case, Hypothesis, Segment, SegmentFinding, SegmentStatus
from visiondoctor.case.graph import consuming_node, target_segment
from visiondoctor.case.isolation import REFERENCE as STRUCTURAL_REFERENCE
from visiondoctor.case.isolation import fault_node
from visiondoctor.environment import FileBundleAdapter, ObservationBundle
from visiondoctor.llm import ModelGateway
from visiondoctor.multimodal import VisionGateway

from .tools import Toolbox
from .turn import CALL_BUDGET, DecisionTurn, Investigation
from .view import SYSTEM_PROMPT, build_view, tools_for


class InvestigationError(RuntimeError):
    """The turn could not be closed; the case is left untouched."""


#: Marks the state snapshot that opens a turn, so an older one can be retired
#: when a fresher one arrives.  Only the newest snapshot is true.
VIEW_MARK = "【案件状态】\n"


def _resume(case: Case) -> list[dict[str, Any]]:
    """Carry the case's own conversation forward into this turn.

    Everything said so far comes back: what the model asked for, what the host
    handed over, what it concluded.  Only the state snapshots are collapsed --
    a superseded one would contradict the current chain.
    """

    if not case.transcript:
        return [{"role": "system", "content": SYSTEM_PROMPT}]
    carried: list[dict[str, Any]] = []
    for message in case.transcript:
        if message.get("role") == "system":
            carried.append({"role": "system", "content": SYSTEM_PROMPT})
            continue
        content = message.get("content")
        if isinstance(content, str) and content.startswith(VIEW_MARK):
            question = content.split("\n\n", maxsplit=1)[-1]
            carried.append({"role": "user", "content": question})
        else:
            carried.append(message)
    return carried


def _decode(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", maxsplit=1)[-1].rsplit("```", maxsplit=1)[0]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        raise InvestigationError("模型没有返回 JSON 结论")
    return json.loads(text[start : end + 1])


def _findings(payload: dict[str, Any], case: Case) -> tuple[SegmentFinding, ...]:
    """Keep the citations the host actually delivered; drop the rest.

    A verdict left with nothing behind it is recorded as untested.  The turn
    still stands: refusing one citation should not throw away the work.
    """

    findings = []
    judged: set[str] = set()
    for item in payload.get("findings") or ():
        # Two verdicts on one object in one turn would silently keep whichever came last.
        key = str(item.get("target_id") or item.get("segment"))
        if key in judged:
            raise ValueError(f"同一对象 {key} 在一轮结论里有两条判定，请合并为一条")
        judged.add(key)
        evidence = tuple(
            name for name in (item.get("evidence_ids") or ()) if name in case.examined
        )
        status = SegmentStatus(item["status"])
        _agrees_with_cited_isolation(case, key, item.get("target_id"), status, evidence)
        limitations = str(item.get("limitations") or "")
        if not evidence:
            status = SegmentStatus.UNTESTED
        elif status is not SegmentStatus.UNTESTED and not case.running_evidence(evidence):
            # Source explains a located fault; it never locates or clears one.
            status = SegmentStatus.UNTESTED
            limitations = (limitations + "；" if limitations else "") + (
                "宿主：该判断只引用了源码，没有运行记录或观测支撑，降为未查"
            )
        findings.append(
            SegmentFinding(
                segment=(
                    target_segment(item["target_id"])
                    if item.get("target_id") else Segment(item["segment"])
                ),
                status=status,
                note=str(item.get("note") or ""),
                evidence_ids=evidence,
                target_id=item.get("target_id"),
                checked_scope=str(item.get("checked_scope") or ""),
                limitations=limitations,
            )
        )
    return tuple(findings)


def _agrees_with_cited_isolation(
    case: Case, key: str, target_id: str | None, status: SegmentStatus, evidence: tuple[str, ...],
) -> None:
    """A verdict may not cite the host's structural diagnosis for the opposite of what it found.

    Clearing the one candidate it isolated, or implicating a node it exonerated,
    while pointing at it, is sent back.  Overturning it takes evidence of its own.
    """

    node = consuming_node(target_id)
    if node is None or status is SegmentStatus.UNTESTED:
        return
    for item in case.evidence:
        if item.evidence_id not in evidence or item.reference != STRUCTURAL_REFERENCE:
            continue
        isolation = item.content["isolation"]
        examined = [fault for key in ("candidates", "exonerated", "unexamined")
                    for fault in isolation[key] if fault_node(fault) == node]
        isolated = {fault_node(fault) for fault in isolation["candidates"]}
        if status is SegmentStatus.CLEARED and isolated == {node}:
            raise ValueError(
                f"{key} 判为 cleared，但引用的结构核算 {item.evidence_id} 把它隔离为唯一候选。"
                "要推翻核算需另有核算之外的证据，且不能引用该核算作为排除依据"
            )
        if status is SegmentStatus.SUSPECT and examined and set(examined) <= set(
            isolation["exonerated"]
        ):
            raise ValueError(
                f"{key} 判为 suspect，但引用的结构核算 {item.evidence_id} 已将它核算排除"
                f"（{'、'.join(isolation['passed_tests'])} 通过）。请改判或去掉该引用并说明新证据"
            )


def _hypotheses(payload: dict[str, Any], case: Case) -> tuple[Hypothesis, ...]:
    return tuple(
        Hypothesis(
            hypothesis_id=str(item.get("hypothesis_id") or f"H{index + 1}"),
            target_segment=(
                target_segment(item["target_id"])
                if item.get("target_id") else Segment(item["target_segment"])
            ),
            statement=str(item.get("statement") or ""),
            evidence_ids=tuple(
                name for name in (item.get("evidence_ids") or ()) if name in case.examined
            ),
            target_id=item.get("target_id"),
            prediction=str(item.get("prediction") or ""),
            counter_evidence_ids=tuple(
                name for name in (item.get("counter_evidence_ids") or ()) if name in case.examined
            ),
            next_check=str(item.get("next_check") or ""),
            remedy=item.get("remedy") or "more_evidence",
        )
        for index, item in enumerate(payload.get("hypotheses") or ())
    )


def investigate(
    *,
    case: Case,
    adapter: FileBundleAdapter | None,
    bundle: ObservationBundle | None,
    prompt: str,
    gateway: ModelGateway,
    vision: VisionGateway | None = None,
    sandbox_root: Path | None = None,
    uploads: dict[str, Path] | None = None,
    observation_dirs: tuple[Path, ...] = (),
    observer: Callable[[Any], None] | None = None,
) -> DecisionTurn:
    """Let the model push the case one turn, and keep the ledger while it does."""

    toolbox = Toolbox(case, adapter, bundle, vision, sandbox_root, uploads, observation_dirs)
    turn = Investigation(case, toolbox, prompt, observer)
    messages: list[dict[str, Any]] = _resume(case) + [
        {
            "role": "user",
            "content": VIEW_MARK
            + json.dumps(build_view(case, bundle), ensure_ascii=False, indent=2)
            + "\n\n"
            + prompt,
        },
    ]
    # Keep delivered evidence even if the provider later fails to finish. A
    # resumed turn must not have to repeat already completed multimodal reads.
    case.transcript = messages
    malformed = False
    budget_warning = False
    closing = False
    for _ in range(CALL_BUDGET + 2):
        remaining = max(0, CALL_BUDGET - len(turn.calls))
        if remaining == 0 and not closing:
            closing = True
            messages.append({
                "role": "user",
                "content": "本轮工具预算已用尽。现在只根据已交付证据返回规定的 JSON 结论，"
                "不再调用工具。不能确定的保持未知，列出下一项取证；不要捏造排除或修复。",
            })
        elif remaining <= 3 and not budget_warning:
            budget_warning = True
            messages.append({
                "role": "user",
                "content": f"本轮还可调用 {remaining} 次工具。优先完成有区分价值的检查并收束，"
                "不需要逐项遍历系统图；已经充分支持的结论可以现在提交。",
            })
        answer = gateway.complete(messages, () if closing else tools_for(case))
        if not answer.tool_calls:
            try:
                payload = _decode(answer.content)
                findings = _findings(payload, case)
                hypotheses = _hypotheses(payload, case)
            except (InvestigationError, ValueError, KeyError, TypeError) as exc:
                if malformed:
                    raise InvestigationError(f"模型两次都没给出有效的 JSON 结论：{exc}") from exc
                malformed = True
                messages.append(answer.raw_message)
                messages.append(
                    {
                        "role": "user",
                        "content": f"上一条结论的 JSON 或字段无效（{exc}），尚未提交。"
                        "请只修正格式并返回规定 JSON；findings 和 hypotheses 使用案件图中"
                        "节点或边的 target_id，不把节点 ID 写成 target_segment。"
                        "保留证据依据与未知边界，不新增观察或改变事实。",
                    }
                )
                continue
            messages.append(answer.raw_message)
            case.transcript = messages
            return turn.commit(
                findings=findings,
                hypotheses=hypotheses,
                next_step=str(payload.get("next_step") or ""),
            )
        if closing:
            messages.append(answer.raw_message)
            raise InvestigationError("模型在无工具的收束阶段仍请求工具，没有给出结论")
        messages.append(answer.raw_message)
        for call in answer.tool_calls:
            try:
                result: Any = turn.invoke(call.name, **call.arguments)
            except (KeyError, ValueError, RuntimeError, TypeError) as exc:
                result = {"error": f"{type(exc).__name__}: {exc}"}
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.call_id,
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                }
            )
    raise InvestigationError("模型用完了调用预算仍未给出结论")
