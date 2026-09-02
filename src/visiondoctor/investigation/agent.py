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
from visiondoctor.environment import FileBundleAdapter, ObservationBundle
from visiondoctor.llm import ModelGateway
from visiondoctor.multimodal import VisionGateway

from .tools import Toolbox
from .turn import CALL_BUDGET, DecisionTurn, Investigation
from .view import SYSTEM_PROMPT, build_view, tools_for


class InvestigationError(RuntimeError):
    """The turn could not be closed; the case is left untouched."""


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
    for item in payload.get("findings") or ():
        evidence = tuple(
            name for name in (item.get("evidence_ids") or ()) if name in case.examined
        )
        status = SegmentStatus(item["status"])
        if not evidence:
            status = SegmentStatus.UNTESTED
        findings.append(
            SegmentFinding(
                segment=Segment(item["segment"]),
                status=status,
                note=str(item.get("note") or ""),
                evidence_ids=evidence,
            )
        )
    return tuple(findings)


def _hypotheses(payload: dict[str, Any], case: Case) -> tuple[Hypothesis, ...]:
    return tuple(
        Hypothesis(
            hypothesis_id=str(item.get("hypothesis_id") or f"H{index + 1}"),
            target_segment=Segment(item["target_segment"]),
            statement=str(item.get("statement") or ""),
            evidence_ids=tuple(
                name for name in (item.get("evidence_ids") or ()) if name in case.examined
            ),
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
    observer: Callable[[Any], None] | None = None,
) -> DecisionTurn:
    """Let the model push the case one turn, and keep the ledger while it does."""

    toolbox = Toolbox(case, adapter, bundle, vision, sandbox_root, uploads)
    turn = Investigation(case, toolbox, prompt, observer)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(build_view(case, bundle), ensure_ascii=False, indent=2)
            + "\n\n"
            + prompt,
        },
    ]
    malformed = False
    for _ in range(CALL_BUDGET + 2):
        answer = gateway.complete(messages, tools_for(case))
        if not answer.tool_calls:
            try:
                payload = _decode(answer.content)
            except (InvestigationError, json.JSONDecodeError) as exc:
                if malformed:
                    raise InvestigationError(f"模型两次都没给出可解析的 JSON：{exc}") from exc
                malformed = True
                messages.append(answer.raw_message)
                messages.append(
                    {
                        "role": "user",
                        "content": f"上一条不是可解析的 JSON（{exc}）。只输出那个 JSON 对象，"
                        "字符串里的引号和换行要正确转义，不要加任何其他文字。",
                    }
                )
                continue
            return turn.commit(
                findings=_findings(payload, case),
                hypotheses=_hypotheses(payload, case),
                next_step=str(payload.get("next_step") or ""),
            )
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
