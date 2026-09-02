"""Drive one investigation turn with a real model.

The model decides what to look at; the host executes every look, records it and
hands back only what it actually read.  When the model stops calling tools it
must produce the turn's conclusions, which are accepted only if the evidence
they cite came back through the host.
"""

from __future__ import annotations

import json
from typing import Any

from visiondoctor.case import Case, Hypothesis, Segment, SegmentFinding, SegmentStatus
from visiondoctor.environment import FileBundleAdapter, ObservationBundle
from visiondoctor.llm import ModelGateway
from visiondoctor.multimodal import VisionGateway

from .tools import Toolbox
from .turn import CALL_BUDGET, DecisionTurn, Investigation
from .view import SYSTEM_PROMPT, TOOLS, build_view


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


def _findings(payload: dict[str, Any]) -> tuple[SegmentFinding, ...]:
    return tuple(
        SegmentFinding(
            segment=Segment(item["segment"]),
            status=SegmentStatus(item["status"]),
            note=str(item.get("note") or ""),
            evidence_ids=tuple(item.get("evidence_ids") or ()),
        )
        for item in payload.get("findings") or ()
    )


def _hypotheses(payload: dict[str, Any]) -> tuple[Hypothesis, ...]:
    return tuple(
        Hypothesis(
            hypothesis_id=str(item.get("hypothesis_id") or f"H{index + 1}"),
            target_segment=Segment(item["target_segment"]),
            statement=str(item.get("statement") or ""),
            evidence_ids=tuple(item.get("evidence_ids") or ()),
        )
        for index, item in enumerate(payload.get("hypotheses") or ())
    )


def investigate(
    *,
    case: Case,
    adapter: FileBundleAdapter,
    bundle: ObservationBundle,
    prompt: str,
    gateway: ModelGateway,
    vision: VisionGateway | None = None,
) -> DecisionTurn:
    """Let the model push the case one turn, and keep the ledger while it does."""

    turn = Investigation(case, Toolbox(case, adapter, bundle, vision), prompt)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(build_view(case, bundle), ensure_ascii=False, indent=2)
            + "\n\n"
            + prompt,
        },
    ]
    for _ in range(CALL_BUDGET + 1):
        answer = gateway.complete(messages, TOOLS)
        if not answer.tool_calls:
            payload = _decode(answer.content)
            return turn.commit(
                findings=_findings(payload),
                hypotheses=_hypotheses(payload),
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
