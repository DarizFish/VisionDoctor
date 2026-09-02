"""One push forward, and the ledger that proves what it did.

A human message moves a case at most one turn.  Inside that turn the model may
call tools; the host runs them, records each call and what it delivered, and
only then accepts conclusions -- and only conclusions that cite evidence which
actually came back.  The ledger answers "what was checked", not "what was said".
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from visiondoctor.case import Case, Hypothesis, SegmentFinding

from .tools import Toolbox

CALL_BUDGET = 20


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    arguments: dict[str, Any]
    at: datetime
    requested: tuple[str, ...] = ()
    delivered: tuple[str, ...] = ()


class DecisionTurn(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    turn_id: str
    case_id: str
    prompt: str
    calls: tuple[ToolCall, ...]
    findings: tuple[SegmentFinding, ...]
    hypotheses: tuple[Hypothesis, ...]
    next_step: str


class Investigation:
    """The host side of one turn: it runs the tools and keeps the ledger."""

    def __init__(
        self,
        case: Case,
        toolbox: Toolbox,
        prompt: str,
        observer: Callable[[ToolCall], None] | None = None,
    ) -> None:
        self.case = case
        self.toolbox = toolbox
        self.prompt = prompt
        self.calls: list[ToolCall] = []
        #: Told about each call as it lands, so a viewer can watch it happen.
        self.observer = observer
        self._committed = False

    def invoke(self, name: str, **arguments: Any) -> Any:
        if self._committed:
            raise RuntimeError("this turn is already closed")
        if len(self.calls) >= CALL_BUDGET:
            raise RuntimeError(f"a turn may make at most {CALL_BUDGET} tool calls")
        tool = getattr(self.toolbox, name, None)
        if tool is None or name.startswith("_"):
            raise KeyError(f"{name} is not on the investigation surface")
        before = set(self.case.examined)
        result = tool(**arguments)
        recorded = ToolCall(
            name=name,
            arguments=arguments,
            at=datetime.now(UTC),
            requested=tuple(
                str(value)
                for key, value in arguments.items()
                if key.endswith("_evidence_id") and value
            )
            or tuple(str(item) for item in arguments.get("evidence_ids") or ()),
            delivered=tuple(sorted(self.case.examined - before)),
        )
        self.calls.append(recorded)
        if self.observer is not None:
            self.observer(recorded)
        return result

    def commit(
        self,
        *,
        findings: tuple[SegmentFinding, ...] = (),
        hypotheses: tuple[Hypothesis, ...] = (),
        next_step: str,
    ) -> DecisionTurn:
        """Accept the turn's conclusions, or refuse them and leave the case alone."""

        if self._committed:
            raise RuntimeError("this turn is already closed")
        for finding in findings:
            self.case.record(finding)
        for hypothesis in hypotheses:
            self.case.propose(hypothesis)
        self._committed = True
        return DecisionTurn(
            turn_id=f"TURN-{uuid.uuid4().hex[:12]}",
            case_id=self.case.case_id,
            prompt=self.prompt,
            calls=tuple(self.calls),
            findings=tuple(findings),
            hypotheses=tuple(hypotheses),
            next_step=next_step,
        )
