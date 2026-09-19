from __future__ import annotations

from typing import Any

from visiondoctor.llm import AssistantTurn, ModelProtocolError, ToolCall


class ProtocolDoubleGateway:
    model = "test-protocol-double"

    def __init__(self, turns: list[AssistantTurn]) -> None:
        self.turns = turns

    def complete(
        self, messages: list[dict[str, Any]], tools: tuple[dict[str, Any], ...]
    ) -> AssistantTurn:
        del messages, tools
        if not self.turns:
            raise ModelProtocolError("test protocol double exhausted")
        return self.turns.pop(0)


def _turn(name: str, arguments: dict[str, Any], index: int) -> AssistantTurn:
    call_id = f"test-call-{index}"
    return AssistantTurn(
        content="",
        tool_calls=(ToolCall(call_id=call_id, name=name, arguments=arguments),),
        finish_reason="tool_calls",
        raw_message={
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": "{}"},
                }
            ],
        },
    )
