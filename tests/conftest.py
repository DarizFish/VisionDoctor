from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.support import REGRESSION_TEST_SOURCE
from visiondoctor.demo.repository_builder import BASELINE_SOURCE
from visiondoctor.demo.scenario import run_demo
from visiondoctor.llm import AssistantTurn, ModelProtocolError, ToolCall
from visiondoctor.workflow import DemoRunResult


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


def make_test_model_gateway() -> ProtocolDoubleGateway:
    return ProtocolDoubleGateway(
        [
            _turn("inspect_commit_diff", {}, 1),
            _turn(
                "submit_diagnosis",
                {
                    "root_cause": (
                        "The faulty commit reverses non-commuting target-source transform "
                        "composition from T_base_camera @ T_camera_object to the opposite order."
                    ),
                    "confirmed": True,
                    "confidence": 0.99,
                    "observations": [
                        "The baseline passes while the faulty commit fails pose gates.",
                        "The commit diff reverses the two transform operands.",
                    ],
                    "recommended_fix": (
                        "Restore target-source multiplication order and add a non-commuting test."
                    ),
                },
                2,
            ),
            _turn(
                "read_repository_file",
                {"path": "src/pose_transformer.py", "start_line": 1, "end_line": 200},
                3,
            ),
            _turn(
                "submit_patch",
                {
                    "rationale": "Restore transform order and add a non-commuting regression.",
                    "changes": [
                        {
                            "path": "src/pose_transformer.py",
                            "operation": "update",
                            "content": BASELINE_SOURCE,
                        },
                        {
                            "path": "tests/test_transform_order_regression.py",
                            "operation": "create",
                            "content": REGRESSION_TEST_SOURCE,
                        },
                    ],
                },
                4,
            ),
        ]
    )


@pytest.fixture
def model_gateway_factory():
    return make_test_model_gateway


@pytest.fixture(scope="session")
def demo_result(tmp_path_factory: pytest.TempPathFactory) -> DemoRunResult:
    workspace = tmp_path_factory.mktemp("visiondoctor") / "demo"
    return run_demo(Path(workspace), model_gateway=make_test_model_gateway())
