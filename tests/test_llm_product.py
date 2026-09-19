from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pytest

from visiondoctor.llm import (
    ModelSettings,
    OpenAICompatibleGateway,
)
from visiondoctor.llm.settings import ModelConfigurationError
from visiondoctor.llm.tools import (
    terminal_tool,
)


def test_model_settings_fail_closed_without_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("VISIONDOCTOR_ENV_FILE", str(tmp_path / "missing.env"))
    monkeypatch.delenv("VISIONDOCTOR_LLM_API_KEY", raising=False)

    with pytest.raises(ModelConfigurationError, match="required|missing"):
        ModelSettings.from_environment()


def test_model_settings_load_allowlisted_dotenv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "IGNORED_SETTING=do-not-load\n"
        "VISIONDOCTOR_LLM_API_KEY=test-key\n"
        "VISIONDOCTOR_LLM_MODEL=deepseek-v4-flash\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("VISIONDOCTOR_ENV_FILE", str(env_file))
    for key in (
        "VISIONDOCTOR_LLM_API_KEY",
        "VISIONDOCTOR_LLM_MODEL",
        "IGNORED_SETTING",
    ):
        monkeypatch.delenv(key, raising=False)

    settings = ModelSettings.from_environment()

    assert settings.api_key == "test-key"
    assert settings.model == "deepseek-v4-flash"
    assert "IGNORED_SETTING" not in os.environ


def test_openai_compatible_gateway_executes_tool_call_and_redacts_key(tmp_path: Path) -> None:
    secret = "unit-test-secret"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == f"Bearer {secret}"
        payload = json.loads(request.content)
        assert payload["model"] == "deepseek-v4-flash"
        assert payload["tool_choice"] == "auto"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "report_ready",
                                        "arguments": '{"ready":true}',
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
            },
        )

    audit_path = tmp_path / "model_audit.jsonl"
    gateway = OpenAICompatibleGateway(
        ModelSettings(api_key=secret),
        audit_path=audit_path,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    turn = gateway.complete(
        [{"role": "user", "content": "check"}],
        (
            terminal_tool(
                "report_ready",
                "protocol check",
                {"ready": {"type": "boolean"}},
                ["ready"],
            ),
        ),
    )

    assert turn.tool_calls[0].arguments == {"ready": True}
    audit = audit_path.read_text(encoding="utf-8")
    assert secret not in audit
    assert "report_ready" in audit
