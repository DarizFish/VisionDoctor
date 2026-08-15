from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import httpx
import pytest

from visiondoctor.llm import (
    AssistantTurn,
    ModelProtocolError,
    ModelSettings,
    OpenAICompatibleGateway,
    ToolCall,
)
from visiondoctor.llm.settings import ModelConfigurationError
from visiondoctor.llm.tools import (
    RepositoryInspector,
    StrictToolLoop,
    build_patch_from_changes,
    terminal_tool,
)
from visiondoctor.product import run_incident
from visiondoctor.sandbox import GitWorktreeSandbox
from visiondoctor.schemas import CandidateKind, CandidateVersion, WorkflowState
from visiondoctor.workflow import DemoRunResult


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


class _PlainTextGateway:
    model = "invalid-test-provider"

    def complete(self, messages, tools) -> AssistantTurn:
        del messages, tools
        return AssistantTurn(
            content="I think it is fixed.",
            tool_calls=(),
            finish_reason="stop",
            raw_message={"role": "assistant", "content": "I think it is fixed."},
        )


class _RepeatedInvalidTerminalGateway:
    model = "repeated-invalid-terminal-test"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages, tools) -> AssistantTurn:
        del messages, tools
        self.calls += 1
        if self.calls == 1:
            calls = (
                ToolCall(
                    call_id="inspect-before-invalid",
                    name="inspect_commit_diff",
                    arguments={},
                ),
            )
        else:
            calls = (
                ToolCall(
                    call_id=f"invalid-terminal-{self.calls}",
                    name="submit",
                    arguments={"result": "invalid"},
                ),
            )
        return AssistantTurn(
            content="",
            tool_calls=calls,
            finish_reason="tool_calls",
            raw_message={"role": "assistant", "tool_calls": []},
        )


def test_agent_loop_does_not_fallback_when_model_skips_tools(
    demo_result: DemoRunResult,
) -> None:
    incident = demo_result.incident
    inspector = RepositoryInspector(
        Path(incident.repository.path), incident.baseline_commit, incident.faulty_commit
    )
    loop = StrictToolLoop(_PlainTextGateway(), inspector)

    with pytest.raises(ModelProtocolError, match="without calling"):
        loop.run(
            system_prompt="Use tools.",
            task_payload={"issue": "untrusted"},
            terminal_tool=terminal_tool(
                "submit",
                "submit result",
                {"result": {"type": "string"}},
                ["result"],
            ),
            terminal_name="submit",
            validate_terminal=lambda value: value,
        )


def test_agent_loop_bounds_rejected_terminal_repairs(
    demo_result: DemoRunResult,
) -> None:
    incident = demo_result.incident
    inspector = RepositoryInspector(
        Path(incident.repository.path), incident.baseline_commit, incident.faulty_commit
    )
    gateway = _RepeatedInvalidTerminalGateway()
    loop = StrictToolLoop(gateway, inspector, max_iterations=10)

    def reject_terminal(value: dict[str, str]) -> dict[str, str]:
        del value
        raise ValueError("customer-facing result is invalid")

    with pytest.raises(ModelProtocolError, match="rejected after 3 attempts"):
        loop.run(
            system_prompt="Use tools.",
            task_payload={"issue": "untrusted"},
            terminal_tool=terminal_tool(
                "submit",
                "submit result",
                {"result": {"type": "string"}},
                ["result"],
            ),
            terminal_name="submit",
            validate_terminal=reject_terminal,
            max_terminal_attempts=3,
        )

    assert gateway.calls == 4


def test_product_run_forbids_local_execution_without_fallback(
    demo_result: DemoRunResult, tmp_path: Path, model_gateway_factory
) -> None:
    with pytest.raises(ValueError, match="no local fallback"):
        run_incident(
            tmp_path / "product-local",
            demo_result.incident,
            sandbox_mode="local",  # type: ignore[arg-type]
            model_gateway=model_gateway_factory(),
        )


def test_product_runs_a_caller_supplied_incident(
    demo_result: DemoRunResult, tmp_path: Path, model_gateway_factory
) -> None:
    result = run_incident(
        tmp_path / "external-product-run",
        demo_result.incident,
        model_gateway=model_gateway_factory(),
    )

    assert result.state is WorkflowState.AWAITING_HUMAN_APPROVAL
    assert result.diagnosis.model == "test-protocol-double"
    assert result.selected_candidate.kind.value == "generated"
    assert result.incident.repository.path == demo_result.incident.repository.path


@pytest.mark.parametrize(
    ("before", "after", "operation"),
    [
        ("value = 1\n", "value = 2", "update"),
        ("value = 1", "value = 2\n", "update"),
        ("value = 1", "value = 2", "update"),
        (None, "value = 2", "create"),
    ],
)
def test_generated_patch_preserves_missing_final_newline_and_applies(
    tmp_path: Path,
    before: str | None,
    after: str,
    operation: str,
) -> None:
    repository = tmp_path / "patch-repository"
    repository.mkdir()

    def git(*arguments: str, input_bytes: bytes | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(repository), *arguments],
            input=input_bytes,
            capture_output=True,
            check=False,
        )

    assert git("init", "-q", "-b", "main").returncode == 0
    assert git("config", "user.name", "Patch Test").returncode == 0
    assert git("config", "user.email", "patch-test@example.invalid").returncode == 0
    assert git("config", "core.autocrlf", "false").returncode == 0
    (repository / "anchor.txt").write_text("anchor\n", encoding="utf-8")
    if before is not None:
        (repository / "target.py").write_bytes(before.encode("utf-8"))
    assert git("add", ".").returncode == 0
    assert git("commit", "-q", "-m", "base").returncode == 0
    commit = git("rev-parse", "HEAD").stdout.decode("utf-8").strip()
    inspector = RepositoryInspector(repository, commit, commit)

    patch_text, files = build_patch_from_changes(
        inspector,
        [{"path": "target.py", "operation": operation, "content": after}],
    )

    checked = git("apply", "--check", "-", input_bytes=patch_text.encode("utf-8"))
    assert checked.returncode == 0, checked.stderr.decode("utf-8", errors="replace")
    assert patch_text.endswith("\n")
    assert "\\ No newline at end of file\n" in patch_text
    sandbox = GitWorktreeSandbox(repository, tmp_path / "worktrees")
    handle = sandbox.create(
        CandidateVersion(
            candidate_id="missing-final-newline",
            kind=CandidateKind.ROOT_CAUSE_FIX,
            base_commit=commit,
            patch_text=patch_text,
            rationale="exercise the production patch application path",
            expected_changed_files=("target.py",),
        )
    )
    assert (handle.worktree / "target.py").read_bytes() == after.encode("utf-8")
    assert sandbox.cleanup(handle, rollback=False)
    applied = git("apply", "-", input_bytes=patch_text.encode("utf-8"))
    assert applied.returncode == 0, applied.stderr.decode("utf-8", errors="replace")
    assert (repository / "target.py").read_bytes() == after.encode("utf-8")
    assert files == ("target.py",)
