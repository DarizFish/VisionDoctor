from __future__ import annotations

import base64
import io
import subprocess
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from visiondoctor.intake import inspect_repository
from visiondoctor.llm import AssistantTurn, ModelProtocolError, ToolCall
from visiondoctor.sessions import DiagnosisSessionService


class ConversationGatewayDouble:
    model = "conversation-test-model"

    def __init__(self, baseline: str, current: str) -> None:
        self.baseline = baseline
        self.current = current
        self.messages: list[dict[str, Any]] = []

    def complete(
        self, messages: list[dict[str, Any]], tools: tuple[dict[str, Any], ...]
    ) -> AssistantTurn:
        self.messages = messages
        assert tools[0]["function"]["name"] == "reply_to_diagnosis_session"
        return AssistantTurn(
            content="",
            tool_calls=(
                ToolCall(
                    call_id="conversation-call-1",
                    name="reply_to_diagnosis_session",
                    arguments={
                        "assistant_message": "我已经结合现象、代码历史和仿真采集整理好诊断范围。",
                        "title": "目标位姿稳定偏移",
                        "questions": [],
                        "next_actions": ["开始深入诊断"],
                        "repository_choice": {
                            "baseline_commit": self.baseline,
                            "current_commit": self.current,
                            "reason": "症状出现在最近一次视觉节点改动之后。",
                        },
                        "execution_contract": {
                            "runner_script": "runner.py",
                            "test_runner_script": "test_runner.py",
                        },
                        "attachment_assessments": [],
                    },
                ),
            ),
            finish_reason="tool_calls",
            raw_message={"role": "assistant", "tool_calls": []},
        )


class VisionGatewayDouble:
    model = "vision-test-model"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def assess(
        self,
        image_path: Path,
        *,
        attachment_id: str,
        visible_name: str,
        user_context: str,
    ) -> dict[str, Any]:
        with Image.open(image_path) as image:
            assert image.format == "PNG"
            assert image.width <= 1600 and image.height <= 1600
        self.calls.append(
            {
                "attachment_id": attachment_id,
                "visible_name": visible_name,
                "user_context": user_context,
            }
        )
        return {
            "attachment_id": attachment_id,
            "visible_name": visible_name,
            "observations": ["画面中可见一个蓝色区域。"],
            "diagnostic_relevance": "可用于检查目标在画面中的相对位置。",
            "limitations": ["单张彩色图不能提供可靠深度。"],
            "confidence": 0.9,
            "model": self.model,
        }


class VocabularyRepairGateway(ConversationGatewayDouble):
    def __init__(self, baseline: str, current: str) -> None:
        super().__init__(baseline, current)
        self.call_count = 0

    def complete(
        self, messages: list[dict[str, Any]], tools: tuple[dict[str, Any], ...]
    ) -> AssistantTurn:
        self.call_count += 1
        turn = super().complete(messages, tools)
        if self.call_count == 1:
            arguments = dict(turn.tool_calls[0].arguments)
            arguments["assistant_message"] = (
                "QA 已比较 baseline commit 1216dc0 和 scene-test，并读取 manifest。"
            )
            return AssistantTurn(
                content="",
                tool_calls=(
                    ToolCall(
                        call_id="conversation-repair-1",
                        name="reply_to_diagnosis_session",
                        arguments=arguments,
                    ),
                ),
                finish_reason="tool_calls",
                raw_message={"role": "assistant", "tool_calls": []},
            )
        return turn


def _git(repository: Path, *arguments: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return process.stdout.strip()


def _repository(root: Path) -> tuple[str, str]:
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "VisionDoctor Test")
    _git(root, "config", "user.email", "visiondoctor@example.invalid")
    (root / "runner.py").write_text("print('run')\n", encoding="utf-8")
    (root / "test_runner.py").write_text("print('test')\n", encoding="utf-8")
    (root / "vision.py").write_text("ORDER = 'before'\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "working vision")
    baseline = _git(root, "rev-parse", "HEAD")
    (root / "vision.py").write_text("ORDER = 'after'\n", encoding="utf-8")
    _git(root, "add", "vision.py")
    _git(root, "commit", "-q", "-m", "update pose transform")
    return baseline, _git(root, "rev-parse", "HEAD")


def _image_attachment() -> dict[str, str]:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 12), (20, 90, 180)).save(buffer, format="PNG")
    return {
        "name": "现场截图.png",
        "media_type": "image/png",
        "content_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


def test_sessions_are_persistent_and_keep_conversation_sources(tmp_path: Path) -> None:
    baseline, current = _repository(tmp_path / "repository")
    manifest = tmp_path / "manifest.json"
    reference = tmp_path / "reference.json"
    Image.new("RGB", (24, 16), (10, 200, 70)).save(tmp_path / "rgb.png", format="PNG")
    manifest.write_text('{"rgb_path": "rgb.png"}\n', encoding="utf-8")
    reference.write_text("{}\n", encoding="utf-8")
    gateway = ConversationGatewayDouble(baseline, current)
    vision_gateway = VisionGatewayDouble()
    service = DiagnosisSessionService(
        tmp_path / "sessions", gateway=gateway, vision_gateway=vision_gateway
    )

    first = service.create()
    second = service.create()
    service.connect_repository(first["session_id"], str(tmp_path / "repository"))
    service.attach_simulation_capture(
        first["session_id"],
        {
            "case_id": "scene-001",
            "manifest_path": str(manifest),
            "reference_path": str(reference),
        },
    )
    result = service.turn(
        first["session_id"],
        message="升级视觉节点后，机器人每次都向目标右侧偏移。",
        attachments=(_image_attachment(),),
    )

    assert {item["session_id"] for item in service.list()} == {
        first["session_id"],
        second["session_id"],
    }
    assert result["title"] == "目标位姿稳定偏移"
    assert result["readiness"]["can_start"] is True
    assert result["comparison"]["baseline_commit"] == baseline
    user_message = next(item for item in result["messages"] if item["role"] == "user")
    attachment = user_message["attachments"][0]
    path, media_type, filename = service.attachment_path(
        first["session_id"], attachment["attachment_id"]
    )
    assert path.is_file()
    assert media_type == "image/png"
    assert filename == "现场截图.png"
    assert len(vision_gateway.calls) == 2
    assert "latest_image_observations" in str(gateway.messages[1]["content"])
    assert str(reference) not in str(gateway.messages[1]["content"])
    assistant = next(item for item in reversed(result["messages"]) if item["role"] == "assistant")
    assert assistant["image_assessments"][0]["visible_name"] == "现场截图.png"


def test_session_delete_moves_content_to_recoverable_local_trash(tmp_path: Path) -> None:
    service = DiagnosisSessionService(tmp_path / "server" / "sessions")
    session = service.create()
    session_id = str(session["session_id"])
    source = tmp_path / "server" / "sessions" / session_id

    result = service.delete(session_id)

    assert result["deleted"] is True
    assert result["recoverable"] is True
    assert not source.exists()
    recovered = tmp_path / "server" / "session-trash" / result["recovery_id"]
    assert (recovered / "session.json").is_file()
    assert service.list() == ()
    with pytest.raises(KeyError):
        service.get(session_id)


def test_simulation_capture_sync_is_idempotent_per_operation(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    reference = tmp_path / "reference.json"
    Image.new("RGB", (24, 16), (10, 200, 70)).save(tmp_path / "rgb.png", format="PNG")
    manifest.write_text('{"rgb_path": "rgb.png"}\n', encoding="utf-8")
    reference.write_text("{}\n", encoding="utf-8")
    vision_gateway = VisionGatewayDouble()
    service = DiagnosisSessionService(
        tmp_path / "sessions", vision_gateway=vision_gateway
    )
    session = service.create()
    capture = {
        "operation_id": "SIM-capture-001",
        "case_id": "scene-001",
        "manifest_path": str(manifest),
        "reference_path": str(reference),
    }

    first = service.attach_simulation_capture(session["session_id"], capture)
    second = service.attach_simulation_capture(session["session_id"], capture)

    assert len(first["messages"]) == 1
    assert len(second["messages"]) == 1
    assert second["simulation_capture_operation_ids"] == ["SIM-capture-001"]
    assert len(vision_gateway.calls) == 1


def test_user_never_has_to_supply_version_labels(tmp_path: Path) -> None:
    baseline, current = _repository(tmp_path / "repository")
    gateway = ConversationGatewayDouble(baseline, current)
    service = DiagnosisSessionService(tmp_path / "sessions", gateway=gateway)
    session = service.create()
    service.connect_repository(session["session_id"], str(tmp_path / "repository"))

    service.turn(
        session["session_id"],
        message="问题是昨天升级之后出现的，但我不知道具体是哪次代码改动。",
    )

    prompt = str(gateway.messages[0]["content"])
    assert "not expected to know which commit" in prompt
    assert service.get(session["session_id"])["repository"]["commits_differ"] is True


def test_non_python_entrypoint_is_understood_but_not_automatic_execution(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "shell-vision-project"
    repository.mkdir()
    subprocess.run(["git", "-C", str(repository), "init", "-q", "-b", "main"], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "Contract Test"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "config",
            "user.email",
            "contract@example.invalid",
        ],
        check=True,
    )
    (repository / "launch.sh").write_text("#!/bin/sh\necho run\n", encoding="utf-8")
    (repository / "test_runner.py").write_text("print('Ran 1 test')\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-q", "-m", "shell entry"],
        check=True,
    )

    status = inspect_repository(
        str(repository),
        baseline_commit=None,
        faulty_commit=None,
        runner_script="launch.sh",
        test_runner_script="test_runner.py",
    )

    assert status["available"] is True
    assert status["execution_contract"]["runtime"] == "python"
    assert status["execution_contract"]["runner_available"] is False
    assert status["execution_contract"]["supported"] is False
    assert "Python runner" in status["execution_contract"]["reason"]


def test_conversation_cannot_select_a_revision_older_than_project_snapshot(
    tmp_path: Path,
) -> None:
    baseline, _current = _repository(tmp_path / "repository")
    gateway = ConversationGatewayDouble(baseline, baseline)
    service = DiagnosisSessionService(tmp_path / "sessions", gateway=gateway)
    session = service.create()
    service.connect_repository(session["session_id"], str(tmp_path / "repository"))

    with pytest.raises(ModelProtocolError, match="canonical project snapshot"):
        service.turn(
            session["session_id"],
            message="最近的视觉改动后目标位置持续偏移，请检查。",
        )


def test_uncommitted_repository_changes_are_visible_and_block_a_stale_run(
    tmp_path: Path,
) -> None:
    _baseline, _current = _repository(tmp_path / "repository")
    service = DiagnosisSessionService(tmp_path / "sessions")
    session = service.create()
    service.connect_repository(session["session_id"], str(tmp_path / "repository"))
    (tmp_path / "repository" / "vision.py").write_text(
        "ORDER = 'uncommitted'\n", encoding="utf-8"
    )

    result = service.get(session["session_id"])

    clean_check = next(
        item for item in result["readiness"]["checks"] if item["key"] == "clean_snapshot"
    )
    assert result["repository"]["working_tree_dirty"] is True
    assert result["repository"]["working_tree_changes"] == [" M vision.py"]
    assert clean_check["ready"] is False
    assert "已保存的代码版本" in result["readiness"]["missing"]


def test_internal_vocabulary_is_repaired_by_the_model_not_shown_to_user(
    tmp_path: Path,
) -> None:
    baseline, current = _repository(tmp_path / "repository")
    gateway = VocabularyRepairGateway(baseline, current)
    service = DiagnosisSessionService(tmp_path / "sessions", gateway=gateway)
    session = service.create()
    service.connect_repository(session["session_id"], str(tmp_path / "repository"))

    result = service.turn(
        session["session_id"],
        message="升级之后目标位置出现固定偏移，请检查近期改动。",
    )

    assistant = next(item for item in reversed(result["messages"]) if item["role"] == "assistant")
    assert gateway.call_count == 2
    assert "QA" not in assistant["content"]
    assert "1216dc0" not in assistant["content"]
