from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from PIL import Image

from visiondoctor.llm import AssistantTurn, ToolCall
from visiondoctor.multimodal import (
    OpenAIVisionGateway,
    VisionModelError,
    VisionModelProtocolError,
    VisionSettings,
)


def _settings() -> VisionSettings:
    return VisionSettings(
        base_url="https://vision.test/v1",
        model="vision-test-model",
        api_key="vision-test-key",
    )


def _png_attachment(name: str, color: tuple[int, int, int]) -> dict[str, str]:
    buffer = io.BytesIO()
    Image.new("RGB", (18, 12), color).save(buffer, format="PNG")
    return {
        "name": name,
        "media_type": "image/png",
        "content_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


class _ConversationGateway:
    model = "conversation-test-model"

    def __init__(self) -> None:
        self.calls = 0

    def complete(
        self, messages: list[dict[str, Any]], tools: tuple[dict[str, Any], ...]
    ) -> AssistantTurn:
        del messages, tools
        self.calls += 1
        return AssistantTurn(
            content="",
            tool_calls=(
                ToolCall(
                    call_id="conversation-1",
                    name="reply_to_diagnosis_session",
                    arguments={
                        "assistant_message": "我已经逐张查看图片，并会结合这些线索继续诊断。",
                        "title": "图片辅助诊断",
                        "questions": [],
                        "next_actions": [],
                        "repository_choice": {
                            "baseline_commit": "",
                            "current_commit": "",
                            "reason": "",
                        },
                        "execution_contract": {
                            "runner_script": "",
                            "test_runner_script": "",
                        },
                        "attachment_assessments": [],
                    },
                ),
            ),
            finish_reason="tool_calls",
            raw_message={"role": "assistant", "tool_calls": []},
        )


class _VisionGateway:
    model = "vision-test-model"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.names: list[str] = []

    def assess(
        self,
        image_path: Path,
        *,
        attachment_id: str,
        visible_name: str,
        user_context: str,
    ) -> dict[str, Any]:
        del user_context
        if self.fail:
            raise VisionModelError("视觉模型不可用")
        with Image.open(image_path) as image:
            assert image.format == "PNG"
        self.names.append(visible_name)
        return {
            "attachment_id": attachment_id,
            "visible_name": visible_name,
            "observations": [f"已直接查看 {visible_name} 的像素。"],
            "diagnostic_relevance": "可作为诊断线索。",
            "limitations": [],
            "confidence": 0.8,
            "model": self.model,
        }


def test_ollama_gateway_sends_pixels_and_requires_structured_observation(tmp_path: Path) -> None:
    image_path = tmp_path / "model-preview.png"
    Image.new("RGB", (8, 6), (20, 180, 90)).save(image_path, format="PNG")

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer vision-test-key"
        assert body["model"] == "vision-test-model"
        assert body["response_format"] == {"type": "json_object"}
        parts = body["messages"][1]["content"]
        assert [item["type"] for item in parts] == ["text", "image_url"]
        prefix, encoded = parts[1]["image_url"]["url"].split(",", maxsplit=1)
        assert prefix == "data:image/png;base64"
        assert base64.b64decode(encoded)[:4] == bytes([0x89, 0x50, 0x4E, 0x47])
        content = json.dumps(
            {
                "observations": ["画面中央有绿色目标。"],
                "diagnostic_relevance": "目标位置可与用户描述比较。",
                "limitations": ["缺少深度。"],
                "confidence": 0.91,
            },
            ensure_ascii=False,
        )
        return httpx.Response(
            200, json={"choices": [{"message": {"content": content}}]}
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    gateway = OpenAIVisionGateway(_settings(), client=client)
    result = gateway.assess(
        image_path,
        attachment_id="ATT-123456789abc",
        visible_name="现场.png",
        user_context="机器人向右偏移",
    )

    assert result["attachment_id"] == "ATT-123456789abc"
    assert result["visible_name"] == "现场.png"
    assert result["observations"] == ["画面中央有绿色目标。"]


def test_ollama_gateway_rejects_incomplete_response_without_fallback(tmp_path: Path) -> None:
    image_path = tmp_path / "model-preview.png"
    Image.new("RGB", (8, 6)).save(image_path, format="PNG")
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": '{"observations": []}'}}]
                },
            )
        )
    )
    gateway = OpenAIVisionGateway(_settings(), client=client)

    with pytest.raises(VisionModelProtocolError):
        gateway.assess(
            image_path,
            attachment_id="ATT-123456789abc",
            visible_name="现场.png",
            user_context="",
        )
