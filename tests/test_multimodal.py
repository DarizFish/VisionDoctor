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
    OllamaVisionGateway,
    VisionModelError,
    VisionModelProtocolError,
    VisionSettings,
)
from visiondoctor.sessions import DiagnosisSessionService


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
        assert request.url.path == "/api/chat"
        assert body["model"] == "qwen3-vl:4b"
        assert body["stream"] is False
        assert len(body["messages"][1]["images"]) == 1
        assert base64.b64decode(body["messages"][1]["images"][0]).startswith(b"\x89PNG")
        assert body["format"]["required"] == [
            "observations",
            "diagnostic_relevance",
            "limitations",
            "confidence",
        ]
        return httpx.Response(
            200,
            json={
                "message": {
                    "content": json.dumps(
                        {
                            "observations": ["画面中央有绿色目标。"],
                            "diagnostic_relevance": "目标位置可与用户描述比较。",
                            "limitations": ["缺少深度。"],
                            "confidence": 0.91,
                        },
                        ensure_ascii=False,
                    )
                }
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    gateway = OllamaVisionGateway(
        VisionSettings(model="qwen3-vl:4b"), client=client
    )
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
                200, json={"message": {"content": '{"observations": []}'}}
            )
        )
    )
    gateway = OllamaVisionGateway(VisionSettings(), client=client)

    with pytest.raises(VisionModelProtocolError):
        gateway.assess(
            image_path,
            attachment_id="ATT-123456789abc",
            visible_name="现场.png",
            user_context="",
        )


def test_image_format_pixel_limit_and_exif_are_enforced() -> None:
    png = base64.b64decode(_png_attachment("x.png", (1, 2, 3))["content_base64"])
    with pytest.raises(ValueError, match="declared image type"):
        DiagnosisSessionService._prepare_model_image(
            png, declared_media_type="image/jpeg"
        )

    oversized = io.BytesIO()
    Image.new("1", (6000, 5000)).save(oversized, format="PNG")
    with pytest.raises(ValueError, match="safe pixel limit"):
        DiagnosisSessionService._prepare_model_image(
            oversized.getvalue(), declared_media_type="image/png"
        )

    source = io.BytesIO()
    image = Image.new("RGB", (10, 20), (40, 90, 180))
    exif = Image.Exif()
    exif[274] = 6
    image.save(source, format="JPEG", exif=exif)
    preview, metadata = DiagnosisSessionService._prepare_model_image(
        source.getvalue(), declared_media_type="image/jpeg"
    )
    with Image.open(io.BytesIO(preview)) as normalized:
        assert normalized.size == (20, 10)
        assert not normalized.getexif()
    assert metadata["exif_orientation_applied"] is True


def test_multiple_images_are_all_assessed_and_failure_never_calls_text_model(
    tmp_path: Path,
) -> None:
    conversation = _ConversationGateway()
    vision = _VisionGateway()
    service = DiagnosisSessionService(
        tmp_path / "sessions", gateway=conversation, vision_gateway=vision
    )
    session = service.create()
    result = service.turn(
        session["session_id"],
        message="比较这两张现场图。",
        attachments=(
            _png_attachment("左侧相机.png", (250, 10, 10)),
            _png_attachment("右侧相机.png", (10, 10, 250)),
        ),
    )
    assistant = result["messages"][-1]
    assert vision.names == ["左侧相机.png", "右侧相机.png"]
    assert len(assistant["image_assessments"]) == 2
    assert conversation.calls == 1

    failed_conversation = _ConversationGateway()
    failed_service = DiagnosisSessionService(
        tmp_path / "failed-sessions",
        gateway=failed_conversation,
        vision_gateway=_VisionGateway(fail=True),
    )
    failed = failed_service.create()
    with pytest.raises(VisionModelError):
        failed_service.turn(
            failed["session_id"],
            message="看这张图。",
            attachments=(_png_attachment("失败.png", (1, 2, 3)),),
        )
    assert failed_conversation.calls == 0
