from __future__ import annotations

import base64
import io
import json
import stat
import zipfile
from pathlib import Path
from typing import Any

import pytest
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, StreamObject

from visiondoctor.attachments import AttachmentAnalysisError, analyze_attachment
from visiondoctor.llm import AssistantTurn, ModelProtocolError, ToolCall
from visiondoctor.sessions import DiagnosisSessionService


class _ConversationGateway:
    model = "conversation-test-model"

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.calls = 0

    def complete(
        self, messages: list[dict[str, Any]], tools: tuple[dict[str, Any], ...]
    ) -> AssistantTurn:
        del tools
        self.messages = messages
        self.calls += 1
        envelope = json.loads(str(messages[1]["content"]).split("\n", 1)[1])
        latest_user = next(
            item
            for item in reversed(envelope["conversation"])
            if item["role"] == "user"
        )
        document_ids = [
            item["attachment_id"]
            for item in latest_user["attachments"]
            if not str(item["media_type"]).startswith("image/")
        ]
        return AssistantTurn(
            content="",
            tool_calls=(
                ToolCall(
                    call_id=f"conversation-{self.calls}",
                    name="reply_to_diagnosis_session",
                    arguments={
                        "assistant_message": "我已经读取附件中的实际内容，并会据此继续诊断。",
                        "title": "附件辅助诊断",
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
                        "attachment_assessments": [
                            {
                                "attachment_id": attachment_id,
                                "observations": ["附件记录了可用于诊断的具体内容。"],
                                "diagnostic_relevance": "这些内容与当前故障现象有关。",
                                "limitations": ["结论仍需结合代码与运行结果确认。"],
                            }
                            for attachment_id in document_ids
                        ],
                    },
                ),
            ),
            finish_reason="tool_calls",
            raw_message={"role": "assistant", "tool_calls": []},
        )


class _AssessmentProtocolGateway(_ConversationGateway):
    def __init__(self, outcomes: list[str]) -> None:
        super().__init__()
        self.outcomes = outcomes

    def complete(
        self, messages: list[dict[str, Any]], tools: tuple[dict[str, Any], ...]
    ) -> AssistantTurn:
        turn = super().complete(messages, tools)
        arguments = dict(turn.tool_calls[0].arguments)
        outcome = self.outcomes[min(self.calls - 1, len(self.outcomes) - 1)]
        if outcome == "missing":
            arguments["attachment_assessments"] = []
        elif outcome == "unknown":
            arguments["attachment_assessments"][0]["attachment_id"] = (
                "ATT-ffffffffffff"
            )
        return AssistantTurn(
            content="",
            tool_calls=(
                ToolCall(
                    call_id=f"protocol-{self.calls}",
                    name="reply_to_diagnosis_session",
                    arguments=arguments,
                ),
            ),
            finish_reason="tool_calls",
            raw_message={"role": "assistant", "tool_calls": []},
        )


def _attachment(name: str, media_type: str, payload: bytes) -> dict[str, str]:
    return {
        "name": name,
        "media_type": media_type,
        "content_base64": base64.b64encode(payload).decode("ascii"),
    }


def _text_pdf(text: str) -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=200)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_reference = writer._add_object(font)
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): font_reference}
            )
        }
    )
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    content = StreamObject()
    content.set_data(f"BT /F1 12 Tf 20 100 Td ({escaped}) Tj ET".encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(content)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def _zip(entries: dict[str, bytes], *, compression: int = zipfile.ZIP_DEFLATED) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=compression) as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


def test_text_and_json_are_strictly_parsed_and_bounded() -> None:
    text = analyze_attachment("camera.log", "text/plain", "偏移 3mm\n".encode())
    assert text.text == "偏移 3mm\n"
    assert text.metadata["kind"] == "log"
    assert text.metadata["status"] == "ready"

    structured = analyze_attachment(
        "config.json", "application/json", b'{"z": 1, "a": {"enabled": true}}'
    )
    assert structured.text.index('"a"') < structured.text.index('"z"')
    assert structured.metadata["kind"] == "json"

    with pytest.raises(AttachmentAnalysisError, match="重复字段"):
        analyze_attachment("bad.json", "application/json", b'{"x": 1, "x": 2}')
    with pytest.raises(AttachmentAnalysisError, match="UTF-8"):
        analyze_attachment("binary.log", "text/plain", b"\xff\x00\xfe")

    long_text = analyze_attachment("long.log", "text/plain", b"A" * 50_000)
    assert long_text.metadata["status"] == "partial"
    assert long_text.metadata["truncated"] is True
    assert "中间内容" in long_text.text
    assert len(long_text.text) == 40_000


def test_pdf_extracts_searchable_text_and_rejects_unreadable_pages() -> None:
    result = analyze_attachment(
        "report.pdf", "application/pdf", _text_pdf("camera target offset 3 mm")
    )
    assert "camera target offset 3 mm" in result.text
    assert result.metadata["page_count"] == 1
    assert result.metadata["status"] == "ready"
    assert "不分析页面图形" in result.metadata["limitations"][0]

    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    blank = io.BytesIO()
    writer.write(blank)
    with pytest.raises(AttachmentAnalysisError, match="没有可搜索文字"):
        analyze_attachment("scan.pdf", "application/pdf", blank.getvalue())
    with pytest.raises(AttachmentAnalysisError, match="类型不匹配"):
        analyze_attachment("fake.pdf", "application/pdf", b"not a pdf")


def test_zip_reads_only_bounded_safe_text_entries() -> None:
    payload = _zip(
        {
            "logs/camera.log": b"pose offset\n",
            "config/settings.json": b'{"frame": "camera"}',
            "images/frame.png": b"not inspected as zip text",
        }
    )
    result = analyze_attachment("bundle.zip", "application/zip", payload)
    assert "logs/camera.log" in result.text
    assert "pose offset" in result.text
    assert '"frame": "camera"' in result.text
    assert result.metadata["analyzed_entry_count"] == 2
    assert result.metadata["skipped_entry_count"] == 1
    assert result.metadata["status"] == "partial"
    assert "images/frame.png" in result.metadata["limitations"][0]

    with pytest.raises(AttachmentAnalysisError, match="不安全路径"):
        analyze_attachment(
            "traversal.zip",
            "application/zip",
            _zip({"../outside.log": b"unsafe"}),
        )
    with pytest.raises(AttachmentAnalysisError, match="没有支持的文本文件"):
        analyze_attachment(
            "binary.zip", "application/zip", _zip({"frame.png": b"pixels"})
        )


def test_zip_rejects_symlinks_and_compression_bombs() -> None:
    symlink_buffer = io.BytesIO()
    with zipfile.ZipFile(symlink_buffer, mode="w") as archive:
        entry = zipfile.ZipInfo("link.log")
        entry.create_system = 3
        entry.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(entry, "target.log")
    with pytest.raises(AttachmentAnalysisError, match="符号链接"):
        analyze_attachment(
            "links.zip", "application/zip", symlink_buffer.getvalue()
        )

    bomb = _zip({"huge.log": b"0" * 2_000_000})
    with pytest.raises(AttachmentAnalysisError, match="压缩比例"):
        analyze_attachment("bomb.zip", "application/zip", bomb)


def test_session_sends_real_attachment_content_and_verifies_integrity(tmp_path: Path) -> None:
    gateway = _ConversationGateway()
    service = DiagnosisSessionService(tmp_path / "sessions", gateway=gateway)
    session = service.create()
    result = service.turn(
        session["session_id"],
        message="请结合日志分析固定偏移。",
        attachments=(
            _attachment(
                "camera.log",
                "text/plain",
                b"frame=camera\ntarget_x=0.314\nerror=stable_right_shift\n",
            ),
        ),
    )

    trusted_context = str(gateway.messages[1]["content"])
    assert "target_x=0.314" in trusted_context
    assert "stable_right_shift" in trusted_context
    user_message = next(item for item in result["messages"] if item["role"] == "user")
    saved = user_message["attachments"][0]
    assert saved["content_analysis"]["status"] == "ready"
    assert "纳入诊断" in saved["content_analysis"]["description"]

    analysis_path = (
        service._root(session["session_id"])
        / saved["content_analysis_relative_path"]
    )
    analysis_path.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="完整性检查失败"):
        service.turn(session["session_id"], message="继续分析。")
    assert gateway.calls == 1


def test_turn_text_budget_fails_closed_and_removes_partial_files(tmp_path: Path) -> None:
    service = DiagnosisSessionService(tmp_path / "sessions", gateway=_ConversationGateway())
    session = service.create()
    attachment = _attachment("large.log", "text/plain", b"A" * 50_000)

    with pytest.raises(ValueError, match="拆成多轮"):
        service._save_attachments(
            session["session_id"], (attachment, attachment, attachment)
        )

    assert list((service._root(session["session_id"]) / "attachments").iterdir()) == []


def _assessment(
    attachment_id: str, *, chinese: bool = True
) -> dict[str, Any]:
    return {
        "attachment_id": attachment_id,
        "observations": ["日志记录了稳定偏移。" if chinese else "stable offset"],
        "diagnostic_relevance": "这与当前问题直接相关。" if chinese else "relevant",
        "limitations": ["仍需结合运行结果确认。" if chinese else "needs runtime"],
    }


def test_attachment_assessment_requires_exact_current_document_mapping() -> None:
    first_id = "ATT-111111111111"
    second_id = "ATT-222222222222"
    image_id = "ATT-333333333333"
    attachments = [
        {"attachment_id": first_id, "name": "一.log", "media_type": "text/plain"},
        {"attachment_id": second_id, "name": "二.json", "media_type": "application/json"},
        {"attachment_id": image_id, "name": "图.png", "media_type": "image/png"},
    ]

    valid = DiagnosisSessionService._validate_attachment_assessments(
        [_assessment(second_id), _assessment(first_id)], attachments
    )
    assert [item["visible_name"] for item in valid] == ["一.log", "二.json"]

    invalid_values = (
        [_assessment(first_id)],
        [_assessment(first_id), _assessment("ATT-ffffffffffff")],
        [_assessment(first_id), _assessment(first_id)],
        [_assessment(first_id), _assessment(image_id)],
        [_assessment(first_id), _assessment(second_id, chinese=False)],
    )
    for value in invalid_values:
        with pytest.raises(ModelProtocolError):
            DiagnosisSessionService._validate_attachment_assessments(
                value, attachments
            )


def test_attachment_assessment_retries_once_then_succeeds(tmp_path: Path) -> None:
    gateway = _AssessmentProtocolGateway(["missing", "valid"])
    service = DiagnosisSessionService(tmp_path / "sessions", gateway=gateway)
    session = service.create()

    result = service.turn(
        session["session_id"],
        message="请分析这份日志。",
        attachments=(_attachment("camera.log", "text/plain", b"offset=3mm\n"),),
    )

    assert gateway.calls == 2
    assistant = result["messages"][-1]
    assert assistant["attachment_assessments"][0]["visible_name"] == "camera.log"
    assert "attachment_id" in assistant["attachment_assessments"][0]


def test_attachment_assessment_fails_closed_after_two_invalid_responses(
    tmp_path: Path,
) -> None:
    gateway = _AssessmentProtocolGateway(["missing", "missing"])
    service = DiagnosisSessionService(tmp_path / "sessions", gateway=gateway)
    session = service.create()

    with pytest.raises(ModelProtocolError, match="未按要求完成每个附件的分析"):
        service.turn(
            session["session_id"],
            message="请分析这份日志。",
            attachments=(_attachment("camera.log", "text/plain", b"offset=3mm\n"),),
        )

    assert gateway.calls == 2
    assert service.transcript(session["session_id"]) == ()
    assert list((service._root(session["session_id"]) / "attachments").iterdir()) == []
