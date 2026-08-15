from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import httpx

from visiondoctor.llm.settings import ModelSettings


class ModelGatewayError(RuntimeError):
    """A real model request failed; callers must fail closed."""


class ModelProtocolError(ModelGatewayError):
    """The provider returned a response that violates the tool protocol."""


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class AssistantTurn:
    content: str
    tool_calls: tuple[ToolCall, ...]
    finish_reason: str
    raw_message: dict[str, Any]


class ModelGateway(Protocol):
    model: str

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: tuple[dict[str, Any], ...],
    ) -> AssistantTurn: ...


class ModelAuditLog:
    """Append-only metadata log. Prompts, responses, and credentials are never persisted."""

    def __init__(self, path: Path | None) -> None:
        self.path = path.resolve() if path is not None else None
        self._lock = threading.Lock()

    def record(self, payload: dict[str, Any]) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(encoded)


class OpenAICompatibleGateway:
    """Strict OpenAI-compatible chat/tool gateway used by DeepSeek and compatible APIs."""

    def __init__(
        self,
        settings: ModelSettings,
        *,
        audit_path: Path | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.settings = settings
        self.model = settings.model
        self._audit = ModelAuditLog(audit_path)
        self._client = client or httpx.Client(timeout=settings.timeout_s)

    @property
    def endpoint(self) -> str:
        return f"{self.settings.base_url.rstrip('/')}/chat/completions"

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: tuple[dict[str, Any], ...],
    ) -> AssistantTurn:
        request_id = f"MODEL-{uuid.uuid4().hex[:16]}"
        body = {
            "model": self.settings.model,
            "messages": messages,
            "tools": list(tools),
            "tool_choice": "auto",
            "temperature": 0,
            "max_tokens": self.settings.max_tokens,
        }
        request_bytes = json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8")
        started = time.perf_counter()
        try:
            response = self._client.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self.settings.api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
        except httpx.HTTPError as exc:
            self._audit_failure(request_id, request_bytes, started, type(exc).__name__)
            raise ModelGatewayError(f"model request failed: {type(exc).__name__}") from exc
        if response.status_code < 200 or response.status_code >= 300:
            self._audit_failure(request_id, request_bytes, started, f"HTTP {response.status_code}")
            raise ModelGatewayError(f"model provider returned HTTP {response.status_code}")
        try:
            payload = response.json()
            choice = payload["choices"][0]
            message = choice["message"]
            raw_tool_calls = message.get("tool_calls") or []
            tool_calls: list[ToolCall] = []
            for item in raw_tool_calls:
                function = item["function"]
                arguments = json.loads(function.get("arguments") or "{}")
                if not isinstance(arguments, dict):
                    raise TypeError("tool arguments are not an object")
                tool_calls.append(
                    ToolCall(
                        call_id=str(item["id"]),
                        name=str(function["name"]),
                        arguments=arguments,
                    )
                )
            turn = AssistantTurn(
                content=str(message.get("content") or ""),
                tool_calls=tuple(tool_calls),
                finish_reason=str(choice.get("finish_reason") or ""),
                raw_message={
                    "role": "assistant",
                    "content": message.get("content"),
                    "tool_calls": raw_tool_calls,
                },
            )
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._audit_failure(request_id, request_bytes, started, "invalid_response")
            raise ModelProtocolError("model provider returned an invalid tool response") from exc
        response_bytes = response.content
        usage = payload.get("usage") or {}
        self._audit.record(
            {
                "request_id": request_id,
                "created_at": datetime.now(UTC).isoformat(),
                "provider_endpoint": self.endpoint,
                "model": self.model,
                "duration_s": round(time.perf_counter() - started, 6),
                "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
                "response_sha256": hashlib.sha256(response_bytes).hexdigest(),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "finish_reason": turn.finish_reason,
                "tool_names": [call.name for call in turn.tool_calls],
                "status": "succeeded",
            }
        )
        return turn

    def _audit_failure(
        self, request_id: str, request_bytes: bytes, started: float, error: str
    ) -> None:
        self._audit.record(
            {
                "request_id": request_id,
                "created_at": datetime.now(UTC).isoformat(),
                "provider_endpoint": self.endpoint,
                "model": self.model,
                "duration_s": round(time.perf_counter() - started, 6),
                "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
                "status": "failed",
                "error": error,
            }
        )
