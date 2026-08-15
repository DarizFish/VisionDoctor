from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse


class ModelConfigurationError(RuntimeError):
    """Raised when a real model cannot be configured safely."""


_ENV_KEYS = frozenset(
    {
        "VISIONDOCTOR_LLM_API_KEY",
        "VISIONDOCTOR_LLM_BASE_URL",
        "VISIONDOCTOR_LLM_MODEL",
        "VISIONDOCTOR_LLM_TIMEOUT_S",
        "VISIONDOCTOR_LLM_MAX_TOKENS",
        "VISIONDOCTOR_LLM_MAX_TOOL_ITERATIONS",
    }
)


def _load_env_file() -> None:
    """Load only the model allowlist from .env without overriding process variables."""
    configured = os.getenv("VISIONDOCTOR_ENV_FILE")
    path = Path(configured).expanduser() if configured else Path.cwd() / ".env"
    if not path.is_file():
        return
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ModelConfigurationError(f"invalid .env assignment at line {line_number}")
        key, value = (part.strip() for part in line.split("=", maxsplit=1))
        if key not in _ENV_KEYS:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class ModelSettings:
    api_key: str = field(repr=False)
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-flash"
    timeout_s: float = 120.0
    max_tokens: int = 8192
    max_tool_iterations: int = 12

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise ModelConfigurationError("VISIONDOCTOR_LLM_API_KEY is required")
        parsed = urlparse(self.base_url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ModelConfigurationError("VISIONDOCTOR_LLM_BASE_URL must be an HTTPS URL")
        if not self.model.strip():
            raise ModelConfigurationError("VISIONDOCTOR_LLM_MODEL is required")
        if self.timeout_s <= 0 or self.max_tokens < 256 or self.max_tool_iterations < 2:
            raise ModelConfigurationError("invalid model timeout, token, or tool-loop limits")

    @classmethod
    def from_environment(cls) -> ModelSettings:
        _load_env_file()
        key = os.getenv("VISIONDOCTOR_LLM_API_KEY", "")
        if not key:
            raise ModelConfigurationError(
                "real model credentials are missing; set VISIONDOCTOR_LLM_API_KEY"
            )
        return cls(
            api_key=key,
            base_url=os.getenv("VISIONDOCTOR_LLM_BASE_URL", "https://api.deepseek.com"),
            model=os.getenv("VISIONDOCTOR_LLM_MODEL", "deepseek-v4-flash"),
            timeout_s=float(os.getenv("VISIONDOCTOR_LLM_TIMEOUT_S", "120")),
            max_tokens=int(os.getenv("VISIONDOCTOR_LLM_MAX_TOKENS", "8192")),
            max_tool_iterations=int(
                os.getenv("VISIONDOCTOR_LLM_MAX_TOOL_ITERATIONS", "12")
            ),
        )

    def public_summary(self) -> dict[str, str | float | int | bool]:
        return {
            "configured": True,
            "base_url": self.base_url,
            "model": self.model,
            "timeout_s": self.timeout_s,
            "max_tokens": self.max_tokens,
            "max_tool_iterations": self.max_tool_iterations,
        }
