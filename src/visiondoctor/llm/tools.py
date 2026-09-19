from __future__ import annotations

from typing import Any


def _tool(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": parameters},
    }


def terminal_tool(name: str, description: str, properties: dict[str, Any], required: list[str]):
    return _tool(
        name,
        description,
        {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    )
