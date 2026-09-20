"""Compare what a program read, wrote and declared, by content rather than by bytes.

Two configuration files that differ only in line endings are the same
configuration; two outputs whose run identifiers differ are not thereby
different results.  Identity and time fields are left out of the comparison and
counted, so the omission stays visible.
"""

from __future__ import annotations

import json
from typing import Any

import yaml

#: Fields that name a run, a record or a moment rather than a result.
IDENTITY_KEYS = frozenset({
    "run_id", "detection_id", "capture_id", "frame_id", "timestamp", "source_stamp_s",
})
IDENTITY_PARENTS = frozenset({"source_hashes", "config_hashes"})
LIMIT = 24


def parse(reference: str, payload: bytes) -> Any:
    """Structured content, or ``None`` for anything that is not a single document."""

    text = payload.decode("utf-8")
    if reference.endswith((".yaml", ".yml")):
        return yaml.safe_load(text)
    if reference.endswith(".json"):
        return json.loads(text)
    return None


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        flat: dict[str, Any] = {}
        for key, item in value.items():
            flat.update(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return flat
    if isinstance(value, list):
        flat = {}
        for index, item in enumerate(value):
            flat.update(_flatten(item, f"{prefix}[{index}]"))
        return flat
    return {prefix: value}


def _identity(path: str) -> bool:
    parts = path.replace("[", ".").split(".")
    return parts[-1] in IDENTITY_KEYS or any(part in IDENTITY_PARENTS for part in parts)


def differences(baseline: Any, candidate: Any, *, tolerance: float = 1e-9) -> dict[str, Any]:
    """Field-level differences; numbers within ``tolerance`` count as equal."""

    left, right = _flatten(baseline), _flatten(candidate)
    omitted = sorted({key for key in (*left, *right) if _identity(key)})
    changed: list[dict[str, Any]] = []
    for key in sorted(set(left) | set(right)):
        if _identity(key):
            continue
        a, b = left.get(key, "<absent>"), right.get(key, "<absent>")
        numeric = all(isinstance(v, int | float) and not isinstance(v, bool) for v in (a, b))
        if numeric and abs(float(a) - float(b)) <= tolerance:
            continue
        if not numeric and a == b:
            continue
        row: dict[str, Any] = {"field": key, "baseline": a, "candidate": b}
        if numeric:
            row["delta"] = round(float(b) - float(a), 9)
        changed.append(row)
    return {
        "same": not changed,
        "differences": changed[:LIMIT],
        "truncated": len(changed) > LIMIT,
        "identity_fields_omitted": len(omitted),
    }
