from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class EvidenceStore:
    def __init__(self, run_root: Path) -> None:
        self.run_root = run_root.resolve()
        self.run_root.mkdir(parents=True, exist_ok=False)

    def write_model(self, relative_path: str, model: BaseModel) -> Path:
        return self.write_json(relative_path, model.model_dump(mode="json"))

    def write_json(self, relative_path: str, value: Any) -> Path:
        path = self._resolve(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
        self._atomic_write(path, payload + "\n")
        return path

    def write_text(self, relative_path: str, value: str) -> Path:
        path = self._resolve(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write(path, value)
        return path

    def sha256(self, relative_path: str) -> str:
        return hashlib.sha256(self._resolve(relative_path).read_bytes()).hexdigest()

    def _resolve(self, relative_path: str) -> Path:
        path = (self.run_root / relative_path).resolve()
        if path != self.run_root and self.run_root not in path.parents:
            raise ValueError("artifact path escaped the run root")
        return path

    @staticmethod
    def _atomic_write(path: Path, value: str) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(value, encoding="utf-8")
        temporary.replace(path)


class TraceRecorder:
    def __init__(self, store: EvidenceStore) -> None:
        self.path = store.run_root / "trace.jsonl"

    def record(self, event: str, **data: Any) -> None:
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            **data,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
