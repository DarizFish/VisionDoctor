from __future__ import annotations

from typing import Any

from visiondoctor.schemas import TaskKind
from visiondoctor.tasks.validators.base import TaskValidatorPlugin
from visiondoctor.tasks.validators.detection import DetectionValidator
from visiondoctor.tasks.validators.exact import ExactStructuredValidator
from visiondoctor.tasks.validators.ocr import OcrValidator
from visiondoctor.tasks.validators.segmentation import SegmentationValidator

_VALIDATORS: dict[TaskKind, TaskValidatorPlugin] = {
    TaskKind.STRUCTURED_OUTPUT: ExactStructuredValidator(),
    TaskKind.DETECTION: DetectionValidator(),
    TaskKind.OCR: OcrValidator(),
    TaskKind.SEGMENTATION: SegmentationValidator(),
}


def get_validator_plugin(kind: TaskKind | str) -> TaskValidatorPlugin:
    normalized = TaskKind(kind)
    try:
        return _VALIDATORS[normalized]
    except KeyError as exc:
        raise ValueError(f"task validator is not implemented: {normalized}") from exc


def public_validator_capabilities() -> tuple[dict[str, Any], ...]:
    return tuple(
        _VALIDATORS[kind].capability()
        for kind in (TaskKind.DETECTION, TaskKind.OCR, TaskKind.SEGMENTATION)
    )


__all__ = [
    "TaskValidatorPlugin",
    "get_validator_plugin",
    "public_validator_capabilities",
]
