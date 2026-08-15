from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from visiondoctor.schemas import (
    ArtifactRef,
    CaseEvidence,
    MetricResult,
    TaskKind,
    TaskSpecification,
)


@dataclass(frozen=True)
class TaskCaseEvaluation:
    case_id: str
    contract_valid: bool
    passed: bool
    failures: tuple[str, ...] = ()
    measurements: dict[str, float] = field(default_factory=dict)
    aggregate_data: dict[str, Any] = field(default_factory=dict)


class TaskValidatorPlugin(ABC):
    kind: TaskKind

    @abstractmethod
    def validate_input(self, value: Any, artifacts: tuple[ArtifactRef, ...]) -> None: ...

    @abstractmethod
    def validate_reference(self, expected: Any) -> None: ...

    @abstractmethod
    def evaluate_case(
        self,
        case_id: str,
        actual: Any,
        expected: Any,
        evidence: CaseEvidence,
        specification: TaskSpecification,
    ) -> TaskCaseEvaluation: ...

    @abstractmethod
    def aggregate(
        self,
        evaluations: tuple[TaskCaseEvaluation, ...],
        specification: TaskSpecification,
    ) -> tuple[MetricResult, ...]: ...

    @abstractmethod
    def capability(self) -> dict[str, Any]: ...


def metric(
    name: str,
    value: float | bool,
    threshold: float | bool,
    comparator: str,
    passed: bool,
    unit: str | None = None,
) -> MetricResult:
    return MetricResult(
        name=name,
        value=value,
        threshold=threshold,
        comparator=comparator,  # type: ignore[arg-type]
        passed=passed,
        unit=unit,
    )


def finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def image_input(
    value: Any, artifacts: tuple[ArtifactRef, ...]
) -> tuple[ArtifactRef, int, int]:
    if not isinstance(value, dict):
        raise ValueError("visual task input must be an object")
    artifact_id = value.get("image_artifact_id")
    if not isinstance(artifact_id, str) or not artifact_id.strip():
        raise ValueError("visual task input requires image_artifact_id")
    matches = [
        artifact
        for artifact in artifacts
        if artifact.artifact_id == artifact_id
        or artifact.artifact_id.endswith(f":{artifact_id}")
    ]
    if len(matches) != 1:
        raise ValueError("image_artifact_id must identify exactly one input artifact")
    artifact = matches[0]
    if artifact.media_type not in {"image/png", "image/jpeg", "image/webp"}:
        raise ValueError("visual task artifact must be PNG, JPEG, or WebP")
    try:
        with Image.open(Path(artifact.path)) as image:
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > 24_000_000:
                raise ValueError(
                    "visual task image dimensions are outside the supported bound"
                )
            image.verify()
    except (OSError, ValueError) as exc:
        raise ValueError("visual task artifact is not a decodable image") from exc
    declared_width = value.get("width")
    declared_height = value.get("height")
    if declared_width is not None and declared_width != width:
        raise ValueError("declared image width does not match the artifact")
    if declared_height is not None and declared_height != height:
        raise ValueError("declared image height does not match the artifact")
    return artifact, width, height
