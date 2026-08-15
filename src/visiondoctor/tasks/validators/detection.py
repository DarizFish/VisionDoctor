from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

from visiondoctor.schemas import ArtifactRef, CaseEvidence, TaskKind, TaskSpecification
from visiondoctor.tasks.validators.base import (
    TaskCaseEvaluation,
    TaskValidatorPlugin,
    finite_number,
    image_input,
    metric,
)


@dataclass(frozen=True)
class _Detection:
    label: str
    bbox: tuple[float, float, float, float]
    score: float


class DetectionValidator(TaskValidatorPlugin):
    kind = TaskKind.DETECTION

    def validate_input(self, value: Any, artifacts: tuple[ArtifactRef, ...]) -> None:
        image_input(value, artifacts)

    def validate_reference(self, expected: Any) -> None:
        self._parse_reference(expected)

    def evaluate_case(
        self,
        case_id: str,
        actual: Any,
        expected: Any,
        evidence: CaseEvidence,
        specification: TaskSpecification,
    ) -> TaskCaseEvaluation:
        try:
            if evidence.structured_input is None:
                raise ValueError("detection evidence omitted task input")
            _artifact, width, height = image_input(
                evidence.structured_input.value, evidence.input_artifacts
            )
            predictions = self._parse_predictions(actual, width=width, height=height)
            references = self._parse_reference(expected, width=width, height=height)
        except (KeyError, TypeError, ValueError) as exc:
            return TaskCaseEvaluation(
                case_id=case_id,
                contract_valid=False,
                passed=False,
                failures=(f"detection output contract invalid: {exc}",),
            )
        kept = tuple(
            item
            for item in predictions
            if item.score + 1e-12 >= specification.detection_score_threshold
        )
        matches = self._match(
            kept, references, iou_threshold=specification.detection_iou_threshold
        )
        true_positive = len(matches)
        false_positive = len(kept) - true_positive
        false_negative = len(references) - true_positive
        precision = self._ratio(true_positive, true_positive + false_positive)
        recall = self._ratio(true_positive, true_positive + false_negative)
        mean_iou = (
            statistics.fmean(item[2] for item in matches)
            if matches
            else 1.0 if not kept and not references else 0.0
        )
        failures: list[str] = []
        if precision + 1e-12 < specification.detection_precision_threshold:
            failures.append(
                f"detection precision={precision:.6f} below "
                f"{specification.detection_precision_threshold:.6f}"
            )
        if recall + 1e-12 < specification.detection_recall_threshold:
            failures.append(
                f"detection recall={recall:.6f} below "
                f"{specification.detection_recall_threshold:.6f}"
            )
        if mean_iou + 1e-12 < specification.detection_iou_threshold:
            failures.append(
                f"detection mean_iou={mean_iou:.6f} below "
                f"{specification.detection_iou_threshold:.6f}"
            )
        return TaskCaseEvaluation(
            case_id=case_id,
            contract_valid=True,
            passed=not failures,
            failures=tuple(failures),
            measurements={
                "precision": precision,
                "recall": recall,
                "mean_iou": mean_iou,
            },
            aggregate_data={
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "matched_ious": [item[2] for item in matches],
            },
        )

    def aggregate(
        self,
        evaluations: tuple[TaskCaseEvaluation, ...],
        specification: TaskSpecification,
    ) -> tuple:
        valid = tuple(item for item in evaluations if item.contract_valid)
        true_positive = sum(int(item.aggregate_data["true_positive"]) for item in valid)
        false_positive = sum(int(item.aggregate_data["false_positive"]) for item in valid)
        false_negative = sum(int(item.aggregate_data["false_negative"]) for item in valid)
        matched_ious = [
            float(value)
            for item in valid
            for value in item.aggregate_data["matched_ious"]
        ]
        precision = self._ratio(true_positive, true_positive + false_positive)
        recall = self._ratio(true_positive, true_positive + false_negative)
        mean_iou = statistics.fmean(matched_ious) if matched_ious else 0.0
        if true_positive == false_positive == false_negative == 0:
            mean_iou = 1.0
        return (
            metric(
                "detection_precision",
                precision,
                specification.detection_precision_threshold,
                ">=",
                precision + 1e-12 >= specification.detection_precision_threshold,
                "ratio",
            ),
            metric(
                "detection_recall",
                recall,
                specification.detection_recall_threshold,
                ">=",
                recall + 1e-12 >= specification.detection_recall_threshold,
                "ratio",
            ),
            metric(
                "detection_mean_iou",
                mean_iou,
                specification.detection_iou_threshold,
                ">=",
                mean_iou + 1e-12 >= specification.detection_iou_threshold,
                "ratio",
            ),
        )

    def capability(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "input_contract": "image artifact + width/height metadata",
            "output_contract": "detections[{label,bbox_xyxy,score}]",
            "deterministic_checks": [
                "class-aware one-to-one IoU matching",
                "confidence filtering",
                "micro precision and recall",
                "mean matched IoU",
            ],
            "order_independent": True,
        }

    @classmethod
    def _parse_predictions(
        cls, value: Any, *, width: int, height: int
    ) -> tuple[_Detection, ...]:
        if not isinstance(value, dict) or set(value) != {"detections"}:
            raise ValueError("output must contain only detections")
        items = value["detections"]
        if not isinstance(items, list) or len(items) > 1000:
            raise ValueError("detections must be an array with at most 1000 entries")
        parsed = [
            cls._parse_detection(item, require_score=True, width=width, height=height)
            for item in items
        ]
        return tuple(sorted(parsed, key=lambda item: (item.label, item.bbox, -item.score)))

    @classmethod
    def _parse_reference(
        cls, value: Any, *, width: int | None = None, height: int | None = None
    ) -> tuple[_Detection, ...]:
        if not isinstance(value, dict) or set(value) != {"objects"}:
            raise ValueError("reference must contain only objects")
        items = value["objects"]
        if not isinstance(items, list) or len(items) > 1000:
            raise ValueError("objects must be an array with at most 1000 entries")
        parsed = [
            cls._parse_detection(
                item,
                require_score=False,
                width=width,
                height=height,
            )
            for item in items
        ]
        return tuple(sorted(parsed, key=lambda item: (item.label, item.bbox)))

    @staticmethod
    def _parse_detection(
        value: Any,
        *,
        require_score: bool,
        width: int | None,
        height: int | None,
    ) -> _Detection:
        required = {"label", "bbox_xyxy", "score"} if require_score else {
            "label",
            "bbox_xyxy",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError(f"detection entry fields must be {sorted(required)}")
        label = value["label"]
        bbox = value["bbox_xyxy"]
        if not isinstance(label, str) or not label.strip() or len(label) > 200:
            raise ValueError("detection label must be a non-empty bounded string")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("bbox_xyxy must contain four numbers")
        coords = tuple(
            finite_number(item, label=f"bbox_xyxy[{index}]")
            for index, item in enumerate(bbox)
        )
        x1, y1, x2, y2 = coords
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox_xyxy must have positive area")
        if width is not None and height is not None and (
            x1 < 0 or y1 < 0 or x2 > width or y2 > height
        ):
            raise ValueError("bbox_xyxy lies outside image dimensions")
        score = finite_number(value["score"], label="score") if require_score else 1.0
        if score < 0 or score > 1:
            raise ValueError("score must be between 0 and 1")
        return _Detection(label=label.strip(), bbox=coords, score=score)

    @classmethod
    def _match(
        cls,
        predictions: tuple[_Detection, ...],
        references: tuple[_Detection, ...],
        *,
        iou_threshold: float,
    ) -> tuple[tuple[int, int, float], ...]:
        adjacency: dict[int, list[tuple[int, float]]] = {}
        for pred_index, prediction in enumerate(predictions):
            candidates = [
                (ref_index, cls._iou(prediction.bbox, reference.bbox))
                for ref_index, reference in enumerate(references)
                if prediction.label == reference.label
            ]
            adjacency[pred_index] = sorted(
                (
                    (ref_index, iou)
                    for ref_index, iou in candidates
                    if iou + 1e-12 >= iou_threshold
                ),
                key=lambda item: (-item[1], item[0]),
            )
        matched_reference: dict[int, int] = {}

        def augment(pred_index: int, visited: set[int]) -> bool:
            for ref_index, _iou in adjacency[pred_index]:
                if ref_index in visited:
                    continue
                visited.add(ref_index)
                previous = matched_reference.get(ref_index)
                if previous is None or augment(previous, visited):
                    matched_reference[ref_index] = pred_index
                    return True
            return False

        for pred_index in range(len(predictions)):
            augment(pred_index, set())
        result = [
            (
                pred_index,
                ref_index,
                cls._iou(predictions[pred_index].bbox, references[ref_index].bbox),
            )
            for ref_index, pred_index in matched_reference.items()
        ]
        return tuple(sorted(result))

    @staticmethod
    def _iou(
        first: tuple[float, float, float, float],
        second: tuple[float, float, float, float],
    ) -> float:
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[2], second[2])
        bottom = min(first[3], second[3])
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        first_area = (first[2] - first[0]) * (first[3] - first[1])
        second_area = (second[2] - second[0]) * (second[3] - second[1])
        union = first_area + second_area - intersection
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 1.0
