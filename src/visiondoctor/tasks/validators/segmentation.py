from __future__ import annotations

from typing import Any

import numpy as np

from visiondoctor.schemas import ArtifactRef, CaseEvidence, TaskKind, TaskSpecification
from visiondoctor.tasks.validators.base import (
    TaskCaseEvaluation,
    TaskValidatorPlugin,
    image_input,
    metric,
)


class SegmentationValidator(TaskValidatorPlugin):
    kind = TaskKind.SEGMENTATION
    _MAX_MASK_PIXELS = 4_000_000

    def validate_input(self, value: Any, artifacts: tuple[ArtifactRef, ...]) -> None:
        image_input(value, artifacts)

    def validate_reference(self, expected: Any) -> None:
        self._parse_mask(expected, label="reference")

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
                raise ValueError("segmentation evidence omitted task input")
            _artifact, width, height = image_input(
                evidence.structured_input.value, evidence.input_artifacts
            )
            predicted = self._parse_mask(actual, label="output")
            reference = self._parse_mask(expected, label="reference")
            expected_shape = (height, width)
            if predicted.shape != expected_shape:
                raise ValueError(
                    f"output mask shape {predicted.shape} does not match image {expected_shape}"
                )
            if reference.shape != expected_shape:
                raise ValueError(
                    f"reference mask shape {reference.shape} does not match image {expected_shape}"
                )
        except (KeyError, TypeError, ValueError) as exc:
            return TaskCaseEvaluation(
                case_id=case_id,
                contract_valid=False,
                passed=False,
                failures=(f"segmentation output contract invalid: {exc}",),
            )

        intersection, union = self._class_counts(predicted, reference)
        class_ious = [
            intersection[label] / union[label]
            for label in sorted(union)
            if union[label] > 0
        ]
        mean_iou = float(np.mean(class_ious)) if class_ious else 1.0
        correct_pixels = int(np.count_nonzero(predicted == reference))
        total_pixels = int(reference.size)
        pixel_accuracy = correct_pixels / total_pixels
        boundary_true_positive_pred, predicted_boundary_count, reference_boundary_count = (
            self._boundary_counts(
                predicted,
                reference,
                specification.segmentation_boundary_tolerance_px,
            )
        )
        boundary_true_positive_ref, _, _ = self._boundary_counts(
            reference,
            predicted,
            specification.segmentation_boundary_tolerance_px,
        )
        boundary_precision = self._ratio(
            boundary_true_positive_pred, predicted_boundary_count
        )
        boundary_recall = self._ratio(
            boundary_true_positive_ref, reference_boundary_count
        )
        boundary_f1 = self._f1(boundary_precision, boundary_recall)
        failures: list[str] = []
        thresholds = (
            (
                "mean_iou",
                mean_iou,
                specification.segmentation_mean_iou_threshold,
            ),
            (
                "pixel_accuracy",
                pixel_accuracy,
                specification.segmentation_pixel_accuracy_threshold,
            ),
            (
                "boundary_f1",
                boundary_f1,
                specification.segmentation_boundary_f1_threshold,
            ),
        )
        for name, value, threshold in thresholds:
            if value + 1e-12 < threshold:
                failures.append(
                    f"segmentation {name}={value:.6f} below {threshold:.6f}"
                )
        return TaskCaseEvaluation(
            case_id=case_id,
            contract_valid=True,
            passed=not failures,
            failures=tuple(failures),
            measurements={
                "mean_iou": mean_iou,
                "pixel_accuracy": pixel_accuracy,
                "boundary_f1": boundary_f1,
            },
            aggregate_data={
                "intersection": intersection,
                "union": union,
                "correct_pixels": correct_pixels,
                "total_pixels": total_pixels,
                "boundary_true_positive_pred": boundary_true_positive_pred,
                "boundary_true_positive_ref": boundary_true_positive_ref,
                "predicted_boundary_count": predicted_boundary_count,
                "reference_boundary_count": reference_boundary_count,
            },
        )

    def aggregate(
        self,
        evaluations: tuple[TaskCaseEvaluation, ...],
        specification: TaskSpecification,
    ) -> tuple:
        valid = tuple(item for item in evaluations if item.contract_valid)
        intersection: dict[int, int] = {}
        union: dict[int, int] = {}
        for item in valid:
            for key, value in item.aggregate_data["intersection"].items():
                label = int(key)
                intersection[label] = intersection.get(label, 0) + int(value)
            for key, value in item.aggregate_data["union"].items():
                label = int(key)
                union[label] = union.get(label, 0) + int(value)
        class_ious = [
            intersection.get(label, 0) / count
            for label, count in sorted(union.items())
            if count > 0
        ]
        mean_iou = float(np.mean(class_ious)) if class_ious else 1.0
        correct_pixels = sum(int(item.aggregate_data["correct_pixels"]) for item in valid)
        total_pixels = sum(int(item.aggregate_data["total_pixels"]) for item in valid)
        pixel_accuracy = self._ratio(correct_pixels, total_pixels)
        matched_predicted = sum(
            int(item.aggregate_data["boundary_true_positive_pred"]) for item in valid
        )
        matched_reference = sum(
            int(item.aggregate_data["boundary_true_positive_ref"]) for item in valid
        )
        predicted_count = sum(
            int(item.aggregate_data["predicted_boundary_count"]) for item in valid
        )
        reference_count = sum(
            int(item.aggregate_data["reference_boundary_count"]) for item in valid
        )
        boundary_precision = self._ratio(matched_predicted, predicted_count)
        boundary_recall = self._ratio(matched_reference, reference_count)
        boundary_f1 = self._f1(boundary_precision, boundary_recall)
        return (
            metric(
                "segmentation_mean_iou",
                mean_iou,
                specification.segmentation_mean_iou_threshold,
                ">=",
                mean_iou + 1e-12
                >= specification.segmentation_mean_iou_threshold,
                "ratio",
            ),
            metric(
                "segmentation_pixel_accuracy",
                pixel_accuracy,
                specification.segmentation_pixel_accuracy_threshold,
                ">=",
                pixel_accuracy + 1e-12
                >= specification.segmentation_pixel_accuracy_threshold,
                "ratio",
            ),
            metric(
                "segmentation_boundary_f1",
                boundary_f1,
                specification.segmentation_boundary_f1_threshold,
                ">=",
                boundary_f1 + 1e-12
                >= specification.segmentation_boundary_f1_threshold,
                "ratio",
            ),
        )

    def capability(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "input_contract": "image artifact + width/height metadata",
            "output_contract": "{mask: two-dimensional non-negative integer array}",
            "deterministic_checks": [
                "mask shape and class contract",
                "class-wise mean intersection over union",
                "pixel accuracy",
                "class-aware boundary F1 with pixel tolerance",
            ],
            "order_independent": False,
        }

    @classmethod
    def _parse_mask(cls, value: Any, *, label: str) -> np.ndarray:
        if not isinstance(value, dict) or set(value) != {"mask"}:
            raise ValueError(f"{label} must contain only mask")
        mask = value["mask"]
        if not isinstance(mask, list) or not mask or len(mask) > cls._MAX_MASK_PIXELS:
            raise ValueError(f"{label} mask must be a non-empty two-dimensional array")
        width: int | None = None
        rows: list[list[int]] = []
        pixels = 0
        for row_index, row in enumerate(mask):
            if not isinstance(row, list) or not row:
                raise ValueError(f"{label} mask row {row_index} must be non-empty")
            if width is None:
                width = len(row)
            elif len(row) != width:
                raise ValueError(f"{label} mask rows must have equal length")
            parsed_row: list[int] = []
            for column_index, item in enumerate(row):
                if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                    raise ValueError(
                        f"{label} mask[{row_index}][{column_index}] must be a "
                        "non-negative integer"
                    )
                if item > 65535:
                    raise ValueError(f"{label} mask class id exceeds 65535")
                parsed_row.append(item)
            pixels += len(parsed_row)
            if pixels > cls._MAX_MASK_PIXELS:
                raise ValueError(f"{label} mask exceeds the supported pixel bound")
            rows.append(parsed_row)
        return np.asarray(rows, dtype=np.int64)

    @staticmethod
    def _class_counts(
        predicted: np.ndarray, reference: np.ndarray
    ) -> tuple[dict[int, int], dict[int, int]]:
        labels = np.union1d(np.unique(predicted), np.unique(reference))
        intersection: dict[int, int] = {}
        union: dict[int, int] = {}
        for item in labels:
            label = int(item)
            predicted_class = predicted == label
            reference_class = reference == label
            intersection[label] = int(np.count_nonzero(predicted_class & reference_class))
            union[label] = int(np.count_nonzero(predicted_class | reference_class))
        return intersection, union

    @classmethod
    def _boundary_counts(
        cls, source: np.ndarray, target: np.ndarray, tolerance: int
    ) -> tuple[int, int, int]:
        matched = 0
        source_count = 0
        target_count = 0
        labels = np.union1d(np.unique(source), np.unique(target))
        for item in labels:
            label = int(item)
            source_boundary = cls._boundary(source == label)
            target_boundary = cls._boundary(target == label)
            source_count += int(np.count_nonzero(source_boundary))
            target_count += int(np.count_nonzero(target_boundary))
            matched += int(
                np.count_nonzero(source_boundary & cls._dilate(target_boundary, tolerance))
            )
        return matched, source_count, target_count

    @staticmethod
    def _boundary(mask: np.ndarray) -> np.ndarray:
        boundary = np.zeros(mask.shape, dtype=bool)
        boundary[1:, :] |= mask[1:, :] != mask[:-1, :]
        boundary[:-1, :] |= mask[:-1, :] != mask[1:, :]
        boundary[:, 1:] |= mask[:, 1:] != mask[:, :-1]
        boundary[:, :-1] |= mask[:, :-1] != mask[:, 1:]
        return boundary & mask

    @staticmethod
    def _dilate(mask: np.ndarray, tolerance: int) -> np.ndarray:
        if tolerance == 0:
            return mask
        dilated = mask.copy()
        for _iteration in range(tolerance):
            previous = dilated.copy()
            dilated[1:, :] |= previous[:-1, :]
            dilated[:-1, :] |= previous[1:, :]
            dilated[:, 1:] |= previous[:, :-1]
            dilated[:, :-1] |= previous[:, 1:]
            dilated[1:, 1:] |= previous[:-1, :-1]
            dilated[1:, :-1] |= previous[:-1, 1:]
            dilated[:-1, 1:] |= previous[1:, :-1]
            dilated[:-1, :-1] |= previous[1:, 1:]
        return dilated

    @staticmethod
    def _ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 1.0

    @staticmethod
    def _f1(precision: float, recall: float) -> float:
        return (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
