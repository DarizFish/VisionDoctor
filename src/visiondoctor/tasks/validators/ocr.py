from __future__ import annotations

import re
import unicodedata
from typing import Any

from visiondoctor.schemas import ArtifactRef, CaseEvidence, TaskKind, TaskSpecification
from visiondoctor.tasks.validators.base import (
    TaskCaseEvaluation,
    TaskValidatorPlugin,
    image_input,
    metric,
)


class OcrValidator(TaskValidatorPlugin):
    kind = TaskKind.OCR

    def validate_input(self, value: Any, artifacts: tuple[ArtifactRef, ...]) -> None:
        image_input(value, artifacts)

    def validate_reference(self, expected: Any) -> None:
        self._parse_text(expected, label="reference")

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
                raise ValueError("OCR evidence omitted task input")
            image_input(evidence.structured_input.value, evidence.input_artifacts)
            actual_text = self._parse_text(actual, label="output")
            expected_text = self._parse_text(expected, label="reference")
        except (KeyError, TypeError, ValueError) as exc:
            return TaskCaseEvaluation(
                case_id=case_id,
                contract_valid=False,
                passed=False,
                failures=(f"OCR output contract invalid: {exc}",),
            )

        normalized_actual = self._normalize(actual_text, specification)
        normalized_expected = self._normalize(expected_text, specification)
        character_distance = self._edit_distance(normalized_actual, normalized_expected)
        actual_words = self._words(normalized_actual)
        expected_words = self._words(normalized_expected)
        word_distance = self._edit_distance(actual_words, expected_words)
        character_error_rate = character_distance / max(1, len(normalized_expected))
        word_error_rate = word_distance / max(1, len(expected_words))
        failures: list[str] = []
        if (
            character_error_rate
            > specification.ocr_max_character_error_rate + 1e-12
        ):
            failures.append(
                f"OCR character_error_rate={character_error_rate:.6f} above "
                f"{specification.ocr_max_character_error_rate:.6f}"
            )
        if word_error_rate > specification.ocr_max_word_error_rate + 1e-12:
            failures.append(
                f"OCR word_error_rate={word_error_rate:.6f} above "
                f"{specification.ocr_max_word_error_rate:.6f}"
            )
        return TaskCaseEvaluation(
            case_id=case_id,
            contract_valid=True,
            passed=not failures,
            failures=tuple(failures),
            measurements={
                "character_error_rate": character_error_rate,
                "word_error_rate": word_error_rate,
            },
            aggregate_data={
                "character_distance": character_distance,
                "reference_characters": len(normalized_expected),
                "word_distance": word_distance,
                "reference_words": len(expected_words),
            },
        )

    def aggregate(
        self,
        evaluations: tuple[TaskCaseEvaluation, ...],
        specification: TaskSpecification,
    ) -> tuple:
        valid = tuple(item for item in evaluations if item.contract_valid)
        character_distance = sum(
            int(item.aggregate_data["character_distance"]) for item in valid
        )
        reference_characters = sum(
            int(item.aggregate_data["reference_characters"]) for item in valid
        )
        word_distance = sum(int(item.aggregate_data["word_distance"]) for item in valid)
        reference_words = sum(int(item.aggregate_data["reference_words"]) for item in valid)
        character_error_rate = character_distance / max(1, reference_characters)
        word_error_rate = word_distance / max(1, reference_words)
        return (
            metric(
                "ocr_character_error_rate",
                character_error_rate,
                specification.ocr_max_character_error_rate,
                "<=",
                character_error_rate
                <= specification.ocr_max_character_error_rate + 1e-12,
                "ratio",
            ),
            metric(
                "ocr_word_error_rate",
                word_error_rate,
                specification.ocr_max_word_error_rate,
                "<=",
                word_error_rate <= specification.ocr_max_word_error_rate + 1e-12,
                "ratio",
            ),
        )

    def capability(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "input_contract": "image artifact + width/height metadata",
            "output_contract": "{text: string}",
            "deterministic_checks": [
                "Unicode normalization",
                "configurable case and whitespace normalization",
                "micro character error rate",
                "micro word error rate",
            ],
            "order_independent": False,
        }

    @staticmethod
    def _parse_text(value: Any, *, label: str) -> str:
        if not isinstance(value, dict) or set(value) != {"text"}:
            raise ValueError(f"{label} must contain only text")
        text = value["text"]
        if not isinstance(text, str) or len(text) > 4096:
            raise ValueError(f"{label} text must be a bounded string")
        return text

    @staticmethod
    def _normalize(value: str, specification: TaskSpecification) -> str:
        normalized = unicodedata.normalize("NFKC", value)
        if specification.ocr_normalize_whitespace:
            normalized = re.sub(r"\s+", " ", normalized).strip()
        if not specification.ocr_case_sensitive:
            normalized = normalized.casefold()
        return normalized

    @staticmethod
    def _words(value: str) -> tuple[str, ...]:
        return tuple(re.findall(r"\S+", value, flags=re.UNICODE))

    @staticmethod
    def _edit_distance(first: Any, second: Any) -> int:
        if len(first) < len(second):
            first, second = second, first
        previous = list(range(len(second) + 1))
        for row, first_item in enumerate(first, start=1):
            current = [row]
            for column, second_item in enumerate(second, start=1):
                current.append(
                    min(
                        current[-1] + 1,
                        previous[column] + 1,
                        previous[column - 1] + (first_item != second_item),
                    )
                )
            previous = current
        return previous[-1]
