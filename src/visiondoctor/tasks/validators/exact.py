from __future__ import annotations

import math
from typing import Any

from visiondoctor.schemas import ArtifactRef, CaseEvidence, TaskKind, TaskSpecification
from visiondoctor.tasks.validators.base import (
    TaskCaseEvaluation,
    TaskValidatorPlugin,
    metric,
)


class ExactStructuredValidator(TaskValidatorPlugin):
    """Compatibility plugin for historical generic structured-output incidents."""

    kind = TaskKind.STRUCTURED_OUTPUT

    def validate_input(self, value: Any, artifacts: tuple[ArtifactRef, ...]) -> None:
        del value, artifacts

    def validate_reference(self, expected: Any) -> None:
        del expected

    def evaluate_case(
        self,
        case_id: str,
        actual: Any,
        expected: Any,
        evidence: CaseEvidence,
        specification: TaskSpecification,
    ) -> TaskCaseEvaluation:
        del evidence
        failures = self._compare(
            actual,
            expected,
            absolute_tolerance=specification.numeric_absolute_tolerance,
            relative_tolerance=specification.numeric_relative_tolerance,
            allow_extra_fields=specification.allow_extra_output_fields,
        )
        return TaskCaseEvaluation(
            case_id=case_id,
            contract_valid=True,
            passed=not failures,
            failures=tuple(failures),
        )

    def aggregate(
        self,
        evaluations: tuple[TaskCaseEvaluation, ...],
        specification: TaskSpecification,
    ) -> tuple:
        del specification
        passed = all(item.passed for item in evaluations)
        return (metric("structured_exact_match", passed, True, "==", passed),)

    def capability(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "legacy": True}

    @classmethod
    def _compare(
        cls,
        actual: object,
        expected: object,
        *,
        absolute_tolerance: float,
        relative_tolerance: float,
        allow_extra_fields: bool,
        path: str = "$",
    ) -> list[str]:
        failures: list[str] = []
        if isinstance(expected, bool) or expected is None or isinstance(expected, str):
            if type(actual) is not type(expected) or actual != expected:
                failures.append(f"{path}: typed value mismatch")
            return failures
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            if not isinstance(actual, (int, float)) or isinstance(actual, bool):
                return [f"{path}: expected a numeric value"]
            if not math.isclose(
                float(actual),
                float(expected),
                abs_tol=absolute_tolerance,
                rel_tol=relative_tolerance,
            ):
                failures.append(f"{path}: numeric value outside tolerance")
            return failures
        if isinstance(expected, list):
            if not isinstance(actual, list):
                return [f"{path}: expected an array"]
            if len(actual) != len(expected):
                failures.append(
                    f"{path}: array length mismatch "
                    f"({len(actual)} returned, {len(expected)} required)"
                )
            for index, expected_item in enumerate(expected[: len(actual)]):
                failures.extend(
                    cls._compare(
                        actual[index],
                        expected_item,
                        absolute_tolerance=absolute_tolerance,
                        relative_tolerance=relative_tolerance,
                        allow_extra_fields=allow_extra_fields,
                        path=f"{path}[{index}]",
                    )
                )
            return failures
        if isinstance(expected, dict):
            if not isinstance(actual, dict):
                return [f"{path}: expected an object"]
            missing = sorted(set(expected) - set(actual))
            extra = sorted(set(actual) - set(expected))
            if missing:
                failures.append(f"{path}: missing fields {missing}")
            if extra and not allow_extra_fields:
                failures.append(f"{path}: unexpected fields {extra}")
            for key in expected.keys() & actual.keys():
                failures.extend(
                    cls._compare(
                        actual[key],
                        expected[key],
                        absolute_tolerance=absolute_tolerance,
                        relative_tolerance=relative_tolerance,
                        allow_extra_fields=allow_extra_fields,
                        path=f"{path}.{key}",
                    )
                )
            return failures
        return [f"{path}: unsupported QA reference type"]
