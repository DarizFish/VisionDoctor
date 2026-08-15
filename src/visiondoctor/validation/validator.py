from __future__ import annotations

import hashlib
import math
import statistics
from pathlib import Path

import numpy as np
from PIL import Image

from visiondoctor.geometry import (
    invert,
    project_origin,
    rotation_error_rad,
    translation_error_m,
)
from visiondoctor.schemas import (
    ArtifactRef,
    Decision,
    EvidenceBundle,
    ExecutionResult,
    ExecutionStatus,
    FailureCategory,
    HumanAction,
    Incident,
    MetricResult,
    PolicyCheck,
    ReferenceSignal,
    TaskKind,
    ValidationReport,
)
from visiondoctor.tasks.validators import get_validator_plugin
from visiondoctor.tasks.validators.base import TaskCaseEvaluation


class DefaultMultimodalValidator:
    """Independent deterministic QA; never accepts candidate-reported metrics."""

    def __init__(self, policy_checks: tuple[PolicyCheck, ...]) -> None:
        self.policy_checks = policy_checks

    def evaluate(
        self,
        incident: Incident,
        evidence: EvidenceBundle,
        execution: ExecutionResult,
        references: tuple[ReferenceSignal, ...],
        baseline_execution: ExecutionResult,
    ) -> ValidationReport:
        if incident.task.kind is not TaskKind.RGBD_POSE:
            return self._evaluate_task_output(
                incident, evidence, execution, references, baseline_execution
            )
        criteria = incident.acceptance_criteria
        evidence_by_case = {item.case_id: item for item in evidence.cases}
        reference_by_case = {item.case_id: item for item in references}
        result_by_case = {item.case_id: item for item in execution.case_results}
        expected_ids = [case.case_id for case in incident.case_set]

        missing = [
            case_id
            for case_id in expected_ids
            if case_id not in evidence_by_case
            or case_id not in reference_by_case
            or case_id not in result_by_case
        ]
        translation_errors: list[float] = []
        rotation_errors: list[float] = []
        reprojection_errors: list[float] = []
        tcp_errors: list[float] = []
        depth_ratios: list[float] = []
        rgb_integrity: list[bool] = []
        artifact_integrity: list[bool] = []
        reference_integrity: list[bool] = []
        passed_cases: list[str] = []
        failed_cases: dict[str, tuple[str, ...]] = {}

        for case_id in expected_ids:
            if case_id in missing:
                failed_cases[case_id] = ("missing evidence, reference, or execution result",)
                continue
            case_evidence = evidence_by_case[case_id]
            reference = reference_by_case[case_id]
            case_result = result_by_case[case_id]
            failures: list[str] = []
            if case_result.status is not ExecutionStatus.SUCCESS:
                failed_cases[case_id] = (f"execution status {case_result.status}",)
                continue
            if case_result.vision_outputs is None or case_result.robot_outputs is None:
                failed_cases[case_id] = ("execution omitted vision or robot outputs",)
                continue

            predicted = case_result.vision_outputs.t_base_object.as_array()
            reference_pose = reference.pose.as_array()
            translation = translation_error_m(predicted, reference_pose)
            rotation = rotation_error_rad(predicted, reference_pose)
            translation_errors.append(translation)
            rotation_errors.append(rotation)

            predicted_camera = invert(case_evidence.t_base_camera.as_array()) @ predicted
            try:
                predicted_pixel = project_origin(
                    predicted_camera, np.asarray(case_evidence.camera_matrix, dtype=float)
                )
                reprojection = float(
                    np.linalg.norm(
                        np.asarray(predicted_pixel) - np.asarray(case_evidence.expected_pixel)
                    )
                )
            except ValueError:
                reprojection = math.inf
            reprojection_errors.append(reprojection)

            reference_tcp = reference_pose.copy()
            reference_tcp[:3, 3] += np.array([0.0, 0.0, 0.10])
            tcp_error = translation_error_m(
                case_result.robot_outputs.actual_tcp.as_array(), reference_tcp
            )
            tcp_errors.append(tcp_error)

            depth_ratios.append(case_evidence.depth_valid_ratio)
            image_ok = self._check_rgb_marker(
                Path(case_evidence.rgb.path), case_evidence.expected_pixel
            )
            rgb_integrity.append(image_ok)
            artifacts_ok = all(
                self._check_artifact(artifact, evidence.artifact_hashes)
                for artifact in (
                    case_evidence.rgb,
                    case_evidence.depth,
                    case_evidence.manifest,
                )
            )
            artifact_integrity.append(artifacts_ok)
            reference_ok = (
                reference.provenance == f"sha256:{reference.artifact.sha256}"
                and self._check_artifact(
                    reference.artifact,
                    {reference.artifact.artifact_id: reference.artifact.sha256},
                )
            )
            reference_integrity.append(reference_ok)

            if translation > criteria.translation_rmse_m:
                failures.append(f"translation={translation:.6f}m")
            if rotation > criteria.mean_rotation_error_rad:
                failures.append(f"rotation={rotation:.6f}rad")
            if reprojection > criteria.reprojection_error_px:
                failures.append(f"reprojection={reprojection:.3f}px")
            if tcp_error > criteria.tcp_error_m:
                failures.append(f"tcp={tcp_error:.6f}m")
            if (
                criteria.fixed_motion_required
                and not case_result.robot_outputs.fixed_motion_completed
            ):
                failures.append("fixed motion not completed")
            if not image_ok:
                failures.append("RGB evidence marker missing")
            if case_evidence.depth_valid_ratio < 0.60:
                failures.append("depth valid ratio below 0.60")
            if not artifacts_ok:
                failures.append("evidence artifact hash mismatch")
            if not reference_ok:
                failures.append("reference provenance mismatch")
            if failures:
                failed_cases[case_id] = tuple(failures)
            else:
                passed_cases.append(case_id)

        translation_rmse = self._rmse(translation_errors)
        mean_rotation = self._mean(rotation_errors)
        mean_reprojection = self._mean(reprojection_errors)
        tcp_rmse = self._rmse(tcp_errors)
        scene_rate = len(passed_cases) / len(expected_ids) if expected_ids else 0.0
        fixed_motion_passed = all(
            result.robot_outputs is not None and result.robot_outputs.fixed_motion_completed
            for result in execution.case_results
        ) and len(execution.case_results) == len(expected_ids)
        depth_passed = bool(depth_ratios) and min(depth_ratios) >= 0.60
        rgb_passed = bool(rgb_integrity) and all(rgb_integrity)
        artifacts_passed = bool(artifact_integrity) and all(artifact_integrity)
        references_passed = bool(reference_integrity) and all(reference_integrity)
        gazebo_rgbd_count = sum(case.source == "gazebo" for case in evidence.cases)
        gazebo_rgbd_required = incident.metadata.get("rgbd_backend") == "gazebo"
        gazebo_rgbd_threshold = 1.0 if gazebo_rgbd_required else 0.0
        gazebo_rgbd_passed = gazebo_rgbd_count >= gazebo_rgbd_threshold
        latency_growth = self._latency_growth(execution, baseline_execution)
        policy_passed = bool(self.policy_checks) and all(
            check.passed for check in self.policy_checks
        )

        metrics = (
            self._metric(
                "unit_tests",
                execution.unit_tests.passed,
                True,
                "==",
                execution.unit_tests.passed,
            ),
            self._metric(
                "translation_rmse",
                translation_rmse,
                criteria.translation_rmse_m,
                "<=",
                translation_rmse <= criteria.translation_rmse_m,
                "m",
            ),
            self._metric(
                "mean_rotation_error",
                mean_rotation,
                criteria.mean_rotation_error_rad,
                "<=",
                mean_rotation <= criteria.mean_rotation_error_rad,
                "rad",
            ),
            self._metric(
                "mean_reprojection_error",
                mean_reprojection,
                criteria.reprojection_error_px,
                "<=",
                mean_reprojection <= criteria.reprojection_error_px,
                "px",
            ),
            self._metric(
                "scene_pass_rate",
                scene_rate,
                criteria.scene_pass_rate,
                ">=",
                scene_rate >= criteria.scene_pass_rate,
                "ratio",
            ),
            self._metric(
                "fixed_robot_motion",
                fixed_motion_passed,
                True,
                "==",
                fixed_motion_passed,
            ),
            self._metric(
                "tcp_validation_rmse",
                tcp_rmse,
                criteria.tcp_error_m,
                "<=",
                tcp_rmse <= criteria.tcp_error_m,
                "m",
            ),
            self._metric(
                "latency_growth",
                latency_growth,
                criteria.latency_growth_ratio,
                "<=",
                latency_growth <= criteria.latency_growth_ratio,
                "ratio",
            ),
            self._metric(
                "patch_policy",
                policy_passed,
                True,
                "==",
                policy_passed,
            ),
            self._metric("rgb_evidence_integrity", rgb_passed, True, "==", rgb_passed),
            self._metric("depth_evidence_quality", depth_passed, True, "==", depth_passed),
            self._metric(
                "gazebo_rgbd_scene_count",
                float(gazebo_rgbd_count),
                gazebo_rgbd_threshold,
                ">=",
                gazebo_rgbd_passed,
                "cases",
            ),
            self._metric(
                "evidence_hash_integrity",
                artifacts_passed,
                True,
                "==",
                artifacts_passed,
            ),
            self._metric(
                "reference_provenance_integrity",
                references_passed,
                True,
                "==",
                references_passed,
            ),
        )

        evidence_incomplete = bool(
            missing
            or not evidence.cases
            or not references
            or not rgb_passed
            or not depth_passed
            or not gazebo_rgbd_passed
            or not artifacts_passed
            or not references_passed
        )
        if execution.status is ExecutionStatus.INFRA_ERROR:
            decision = Decision.INFRA_ERROR
            failure_category = FailureCategory.INTERNAL_ORCHESTRATION_ERROR
        elif evidence_incomplete:
            decision = Decision.NEEDS_REVIEW
            failure_category = FailureCategory.EVIDENCE_INCOMPLETE
        elif all(metric.passed for metric in metrics if metric.mandatory):
            decision = Decision.PASS
            failure_category = None
        else:
            decision = Decision.REJECTED
            failure_category = (
                FailureCategory.CANDIDATE_DEFECT
                if execution.status is not ExecutionStatus.SUCCESS
                or not execution.unit_tests.passed
                else FailureCategory.DETERMINISTIC_QA_FAILURE
            )

        return ValidationReport(
            report_id=f"VAL-{execution.candidate_id}",
            incident_id=incident.incident_id,
            candidate_id=execution.candidate_id,
            metric_results=metrics,
            case_count=len(expected_ids),
            passed_cases=tuple(passed_cases),
            failed_cases=failed_cases,
            policy_checks=self.policy_checks,
            decision=decision,
            failure_category=failure_category,
            retryable=False,
            evidence_refs=(
                evidence.evidence_id,
                execution.execution_id,
                *(reference.reference_id for reference in references),
            ),
            rollback_result="PENDING" if decision is Decision.REJECTED else "NOT_REQUIRED",
            human_action=(
                HumanAction.AWAITING_APPROVAL
                if decision is Decision.PASS
                else HumanAction.NOT_APPLICABLE
            ),
        )

    def _evaluate_task_output(
        self,
        incident: Incident,
        evidence: EvidenceBundle,
        execution: ExecutionResult,
        references: tuple[ReferenceSignal, ...],
        baseline_execution: ExecutionResult,
    ) -> ValidationReport:
        """Apply a task plugin while core retains evidence and policy ownership."""

        plugin = get_validator_plugin(incident.task.kind)
        evidence_by_case = {item.case_id: item for item in evidence.cases}
        reference_by_case = {item.case_id: item for item in references}
        result_by_case = {item.case_id: item for item in execution.case_results}
        expected_ids = [case.case_id for case in incident.case_set]
        missing_evidence = [
            case_id
            for case_id in expected_ids
            if case_id not in evidence_by_case or case_id not in reference_by_case
        ]
        missing_results = [case_id for case_id in expected_ids if case_id not in result_by_case]
        duplicate_evidence = len(evidence_by_case) != len(evidence.cases)
        duplicate_references = len(reference_by_case) != len(references)
        duplicate_results = len(result_by_case) != len(execution.case_results)
        passed_cases: list[str] = []
        failed_cases: dict[str, tuple[str, ...]] = {}
        evidence_integrity: list[bool] = []
        reference_integrity: list[bool] = []
        kind_integrity: list[bool] = []
        evaluations: list[TaskCaseEvaluation] = []
        candidate_execution_failed = bool(missing_results or duplicate_results)

        for case_id in expected_ids:
            if case_id in missing_evidence:
                failed_cases[case_id] = ("missing evidence or QA reference",)
                continue
            if case_id in missing_results:
                failed_cases[case_id] = ("candidate omitted execution result",)
                continue
            case_evidence = evidence_by_case[case_id]
            reference = reference_by_case[case_id]
            result = result_by_case[case_id]
            kind_ok = (
                case_evidence.task_kind is incident.task.kind
                and reference.task_kind is incident.task.kind
            )
            kind_integrity.append(kind_ok)
            artifacts_ok = all(
                self._check_artifact(artifact, evidence.artifact_hashes)
                for artifact in (case_evidence.manifest, *case_evidence.input_artifacts)
            )
            evidence_integrity.append(artifacts_ok)
            reference_ok = (
                reference.provenance == f"sha256:{reference.artifact.sha256}"
                and self._check_artifact(
                    reference.artifact,
                    {reference.artifact.artifact_id: reference.artifact.sha256},
                )
            )
            reference_integrity.append(reference_ok)

            failures: list[str] = []
            input_contract_ok = True
            reference_contract_ok = True
            try:
                if case_evidence.structured_input is None:
                    raise ValueError("task input payload is absent")
                plugin.validate_input(
                    case_evidence.structured_input.value,
                    case_evidence.input_artifacts,
                )
            except (KeyError, TypeError, ValueError) as exc:
                input_contract_ok = False
                failures.append(f"evidence task contract invalid: {exc}")
            try:
                if reference.structured_output is None:
                    raise ValueError("QA expected output is absent")
                plugin.validate_reference(reference.structured_output.expected)
            except (KeyError, TypeError, ValueError) as exc:
                reference_contract_ok = False
                failures.append(f"QA reference contract invalid: {exc}")
            evidence_integrity[-1] = artifacts_ok and input_contract_ok
            reference_integrity[-1] = reference_ok and reference_contract_ok

            if result.status is not ExecutionStatus.SUCCESS:
                failures.append(f"execution status {result.status}")
                candidate_execution_failed = True
            elif result.structured_outputs is None:
                failures.append("candidate omitted task output")
                candidate_execution_failed = True
            elif input_contract_ok and reference_contract_ok and kind_ok:
                evaluation = plugin.evaluate_case(
                    case_id,
                    result.structured_outputs.value,
                    reference.structured_output.expected,
                    case_evidence,
                    incident.task,
                )
                evaluations.append(evaluation)
                failures.extend(evaluation.failures)
            if not kind_ok:
                failures.append("task adapter kind mismatch")
            if not artifacts_ok:
                failures.append("evidence artifact hash mismatch")
            if not reference_ok:
                failures.append("reference provenance mismatch")
            if failures:
                failed_cases[case_id] = tuple(failures[:20])
            else:
                passed_cases.append(case_id)

        case_rate = len(passed_cases) / len(expected_ids) if expected_ids else 0.0
        latency_growth = self._latency_growth(execution, baseline_execution)
        policy_passed = bool(self.policy_checks) and all(
            check.passed for check in self.policy_checks
        )
        artifacts_passed = (
            bool(evidence_integrity)
            and all(evidence_integrity)
            and not duplicate_evidence
        )
        references_passed = (
            bool(reference_integrity)
            and all(reference_integrity)
            and not duplicate_references
        )
        kinds_passed = bool(kind_integrity) and all(kind_integrity)
        output_contract_passed = (
            len(evaluations) == len(expected_ids)
            and not duplicate_results
            and all(item.contract_valid for item in evaluations)
        )
        metrics = (
            self._metric(
                "unit_tests",
                execution.unit_tests.passed,
                True,
                "==",
                execution.unit_tests.passed,
            ),
            self._metric(
                "task_output_contract",
                output_contract_passed,
                True,
                "==",
                output_contract_passed,
            ),
            *plugin.aggregate(tuple(evaluations), incident.task),
            self._metric(
                "case_pass_rate",
                case_rate,
                incident.task.case_pass_rate,
                ">=",
                case_rate + 1e-12 >= incident.task.case_pass_rate,
                "ratio",
            ),
            self._metric(
                "latency_growth",
                latency_growth,
                incident.acceptance_criteria.latency_growth_ratio,
                "<=",
                latency_growth <= incident.acceptance_criteria.latency_growth_ratio + 1e-12,
                "ratio",
            ),
            self._metric("patch_policy", policy_passed, True, "==", policy_passed),
            self._metric(
                "task_adapter_contract", kinds_passed, True, "==", kinds_passed
            ),
            self._metric(
                "evidence_hash_integrity",
                artifacts_passed,
                True,
                "==",
                artifacts_passed,
            ),
            self._metric(
                "reference_provenance_integrity",
                references_passed,
                True,
                "==",
                references_passed,
            ),
        )
        evidence_incomplete = bool(
            missing_evidence
            or duplicate_evidence
            or duplicate_references
            or not evidence.cases
            or not references
            or not kinds_passed
            or not artifacts_passed
            or not references_passed
        )
        if execution.status is ExecutionStatus.INFRA_ERROR:
            decision = Decision.INFRA_ERROR
            failure_category = FailureCategory.INTERNAL_ORCHESTRATION_ERROR
        elif evidence_incomplete:
            decision = Decision.NEEDS_REVIEW
            failure_category = FailureCategory.EVIDENCE_INCOMPLETE
        elif all(item.passed for item in metrics if item.mandatory):
            decision = Decision.PASS
            failure_category = None
        else:
            decision = Decision.REJECTED
            failure_category = (
                FailureCategory.CANDIDATE_DEFECT
                if execution.status is not ExecutionStatus.SUCCESS
                or not execution.unit_tests.passed
                or candidate_execution_failed
                else FailureCategory.DETERMINISTIC_QA_FAILURE
            )
        return ValidationReport(
            report_id=f"VAL-{execution.candidate_id}",
            incident_id=incident.incident_id,
            candidate_id=execution.candidate_id,
            metric_results=metrics,
            case_count=len(expected_ids),
            passed_cases=tuple(passed_cases),
            failed_cases=failed_cases,
            policy_checks=self.policy_checks,
            decision=decision,
            failure_category=failure_category,
            retryable=False,
            evidence_refs=(
                evidence.evidence_id,
                execution.execution_id,
                *(reference.reference_id for reference in references),
            ),
            rollback_result="PENDING" if decision is Decision.REJECTED else "NOT_REQUIRED",
            human_action=(
                HumanAction.AWAITING_APPROVAL
                if decision is Decision.PASS
                else HumanAction.NOT_APPLICABLE
            ),
        )

    @staticmethod
    def _check_rgb_marker(path: Path, expected_pixel: tuple[float, float]) -> bool:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        u = int(round(expected_pixel[0]))
        v = int(round(expected_pixel[1]))
        x0, x1 = max(0, u - 6), min(rgb.shape[1], u + 7)
        y0, y1 = max(0, v - 6), min(rgb.shape[0], v + 7)
        patch = rgb[y0:y1, x0:x1]
        return bool(
            patch.size
            and np.any(
                (patch[..., 1] > 150)
                & (patch[..., 1] > patch[..., 0] * 1.5)
                & (patch[..., 1] > patch[..., 2] * 1.2)
            )
        )

    @staticmethod
    def _check_artifact(artifact: ArtifactRef, expected_hashes: dict[str, str]) -> bool:
        path = Path(artifact.path)
        if not path.is_file():
            return False
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        return actual == artifact.sha256 == expected_hashes.get(artifact.artifact_id)

    @staticmethod
    def _rmse(values: list[float]) -> float:
        if not values:
            return math.inf
        return float(math.sqrt(sum(value * value for value in values) / len(values)))

    @staticmethod
    def _mean(values: list[float]) -> float:
        return float(statistics.fmean(values)) if values else math.inf

    @staticmethod
    def _latency_growth(candidate: ExecutionResult, baseline: ExecutionResult) -> float:
        if (
            candidate.benchmark_normalized_ratio is not None
            and baseline.benchmark_normalized_ratio is not None
        ):
            return max(
                0.0,
                candidate.benchmark_normalized_ratio
                / baseline.benchmark_normalized_ratio
                - 1.0,
            )
        candidate_values = [
            item.latency_s for item in candidate.case_results if item.latency_s is not None
        ]
        baseline_values = [
            item.latency_s for item in baseline.case_results if item.latency_s is not None
        ]
        if not candidate_values or not baseline_values:
            return math.inf
        baseline_median = statistics.median(baseline_values)
        if baseline_median <= 0:
            return math.inf
        candidate_median = statistics.median(candidate_values)
        if (
            candidate.benchmark_calibration_s is not None
            and baseline.benchmark_calibration_s is not None
        ):
            candidate_median /= candidate.benchmark_calibration_s
            baseline_median /= baseline.benchmark_calibration_s
        return max(0.0, candidate_median / baseline_median - 1.0)

    @staticmethod
    def _metric(
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
