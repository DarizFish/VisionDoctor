from __future__ import annotations

import math
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from visiondoctor.projects.models import VisionProject


def utc_now() -> datetime:
    return datetime.now(UTC)


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BackendType(StrEnum):
    DATASET = "dataset"
    GAZEBO = "gazebo"


class TaskKind(StrEnum):
    """Deterministic task contracts supported by the core workflow."""

    RGBD_POSE = "rgbd_pose"
    STRUCTURED_OUTPUT = "structured_output"
    DETECTION = "detection"
    OCR = "ocr"
    SEGMENTATION = "segmentation"


class ExecutionStatus(StrEnum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    INFRA_ERROR = "INFRA_ERROR"


class Decision(StrEnum):
    PASS = "PASS"
    REJECTED = "REJECTED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    INFRA_ERROR = "INFRA_ERROR"


class FailureCategory(StrEnum):
    """Machine-actionable ownership for a failed execution or validation step."""

    CANDIDATE_DEFECT = "candidate_defect"
    DETERMINISTIC_QA_FAILURE = "deterministic_qa_failure"
    SIMULATOR_UNAVAILABLE = "simulator_unavailable"
    PLANNING_EXECUTION_TRANSIENT = "planning_execution_transient"
    EVIDENCE_INCOMPLETE = "evidence_incomplete"
    INTERNAL_ORCHESTRATION_ERROR = "internal_orchestration_error"


class HumanAction(StrEnum):
    NOT_APPLICABLE = "NOT_APPLICABLE"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    ADDITIONAL_TESTING = "ADDITIONAL_TESTING"


class CandidateKind(StrEnum):
    BASELINE = "baseline"
    FAULTY = "faulty"
    GENERATED = "generated"
    FIXED_COMPENSATION = "fixed_compensation"
    ROOT_CAUSE_FIX = "root_cause_fix"


class WorkflowState(StrEnum):
    NEW = "NEW"
    CONTEXT_CHECKING = "CONTEXT_CHECKING"
    NEED_MORE_INFORMATION = "NEED_MORE_INFORMATION"
    REPRODUCING = "REPRODUCING"
    NOT_REPRODUCED = "NOT_REPRODUCED"
    INFRA_ERROR = "INFRA_ERROR"
    DIAGNOSING = "DIAGNOSING"
    COLLECT_MORE_EVIDENCE = "COLLECT_MORE_EVIDENCE"
    ROOT_CAUSE_CONFIRMED = "ROOT_CAUSE_CONFIRMED"
    PATCH_GENERATING = "PATCH_GENERATING"
    VERIFYING = "VERIFYING"
    PATCH_REJECTED = "PATCH_REJECTED"
    ROLLED_BACK = "ROLLED_BACK"
    AWAITING_TECHNICAL_REVIEW = "AWAITING_TECHNICAL_REVIEW"
    AWAITING_HUMAN_APPROVAL = "AWAITING_HUMAN_APPROVAL"
    REJECTED_BY_HUMAN = "REJECTED_BY_HUMAN"
    ADDITIONAL_TESTING = "ADDITIONAL_TESTING"
    PR_READY = "PR_READY"
    POSTMORTEM_COMPLETED = "POSTMORTEM_COMPLETED"


class PoseTransform(FrozenModel):
    """Rigid transform following the frozen T_target_source convention."""

    target_frame: str = Field(min_length=1)
    source_frame: str = Field(min_length=1)
    matrix: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ]
    length_unit: Literal["m"] = "m"
    angle_unit: Literal["rad"] = "rad"
    quaternion_order: Literal["xyzw"] = "xyzw"

    @model_validator(mode="after")
    def validate_rigid_transform(self) -> PoseTransform:
        matrix = np.asarray(self.matrix, dtype=float)
        if not np.isfinite(matrix).all():
            raise ValueError("transform matrix must contain only finite values")
        if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
            raise ValueError("transform matrix must end with [0, 0, 0, 1]")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise ValueError("rotation block must be orthonormal")
        if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6):
            raise ValueError("rotation block determinant must be +1")
        return self

    def as_array(self) -> np.ndarray:
        return np.asarray(self.matrix, dtype=float)

    @classmethod
    def from_array(cls, target_frame: str, source_frame: str, matrix: np.ndarray) -> PoseTransform:
        array = np.asarray(matrix, dtype=float)
        if array.shape != (4, 4):
            raise ValueError("transform matrix must have shape (4, 4)")
        rows = tuple(tuple(float(value) for value in row) for row in array)
        return cls(target_frame=target_frame, source_frame=source_frame, matrix=rows)  # type: ignore[arg-type]


class RepositoryRef(FrozenModel):
    path: str = Field(min_length=1)
    branch: str = "main"
    access_mode: Literal["local", "remote"] = "local"


class TestCaseRef(FrozenModel):
    case_id: str = Field(min_length=1)
    manifest_path: str = Field(min_length=1)
    reference_path: str = Field(
        min_length=1,
        description="Trusted QA-only reference manifest; never exposed to model tools.",
    )


class AcceptanceCriteria(FrozenModel):
    unit_tests_required: bool = True
    translation_rmse_m: float = Field(default=0.005, gt=0)
    mean_rotation_error_rad: float = Field(default=0.01745, gt=0)
    reprojection_error_px: float = Field(default=5.0, gt=0)
    scene_pass_rate: float = Field(default=0.98, ge=0, le=1)
    fixed_motion_required: bool = True
    tcp_error_m: float = Field(default=0.005, gt=0)
    latency_growth_ratio: float = Field(default=0.10, ge=0)
    policy_checks_required: bool = True


class TaskSpecification(FrozenModel):
    """Selects task semantics without changing Agent or sandbox behavior."""

    kind: TaskKind = TaskKind.RGBD_POSE
    display_name: str = Field(default="RGB-D pose pipeline", min_length=1, max_length=200)
    case_pass_rate: float = Field(default=0.98, ge=0, le=1)
    numeric_absolute_tolerance: float = Field(default=1e-6, ge=0)
    numeric_relative_tolerance: float = Field(default=1e-6, ge=0)
    allow_extra_output_fields: bool = False
    detection_iou_threshold: float = Field(default=0.50, gt=0, le=1)
    detection_score_threshold: float = Field(default=0.25, ge=0, le=1)
    detection_precision_threshold: float = Field(default=0.90, ge=0, le=1)
    detection_recall_threshold: float = Field(default=0.90, ge=0, le=1)
    ocr_max_character_error_rate: float = Field(default=0.02, ge=0)
    ocr_max_word_error_rate: float = Field(default=0.05, ge=0)
    ocr_case_sensitive: bool = True
    ocr_normalize_whitespace: bool = True
    segmentation_mean_iou_threshold: float = Field(default=0.85, ge=0, le=1)
    segmentation_pixel_accuracy_threshold: float = Field(default=0.95, ge=0, le=1)
    segmentation_boundary_f1_threshold: float = Field(default=0.90, ge=0, le=1)
    segmentation_boundary_tolerance_px: int = Field(default=1, ge=0, le=20)

    @model_validator(mode="before")
    @classmethod
    def name_matches_default_kind(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        kind = value.get("kind", TaskKind.RGBD_POSE)
        normalized = TaskKind(kind)
        if normalized is TaskKind.RGBD_POSE:
            return value
        if value.get("display_name", "RGB-D pose pipeline") != "RGB-D pose pipeline":
            return value
        names = {
            TaskKind.STRUCTURED_OUTPUT: "Structured vision output",
            TaskKind.DETECTION: "Object detection",
            TaskKind.OCR: "Optical character recognition",
            TaskKind.SEGMENTATION: "Semantic segmentation",
        }
        return {**value, "display_name": names[normalized]}


class PatchPolicy(FrozenModel):
    allowed_globs: tuple[str, ...] = (
        "src/pose_transformer.py",
        "config/*.yaml",
        "tests/test_*.py",
    )
    forbidden_globs: tuple[str, ...] = (
        "acceptance/**",
        "references/**",
        "dataset/**",
        "gazebo/**",
        "moveit/**",
        "qa/**",
        "security/**",
    )
    max_files: int = Field(default=3, ge=1)
    max_changed_lines: int = Field(default=100, ge=1)
    allow_test_changes: bool = True
    forbid_test_removal: bool = True
    forbid_test_skips: bool = True


class Incident(FrozenModel):
    incident_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    repository: RepositoryRef
    project: VisionProject | None = None
    baseline_commit: str = Field(min_length=7)
    faulty_commit: str = Field(min_length=7)
    case_set: tuple[TestCaseRef, ...] = Field(min_length=1)
    task: TaskSpecification = Field(default_factory=TaskSpecification)
    acceptance_criteria: AcceptanceCriteria
    allowed_patch_scope: PatchPolicy
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def commits_must_differ(self) -> Incident:
        if self.baseline_commit == self.faulty_commit:
            raise ValueError("baseline_commit and faulty_commit must differ")
        case_ids = [case.case_id for case in self.case_set]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("case_set contains duplicate case_id values")
        return self


class ArtifactRef(FrozenModel):
    artifact_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    media_type: str = Field(min_length=1)


class StructuredCaseInput(FrozenModel):
    value: Any


class StructuredOutputs(FrozenModel):
    value: Any


class StructuredReferenceOutput(FrozenModel):
    expected: Any


class CaseEvidence(FrozenModel):
    case_id: str
    task_kind: TaskKind = TaskKind.RGBD_POSE
    rgb: ArtifactRef | None = None
    depth: ArtifactRef | None = None
    manifest: ArtifactRef
    t_base_camera: PoseTransform | None = None
    t_camera_object: PoseTransform | None = None
    camera_matrix: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] | None = None
    expected_pixel: tuple[float, float] | None = None
    depth_valid_ratio: float | None = Field(default=None, ge=0, le=1)
    structured_input: StructuredCaseInput | None = None
    input_artifacts: tuple[ArtifactRef, ...] = ()
    source: Literal["dataset", "gazebo"]
    captured_at: datetime

    @model_validator(mode="after")
    def fields_match_task(self) -> CaseEvidence:
        if self.task_kind is TaskKind.RGBD_POSE:
            required = (
                self.rgb,
                self.depth,
                self.t_base_camera,
                self.t_camera_object,
                self.camera_matrix,
                self.expected_pixel,
                self.depth_valid_ratio,
            )
            if any(value is None for value in required):
                raise ValueError("RGB-D pose evidence is incomplete")
            if self.structured_input is not None or self.input_artifacts:
                raise ValueError("RGB-D pose evidence cannot carry structured task fields")
        elif self.structured_input is None:
            raise ValueError("structured-output evidence requires an input payload")
        return self


class EvidenceBundle(FrozenModel):
    evidence_id: str
    incident_id: str
    issue_text: str
    raw_log: str
    code_diff: str
    cases: tuple[CaseEvidence, ...] = Field(min_length=1)
    artifact_hashes: dict[str, str]
    collected_at: datetime = Field(default_factory=utc_now)


class ReferenceSignal(FrozenModel):
    reference_id: str
    case_id: str
    task_kind: TaskKind = TaskKind.RGBD_POSE
    pose: PoseTransform | None = None
    structured_output: StructuredReferenceOutput | None = None
    source_type: Literal["dataset_label", "gazebo_truth", "baseline", "human_label"]
    provenance: str = Field(min_length=1)
    artifact: ArtifactRef
    captured_at: datetime

    @model_validator(mode="after")
    def value_matches_task(self) -> ReferenceSignal:
        if self.task_kind is TaskKind.RGBD_POSE:
            if self.pose is None or self.structured_output is not None:
                raise ValueError("RGB-D pose reference requires only a pose")
        elif self.structured_output is None or self.pose is not None:
            raise ValueError("structured-output reference requires only expected output")
        return self


class CandidateVersion(FrozenModel):
    candidate_id: str
    kind: CandidateKind
    base_commit: str
    patch_text: str = ""
    rationale: str
    expected_changed_files: tuple[str, ...] = ()


class VisionOutputs(FrozenModel):
    t_camera_object: PoseTransform
    t_base_object: PoseTransform
    confidence: float = Field(ge=0, le=1)


class RobotOutputs(FrozenModel):
    target_tcp: PoseTransform
    actual_tcp: PoseTransform
    fixed_motion_completed: bool
    motion_sequence: tuple[Literal["HOME", "OBSERVATION_POSE", "VALIDATION_POSE"], ...]
    error_code: str | None = None
    source: Literal["dataset_simulation", "gazebo"]


class CaseExecutionResult(FrozenModel):
    case_id: str
    status: ExecutionStatus
    vision_outputs: VisionOutputs | None = None
    robot_outputs: RobotOutputs | None = None
    structured_outputs: StructuredOutputs | None = None
    latency_s: float | None = Field(default=None, ge=0)
    stdout: str = ""
    stderr: str = ""


class UnitTestResult(FrozenModel):
    passed: bool
    tests_run: int = Field(ge=0)
    failures: int = Field(ge=0)
    errors: int = Field(ge=0)
    duration_s: float = Field(ge=0)
    stdout: str = ""
    stderr: str = ""


class ExecutionResult(FrozenModel):
    execution_id: str
    candidate_id: str
    backend_type: BackendType
    case_results: tuple[CaseExecutionResult, ...]
    unit_tests: UnitTestResult
    status: ExecutionStatus
    benchmark_calibration_s: float | None = Field(default=None, gt=0)
    benchmark_normalized_ratio: float | None = Field(default=None, gt=0)
    artifacts: tuple[ArtifactRef, ...] = ()
    started_at: datetime
    completed_at: datetime


class MetricResult(FrozenModel):
    name: str
    value: float | bool
    threshold: float | bool
    comparator: Literal["<=", ">=", "=="]
    passed: bool
    mandatory: bool = True
    unit: str | None = None


class PolicyCheck(FrozenModel):
    name: str
    passed: bool
    details: str


class ValidationReport(FrozenModel):
    report_id: str
    incident_id: str
    candidate_id: str
    metric_results: tuple[MetricResult, ...]
    case_count: int = Field(ge=0)
    passed_cases: tuple[str, ...]
    failed_cases: dict[str, tuple[str, ...]]
    policy_checks: tuple[PolicyCheck, ...]
    decision: Decision
    failure_category: FailureCategory | None = None
    retryable: bool = False
    evidence_refs: tuple[str, ...]
    rollback_result: Literal["NOT_REQUIRED", "PENDING", "SUCCESS", "FAILED"]
    human_action: HumanAction
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def decision_matches_failure_category(self) -> ValidationReport:
        candidate_failures = {
            FailureCategory.CANDIDATE_DEFECT,
            FailureCategory.DETERMINISTIC_QA_FAILURE,
        }
        infrastructure_failures = {
            FailureCategory.SIMULATOR_UNAVAILABLE,
            FailureCategory.PLANNING_EXECUTION_TRANSIENT,
            FailureCategory.INTERNAL_ORCHESTRATION_ERROR,
        }
        if self.decision is Decision.PASS:
            if self.failure_category is not None or self.retryable:
                raise ValueError("passing validation cannot carry a failure category")
        elif self.failure_category is None:
            raise ValueError("failed validation requires an explicit failure category")
        elif self.decision is Decision.REJECTED and self.failure_category not in candidate_failures:
            raise ValueError("rejected validation must be attributable to candidate or QA")
        elif (
            self.decision is Decision.INFRA_ERROR
            and self.failure_category not in infrastructure_failures
        ):
            raise ValueError("infrastructure decision requires an infrastructure category")
        elif self.decision is Decision.NEEDS_REVIEW and self.failure_category not in {
            FailureCategory.EVIDENCE_INCOMPLETE,
            FailureCategory.SIMULATOR_UNAVAILABLE,
            FailureCategory.PLANNING_EXECUTION_TRANSIENT,
        }:
            raise ValueError("review decision requires incomplete or exhausted external evidence")
        return self

    @field_validator("evidence_refs")
    @classmethod
    def evidence_must_be_referenced(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("validation report must reference supporting evidence")
        return value


class DiagnosisReport(FrozenModel):
    diagnosis_id: str
    incident_id: str
    root_cause: str = Field(min_length=1, max_length=4000)
    confirmed: bool
    confidence: float = Field(ge=0, le=1)
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    observations: tuple[str, ...] = Field(min_length=1)
    counterfactual_rmse_m: float | None = Field(default=None, ge=0)
    faulty_match_rmse_m: float | None = Field(default=None, ge=0)
    suspicious_diff: str
    recommended_fix: str = ""
    model: str = Field(default="", max_length=200)
