from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from visiondoctor.schemas import (
    CandidateVersion,
    CaseExecutionResult,
    EvidenceBundle,
    ExecutionResult,
    FailureCategory,
    Incident,
    PolicyCheck,
    ReferenceSignal,
    TestCaseRef,
    ValidationReport,
)


class AdapterUnavailableError(RuntimeError):
    """An infrastructure problem, never a candidate-quality failure."""


@dataclass(frozen=True)
class CaseContext:
    incident: Incident
    dataset_root: Path


@dataclass(frozen=True)
class RuntimeHandle:
    handle_id: str
    candidate: CandidateVersion
    worktree: Path


@dataclass(frozen=True)
class ExternalGateResult:
    gate_id: str
    name: str
    case_id: str
    passed: bool
    failure_category: FailureCategory | None
    retryable: bool
    details: str
    payload: dict[str, Any]
    logs: str = ""
    attempt_count: int = 1
    retry_exhausted: bool = False

    def __post_init__(self) -> None:
        if self.attempt_count < 1:
            raise ValueError("external gate attempt_count must be positive")
        if self.passed and (
            self.failure_category is not None or self.retryable or self.retry_exhausted
        ):
            raise ValueError("passing external gate cannot carry failure metadata")
        if not self.passed and self.failure_category is None:
            raise ValueError("failed external gate requires an explicit failure category")
        if self.retry_exhausted and self.retryable:
            raise ValueError("exhausted external gate cannot still be retryable")

    @property
    def infrastructure_error(self) -> bool:
        return self.failure_category in {
            FailureCategory.SIMULATOR_UNAVAILABLE,
            FailureCategory.PLANNING_EXECUTION_TRANSIENT,
            FailureCategory.INTERNAL_ORCHESTRATION_ERROR,
        }


@runtime_checkable
class EvidenceProvider(Protocol):
    def prepare_case(self, incident: Incident) -> CaseContext: ...

    def collect(self, context: CaseContext) -> EvidenceBundle: ...


@runtime_checkable
class ExecutionBackend(Protocol):
    def prepare(self, candidate: CandidateVersion) -> RuntimeHandle: ...

    def run_case(self, handle: RuntimeHandle, case: TestCaseRef) -> CaseExecutionResult: ...

    def finalize(self, handle: RuntimeHandle) -> ExecutionResult: ...

    def cleanup(self, handle: RuntimeHandle, *, rollback: bool) -> bool: ...

    def policy_checks(self, handle: RuntimeHandle) -> tuple[PolicyCheck, ...]: ...

    def diff(self, handle: RuntimeHandle) -> str: ...


@runtime_checkable
class CandidateReleaseGate(Protocol):
    def evaluate(
        self, candidate: CandidateVersion, execution: ExecutionResult
    ) -> ExternalGateResult: ...


@runtime_checkable
class ReferenceProvider(Protocol):
    def get_reference(self, case: TestCaseRef) -> ReferenceSignal: ...


@runtime_checkable
class ValidationBackend(Protocol):
    def evaluate(
        self,
        incident: Incident,
        evidence: EvidenceBundle,
        execution: ExecutionResult,
        references: tuple[ReferenceSignal, ...],
        baseline_execution: ExecutionResult,
    ) -> ValidationReport: ...
