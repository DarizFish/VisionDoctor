from __future__ import annotations

from visiondoctor.adapters.base import ExternalGateResult
from visiondoctor.agents import QAAgent
from visiondoctor.evidence import EvidenceStore
from visiondoctor.schemas import Decision, FailureCategory
from visiondoctor.workflow import DemoRunResult, Orchestrator


def _gate(
    *,
    failure_category: FailureCategory,
    retryable: bool = False,
    retry_exhausted: bool = False,
) -> ExternalGateResult:
    return ExternalGateResult(
        gate_id="GAZEBO-test",
        name="gazebo_moveit_fixed_motion",
        case_id="scene-main",
        passed=False,
        failure_category=failure_category,
        retryable=retryable,
        details="injected gate failure",
        payload={"success": False},
        attempt_count=2 if retry_exhausted else 1,
        retry_exhausted=retry_exhausted,
    )


def test_qa_rejects_candidate_quality_failure_from_external_gate(
    demo_result: DemoRunResult,
) -> None:
    report = demo_result.candidate_validations[-1]

    updated = QAAgent().apply_external_gate(
        report,
        _gate(failure_category=FailureCategory.DETERMINISTIC_QA_FAILURE),
    )

    assert updated.decision is Decision.REJECTED
    assert updated.metric_results[-1].name == "gazebo_moveit_fixed_motion"
    assert not updated.metric_results[-1].passed
    assert updated.failed_cases["scene-main"][-1] == "injected gate failure"
    assert updated.evidence_refs[-1] == "GAZEBO-test"


def test_qa_classifies_external_infrastructure_failure(
    demo_result: DemoRunResult,
) -> None:
    report = demo_result.candidate_validations[-1]

    updated = QAAgent().apply_external_gate(
        report,
        _gate(
            failure_category=FailureCategory.PLANNING_EXECUTION_TRANSIENT,
            retryable=True,
        ),
    )

    assert updated.decision is Decision.INFRA_ERROR


def test_qa_routes_exhausted_planning_failure_to_review_not_patch(
    demo_result: DemoRunResult,
) -> None:
    report = demo_result.candidate_validations[-1]

    updated = QAAgent().apply_external_gate(
        report,
        _gate(
            failure_category=FailureCategory.PLANNING_EXECUTION_TRANSIENT,
            retry_exhausted=True,
        ),
    )

    assert updated.decision is Decision.NEEDS_REVIEW
    assert updated.failure_category is FailureCategory.PLANNING_EXECUTION_TRANSIENT
    assert updated.rollback_result == "NOT_REQUIRED"


def test_orchestrator_retries_transient_gate_without_requesting_new_patch(
    tmp_path, demo_result: DemoRunResult
) -> None:
    class TransientGate:
        def __init__(self) -> None:
            self.calls = 0

        def evaluate(self, _candidate, _execution) -> ExternalGateResult:
            self.calls += 1
            return _gate(
                failure_category=FailureCategory.PLANNING_EXECUTION_TRANSIENT,
                retryable=True,
            )

    gate = TransientGate()
    orchestrator = Orchestrator(
        evidence_provider=None,  # type: ignore[arg-type]
        execution_backend=None,  # type: ignore[arg-type]
        reference_provider=None,  # type: ignore[arg-type]
        store=EvidenceStore(tmp_path / "run"),
        diagnosis_agent=None,  # type: ignore[arg-type]
        patch_agent=None,  # type: ignore[arg-type]
        release_gate=gate,
        max_release_gate_attempts=2,
    )

    final_gate = orchestrator._evaluate_release_gate(
        demo_result.selected_candidate,
        object(),  # type: ignore[arg-type]
    )
    validation = QAAgent().apply_external_gate(
        demo_result.candidate_validations[-1], final_gate
    )

    assert gate.calls == 2
    assert final_gate.retry_exhausted is True
    assert final_gate.attempt_count == 2
    assert validation.decision is Decision.NEEDS_REVIEW
    assert Orchestrator._should_request_patch_retry(validation) is False
