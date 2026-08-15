from __future__ import annotations

from visiondoctor.adapters.base import ExternalGateResult
from visiondoctor.schemas import (
    Decision,
    EvidenceBundle,
    ExecutionResult,
    FailureCategory,
    HumanAction,
    Incident,
    MetricResult,
    PolicyCheck,
    ReferenceSignal,
    ValidationReport,
)
from visiondoctor.validation import DefaultMultimodalValidator


class QAAgent:
    """Owns independent execution-derived validation and cannot edit a candidate."""

    def validate(
        self,
        incident: Incident,
        evidence: EvidenceBundle,
        execution: ExecutionResult,
        references: tuple[ReferenceSignal, ...],
        baseline_execution: ExecutionResult,
        policy_checks: tuple[PolicyCheck, ...],
    ) -> ValidationReport:
        validator = DefaultMultimodalValidator(policy_checks)
        return validator.evaluate(incident, evidence, execution, references, baseline_execution)

    def apply_external_gate(
        self, report: ValidationReport, gate: ExternalGateResult
    ) -> ValidationReport:
        metric = MetricResult(
            name=gate.name,
            value=gate.passed,
            threshold=True,
            comparator="==",
            passed=gate.passed,
            mandatory=True,
        )
        if gate.passed:
            return report.model_copy(
                update={
                    "metric_results": (*report.metric_results, metric),
                    "evidence_refs": (*report.evidence_refs, gate.gate_id),
                }
            )
        if gate.failure_category in {
            FailureCategory.CANDIDATE_DEFECT,
            FailureCategory.DETERMINISTIC_QA_FAILURE,
        }:
            decision = Decision.REJECTED
        elif gate.failure_category is FailureCategory.EVIDENCE_INCOMPLETE or (
            gate.retry_exhausted
            and gate.failure_category
            in {
                FailureCategory.SIMULATOR_UNAVAILABLE,
                FailureCategory.PLANNING_EXECUTION_TRANSIENT,
            }
        ):
            decision = Decision.NEEDS_REVIEW
        else:
            decision = Decision.INFRA_ERROR
        failed_cases = dict(report.failed_cases)
        failed_cases[gate.case_id] = (*failed_cases.get(gate.case_id, ()), gate.details)
        return report.model_copy(
            update={
                "metric_results": (*report.metric_results, metric),
                "failed_cases": failed_cases,
                "decision": decision,
                "failure_category": gate.failure_category,
                "retryable": gate.retryable,
                "rollback_result": (
                    "PENDING" if decision is Decision.REJECTED else "NOT_REQUIRED"
                ),
                "human_action": HumanAction.NOT_APPLICABLE,
                "evidence_refs": (*report.evidence_refs, gate.gate_id),
            }
        )
