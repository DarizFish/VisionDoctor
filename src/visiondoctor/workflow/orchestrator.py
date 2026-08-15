from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

from visiondoctor.adapters.base import (
    CandidateReleaseGate,
    EvidenceProvider,
    ExecutionBackend,
    ExternalGateResult,
    ReferenceProvider,
)
from visiondoctor.agents import DiagnosisAgent, PatchAgent, QAAgent
from visiondoctor.evidence import EvidenceStore, TraceRecorder
from visiondoctor.sandbox import SandboxError
from visiondoctor.schemas import (
    CandidateKind,
    CandidateVersion,
    Decision,
    DiagnosisReport,
    ExecutionResult,
    FailureCategory,
    HumanAction,
    Incident,
    ValidationReport,
    WorkflowState,
)
from visiondoctor.workflow.state_machine import WorkflowStateMachine


@dataclass(frozen=True)
class DemoRunResult:
    incident: Incident
    state: WorkflowState
    history: tuple[WorkflowState, ...]
    diagnosis: DiagnosisReport
    baseline_validation: ValidationReport
    faulty_validation: ValidationReport
    candidate_validations: tuple[ValidationReport, ...]
    selected_candidate: CandidateVersion
    external_gate_results: tuple[ExternalGateResult, ...]
    run_root: str


class Orchestrator:
    """State owner and coordinator; it neither writes patches nor decides QA metrics."""

    def __init__(
        self,
        evidence_provider: EvidenceProvider,
        execution_backend: ExecutionBackend,
        reference_provider: ReferenceProvider,
        store: EvidenceStore,
        diagnosis_agent: DiagnosisAgent,
        patch_agent: PatchAgent,
        release_gate: CandidateReleaseGate | None = None,
        max_patch_attempts: int = 3,
        max_release_gate_attempts: int = 2,
    ) -> None:
        if max_patch_attempts < 1:
            raise ValueError("max_patch_attempts must be positive")
        if max_release_gate_attempts < 1:
            raise ValueError("max_release_gate_attempts must be positive")
        self.evidence_provider = evidence_provider
        self.execution_backend = execution_backend
        self.reference_provider = reference_provider
        self.store = store
        self.release_gate = release_gate
        self.max_patch_attempts = max_patch_attempts
        self.max_release_gate_attempts = max_release_gate_attempts
        self.trace = TraceRecorder(store)
        self.state_machine = WorkflowStateMachine()
        self.diagnosis_agent = diagnosis_agent
        self.patch_agent = patch_agent
        self.qa_agent = QAAgent()

    def run(self, incident: Incident) -> DemoRunResult:
        self.store.write_model("incident.json", incident)
        self._transition(WorkflowState.CONTEXT_CHECKING)
        context = self.evidence_provider.prepare_case(incident)
        evidence = self.evidence_provider.collect(context)
        references = tuple(
            self.reference_provider.get_reference(case) for case in incident.case_set
        )
        self.store.write_model("evidence_bundle.json", evidence)
        self.store.write_json(
            "trusted_qa/references.json",
            [reference.model_dump(mode="json") for reference in references],
        )

        self._transition(WorkflowState.REPRODUCING)
        baseline = CandidateVersion(
            candidate_id="baseline",
            kind=CandidateKind.BASELINE,
            base_commit=incident.baseline_commit,
            rationale="known-good commit required by the frozen MVP",
        )
        faulty = CandidateVersion(
            candidate_id="faulty",
            kind=CandidateKind.FAULTY,
            base_commit=incident.faulty_commit,
            rationale="faulty commit under diagnosis",
        )
        baseline_execution, baseline_checks, _baseline_diff = self._execute(baseline, incident)
        baseline_validation = self.qa_agent.validate(
            incident,
            evidence,
            baseline_execution,
            references,
            baseline_execution,
            baseline_checks,
        )
        self.store.write_model("baseline_results/execution.json", baseline_execution)
        self.store.write_model("baseline_results/validation.json", baseline_validation)
        if baseline_validation.decision is not Decision.PASS:
            self._transition(WorkflowState.INFRA_ERROR)
            raise RuntimeError("known-good baseline failed the independent release gate")

        faulty_execution, faulty_checks, _faulty_diff = self._execute(faulty, incident)
        faulty_validation = self.qa_agent.validate(
            incident,
            evidence,
            faulty_execution,
            references,
            baseline_execution,
            faulty_checks,
        )
        self.store.write_model("faulty_results/execution.json", faulty_execution)
        self.store.write_model("faulty_results/validation.json", faulty_validation)
        if faulty_validation.decision is not Decision.REJECTED:
            self._transition(WorkflowState.NOT_REPRODUCED)
            raise RuntimeError("faulty commit did not reproduce the required regression")

        self._transition(WorkflowState.DIAGNOSING)
        diagnosis = self.diagnosis_agent.diagnose(
            incident,
            evidence,
            faulty_execution,
            baseline_validation,
            faulty_validation,
        )
        self.store.write_model("diagnosis_report.json", diagnosis)
        if not diagnosis.confirmed:
            self._transition(WorkflowState.COLLECT_MORE_EVIDENCE)
            raise RuntimeError("root cause was not confirmed by deterministic evidence")
        self._transition(WorkflowState.ROOT_CAUSE_CONFIRMED)
        self._transition(WorkflowState.PATCH_GENERATING)

        validations: list[ValidationReport] = []
        external_gate_results: list[ExternalGateResult] = []
        selected: CandidateVersion | None = None
        technical_review = False
        previous_validation: ValidationReport | None = None
        previous_failure: str | None = None
        patch_hashes: list[str] = []
        for attempt in range(1, self.max_patch_attempts + 1):
            candidate = self.patch_agent.generate(
                incident,
                diagnosis,
                attempt=attempt,
                previous_validation=previous_validation,
                previous_patch_sha256=tuple(patch_hashes),
                previous_failure=previous_failure,
            )
            patch_hashes.append(hashlib.sha256(candidate.patch_text.encode("utf-8")).hexdigest())
            self.store.write_text(f"{candidate.candidate_id}.patch", candidate.patch_text)
            self._transition(WorkflowState.VERIFYING)
            try:
                execution, checks, actual_diff = self._execute(
                    candidate, incident, retain_for_cleanup=True
                )
            except SandboxError as exc:
                previous_failure = f"candidate could not be prepared: {exc}"
                self.store.write_json(
                    f"{candidate.candidate_id}_preparation_failure.json",
                    {
                        "candidate_id": candidate.candidate_id,
                        "error": previous_failure,
                        "retryable": attempt < self.max_patch_attempts,
                    },
                )
                self._transition(WorkflowState.PATCH_REJECTED)
                self._transition(WorkflowState.ROLLED_BACK)
                if attempt < self.max_patch_attempts:
                    self._transition(WorkflowState.PATCH_GENERATING)
                    continue
                break
            self.store.write_text(f"{candidate.candidate_id}.diff", actual_diff)
            validation = self.qa_agent.validate(
                incident,
                evidence,
                execution,
                references,
                baseline_execution,
                checks,
            )
            if validation.decision is Decision.PASS and self.release_gate is not None:
                external_gate = self._evaluate_release_gate(candidate, execution)
                external_gate_results.append(external_gate)
                self.store.write_json(
                    f"{candidate.candidate_id}_external_gate.json",
                    {
                        "gate_id": external_gate.gate_id,
                        "name": external_gate.name,
                        "case_id": external_gate.case_id,
                        "passed": external_gate.passed,
                        "infrastructure_error": external_gate.infrastructure_error,
                        "failure_category": (
                            external_gate.failure_category.value
                            if external_gate.failure_category is not None
                            else None
                        ),
                        "retryable": external_gate.retryable,
                        "attempt_count": external_gate.attempt_count,
                        "retry_exhausted": external_gate.retry_exhausted,
                        "details": external_gate.details,
                        "payload": external_gate.payload,
                    },
                )
                if external_gate.logs:
                    self.store.write_text(
                        f"{candidate.candidate_id}_gazebo.log", external_gate.logs
                    )
                validation = self.qa_agent.apply_external_gate(validation, external_gate)
            handle = self._last_handle
            if self._should_request_patch_retry(validation):
                self._transition(WorkflowState.PATCH_REJECTED)
                rollback_succeeded = self.execution_backend.cleanup(handle, rollback=True)
                validation = validation.model_copy(
                    update={"rollback_result": "SUCCESS" if rollback_succeeded else "FAILED"}
                )
                self.store.write_model(f"{candidate.candidate_id}_validation.json", validation)
                validations.append(validation)
                previous_validation = validation
                previous_failure = None
                self.trace.record(
                    "candidate_rejected",
                    candidate_id=candidate.candidate_id,
                    rollback_succeeded=rollback_succeeded,
                )
                if not rollback_succeeded:
                    raise RuntimeError("candidate rollback failed integrity verification")
                self._transition(WorkflowState.ROLLED_BACK)
                if attempt < self.max_patch_attempts:
                    self._transition(WorkflowState.PATCH_GENERATING)
                continue
            if validation.decision is Decision.REJECTED:
                self.execution_backend.cleanup(handle, rollback=False)
                self._transition(WorkflowState.INFRA_ERROR)
                raise RuntimeError(
                    "rejected validation lacked a candidate-owned failure category"
                )
            if validation.decision is Decision.INFRA_ERROR:
                self.execution_backend.cleanup(handle, rollback=False)
                self.store.write_model(f"{candidate.candidate_id}_validation.json", validation)
                validations.append(validation)
                self._transition(WorkflowState.INFRA_ERROR)
                raise RuntimeError("candidate verification encountered infrastructure failure")
            if validation.decision is Decision.NEEDS_REVIEW:
                self.execution_backend.cleanup(handle, rollback=False)
                self.store.write_model(f"{candidate.candidate_id}_validation.json", validation)
                validations.append(validation)
                self._transition(WorkflowState.AWAITING_TECHNICAL_REVIEW)
                selected = candidate
                technical_review = True
                break

            disposed = self.execution_backend.cleanup(handle, rollback=False)
            if not disposed:
                raise RuntimeError("candidate worktree cleanup failed integrity verification")
            validation = validation.model_copy(
                update={"human_action": HumanAction.AWAITING_APPROVAL}
            )
            self.store.write_model(f"{candidate.candidate_id}_validation.json", validation)
            validations.append(validation)
            selected = candidate
            self._transition(WorkflowState.AWAITING_HUMAN_APPROVAL)
            break

        if selected is None:
            raise RuntimeError("no candidate passed all mandatory gates")
        if technical_review:
            review_validation = validations[-1]
            self.store.write_json(
                "technical_review.json",
                {
                    "incident_id": incident.incident_id,
                    "candidate_id": selected.candidate_id,
                    "decision": WorkflowState.AWAITING_TECHNICAL_REVIEW,
                    "failure_category": (
                        review_validation.failure_category.value
                        if review_validation.failure_category is not None
                        else None
                    ),
                    "automatic_merge": False,
                    "patch_retry_requested": False,
                    "patch": f"{selected.candidate_id}.diff",
                    "validation_report": f"{selected.candidate_id}_validation.json",
                },
            )
            self.store.write_json(
                "workflow_state.json",
                {
                    "state": self.state_machine.state,
                    "history": self.state_machine.history,
                },
            )
            return DemoRunResult(
                incident=incident,
                state=self.state_machine.state,
                history=tuple(self.state_machine.history),
                diagnosis=diagnosis,
                baseline_validation=baseline_validation,
                faulty_validation=faulty_validation,
                candidate_validations=tuple(validations),
                selected_candidate=selected,
                external_gate_results=tuple(external_gate_results),
                run_root=str(self.store.run_root),
            )
        release_decision = {
            "incident_id": incident.incident_id,
            "candidate_id": selected.candidate_id,
            "decision": "AWAITING_HUMAN_APPROVAL",
            "automatic_merge": False,
            "validation_report": f"{selected.candidate_id}_validation.json",
            "patch": f"{selected.candidate_id}.diff",
            "external_gates": [
                f"{selected.candidate_id}_external_gate.json"
                for gate in external_gate_results
                if gate.passed
            ],
            "rollback": "reapply the faulty commit or discard the candidate branch",
        }
        self.store.write_json("release_decision.json", release_decision)
        self.store.write_text(
            "postmortem.md",
            self._postmortem(incident, diagnosis, validations, selected),
        )
        self.store.write_json(
            "workflow_state.json",
            {
                "state": self.state_machine.state,
                "history": self.state_machine.history,
            },
        )
        return DemoRunResult(
            incident=incident,
            state=self.state_machine.state,
            history=tuple(self.state_machine.history),
            diagnosis=diagnosis,
            baseline_validation=baseline_validation,
            faulty_validation=faulty_validation,
            candidate_validations=tuple(validations),
            selected_candidate=selected,
            external_gate_results=tuple(external_gate_results),
            run_root=str(self.store.run_root),
        )

    def _evaluate_release_gate(
        self, candidate: CandidateVersion, execution: ExecutionResult
    ) -> ExternalGateResult:
        if self.release_gate is None:
            raise RuntimeError("release gate is not configured")
        attempts: list[ExternalGateResult] = []
        for attempt in range(1, self.max_release_gate_attempts + 1):
            result = self.release_gate.evaluate(candidate, execution)
            attempts.append(result)
            self.store.write_json(
                f"{candidate.candidate_id}_external_gate_attempt-{attempt:02d}.json",
                self._external_gate_payload(result, attempt_count=attempt),
            )
            self.trace.record(
                "external_gate_attempt",
                candidate_id=candidate.candidate_id,
                attempt=attempt,
                passed=result.passed,
                failure_category=(
                    result.failure_category.value
                    if result.failure_category is not None
                    else None
                ),
                retryable=result.retryable,
            )
            if result.passed or not result.retryable:
                break
        final = attempts[-1]
        prior_retryable_failure = next(
            (result for result in attempts if not result.passed and result.retryable),
            None,
        )
        retry_exhausted = bool(
            not final.passed
            and prior_retryable_failure is not None
            and len(attempts) == self.max_release_gate_attempts
        )
        summaries = [
            self._external_gate_payload(result, attempt_count=index)
            for index, result in enumerate(attempts, start=1)
        ]
        combined_logs = "\n\n".join(
            f"===== release gate attempt {index} =====\n{result.logs}"
            for index, result in enumerate(attempts, start=1)
            if result.logs
        )
        return replace(
            final,
            retryable=False if retry_exhausted else final.retryable,
            failure_category=(
                prior_retryable_failure.failure_category
                if retry_exhausted and prior_retryable_failure is not None
                else final.failure_category
            ),
            attempt_count=len(attempts),
            retry_exhausted=retry_exhausted,
            payload={**final.payload, "attempts": summaries},
            logs=combined_logs,
        )

    @staticmethod
    def _should_request_patch_retry(validation: ValidationReport) -> bool:
        return validation.decision is Decision.REJECTED and validation.failure_category in {
            FailureCategory.CANDIDATE_DEFECT,
            FailureCategory.DETERMINISTIC_QA_FAILURE,
        }

    @staticmethod
    def _external_gate_payload(
        result: ExternalGateResult, *, attempt_count: int
    ) -> dict[str, object]:
        return {
            "gate_id": result.gate_id,
            "name": result.name,
            "case_id": result.case_id,
            "passed": result.passed,
            "infrastructure_error": result.infrastructure_error,
            "failure_category": (
                result.failure_category.value
                if result.failure_category is not None
                else None
            ),
            "retryable": result.retryable,
            "attempt_count": attempt_count,
            "details": result.details,
            "payload": result.payload,
        }

    _last_handle = None

    def _execute(
        self,
        candidate: CandidateVersion,
        incident: Incident,
        *,
        retain_for_cleanup: bool = False,
    ) -> tuple[ExecutionResult, tuple, str]:
        handle = self.execution_backend.prepare(candidate)
        self._last_handle = handle
        self.trace.record(
            "execution_started",
            candidate_id=candidate.candidate_id,
            base_commit=candidate.base_commit,
            worktree=str(handle.worktree),
        )
        for case in incident.case_set:
            self.execution_backend.run_case(handle, case)
        execution = self.execution_backend.finalize(handle)
        checks = self.execution_backend.policy_checks(handle)
        actual_diff = self.execution_backend.diff(handle)
        self.trace.record(
            "execution_completed",
            candidate_id=candidate.candidate_id,
            status=execution.status,
            tests_passed=execution.unit_tests.passed,
        )
        if not retain_for_cleanup:
            cleaned = self.execution_backend.cleanup(handle, rollback=False)
            if not cleaned:
                raise RuntimeError("reference worktree cleanup failed integrity verification")
        return execution, checks, actual_diff

    def _transition(self, target: WorkflowState) -> None:
        previous = self.state_machine.state
        self.state_machine.transition(target)
        self.trace.record("state_transition", previous=previous, current=target)

    @staticmethod
    def _postmortem(
        incident: Incident,
        diagnosis: DiagnosisReport,
        validations: list[ValidationReport],
        selected: CandidateVersion,
    ) -> str:
        rejected = [
            report.candidate_id for report in validations if report.decision is Decision.REJECTED
        ]
        return (
            f"# {incident.incident_id} postmortem draft\n\n"
            f"Root cause: `{diagnosis.root_cause}`. Diagnosed by model "
            f"`{diagnosis.model}` with confidence {diagnosis.confidence:.3f}.\n\n"
            f"Rejected candidates: {', '.join(rejected) or 'none'}.\n\n"
            f"Selected candidate: `{selected.candidate_id}`. It passed deterministic QA and is "
            "waiting for human approval. Any configured Gazebo/MoveIt fixed-motion gate also "
            "passed before this state; no branch was merged automatically.\n"
        )
