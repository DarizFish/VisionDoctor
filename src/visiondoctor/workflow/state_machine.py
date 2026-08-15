from __future__ import annotations

from dataclasses import dataclass, field

from visiondoctor.schemas import WorkflowState


class InvalidTransitionError(ValueError):
    pass


ALLOWED_TRANSITIONS: dict[WorkflowState, frozenset[WorkflowState]] = {
    WorkflowState.NEW: frozenset({WorkflowState.CONTEXT_CHECKING}),
    WorkflowState.CONTEXT_CHECKING: frozenset(
        {WorkflowState.NEED_MORE_INFORMATION, WorkflowState.REPRODUCING}
    ),
    WorkflowState.NEED_MORE_INFORMATION: frozenset({WorkflowState.CONTEXT_CHECKING}),
    WorkflowState.REPRODUCING: frozenset(
        {WorkflowState.NOT_REPRODUCED, WorkflowState.INFRA_ERROR, WorkflowState.DIAGNOSING}
    ),
    WorkflowState.DIAGNOSING: frozenset(
        {WorkflowState.COLLECT_MORE_EVIDENCE, WorkflowState.ROOT_CAUSE_CONFIRMED}
    ),
    WorkflowState.COLLECT_MORE_EVIDENCE: frozenset({WorkflowState.DIAGNOSING}),
    WorkflowState.ROOT_CAUSE_CONFIRMED: frozenset({WorkflowState.PATCH_GENERATING}),
    WorkflowState.PATCH_GENERATING: frozenset({WorkflowState.VERIFYING}),
    WorkflowState.VERIFYING: frozenset(
        {
            WorkflowState.PATCH_REJECTED,
            WorkflowState.AWAITING_HUMAN_APPROVAL,
            WorkflowState.AWAITING_TECHNICAL_REVIEW,
            WorkflowState.INFRA_ERROR,
        }
    ),
    WorkflowState.PATCH_REJECTED: frozenset({WorkflowState.ROLLED_BACK}),
    WorkflowState.ROLLED_BACK: frozenset({WorkflowState.PATCH_GENERATING}),
    WorkflowState.AWAITING_HUMAN_APPROVAL: frozenset(
        {
            WorkflowState.REJECTED_BY_HUMAN,
            WorkflowState.ADDITIONAL_TESTING,
            WorkflowState.PR_READY,
        }
    ),
    WorkflowState.ADDITIONAL_TESTING: frozenset({WorkflowState.VERIFYING}),
    WorkflowState.PR_READY: frozenset({WorkflowState.POSTMORTEM_COMPLETED}),
}


@dataclass
class WorkflowStateMachine:
    state: WorkflowState = WorkflowState.NEW
    history: list[WorkflowState] = field(default_factory=lambda: [WorkflowState.NEW])

    def transition(self, target: WorkflowState) -> None:
        allowed = ALLOWED_TRANSITIONS.get(self.state, frozenset())
        if target not in allowed:
            raise InvalidTransitionError(f"illegal transition: {self.state} -> {target}")
        self.state = target
        self.history.append(target)
