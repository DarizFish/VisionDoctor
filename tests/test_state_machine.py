from __future__ import annotations

import pytest

from visiondoctor.schemas import WorkflowState
from visiondoctor.workflow import InvalidTransitionError, WorkflowStateMachine


def test_state_machine_accepts_reject_rollback_retry_path() -> None:
    machine = WorkflowStateMachine()
    for state in (
        WorkflowState.CONTEXT_CHECKING,
        WorkflowState.REPRODUCING,
        WorkflowState.DIAGNOSING,
        WorkflowState.ROOT_CAUSE_CONFIRMED,
        WorkflowState.PATCH_GENERATING,
        WorkflowState.VERIFYING,
        WorkflowState.PATCH_REJECTED,
        WorkflowState.ROLLED_BACK,
        WorkflowState.PATCH_GENERATING,
        WorkflowState.VERIFYING,
        WorkflowState.AWAITING_HUMAN_APPROVAL,
    ):
        machine.transition(state)

    assert machine.state is WorkflowState.AWAITING_HUMAN_APPROVAL


def test_state_machine_rejects_self_approval_shortcut() -> None:
    machine = WorkflowStateMachine()

    with pytest.raises(InvalidTransitionError):
        machine.transition(WorkflowState.AWAITING_HUMAN_APPROVAL)


def test_state_machine_can_hold_candidate_for_technical_review() -> None:
    machine = WorkflowStateMachine()
    for state in (
        WorkflowState.CONTEXT_CHECKING,
        WorkflowState.REPRODUCING,
        WorkflowState.DIAGNOSING,
        WorkflowState.ROOT_CAUSE_CONFIRMED,
        WorkflowState.PATCH_GENERATING,
        WorkflowState.VERIFYING,
        WorkflowState.AWAITING_TECHNICAL_REVIEW,
    ):
        machine.transition(state)

    assert machine.state is WorkflowState.AWAITING_TECHNICAL_REVIEW
