from __future__ import annotations

from pathlib import Path

import pytest

from tests.support import root_cause_patch
from visiondoctor.adapters.dataset_execution import DatasetExecutionBackend
from visiondoctor.sandbox import GitWorktreeSandbox, SandboxError
from visiondoctor.schemas import CandidateKind, CandidateVersion
from visiondoctor.workflow import DemoRunResult


def test_local_backend_refuses_unregistered_generated_patch(
    demo_result: DemoRunResult, tmp_path: Path
) -> None:
    repository = Path(demo_result.incident.repository.path)
    backend = DatasetExecutionBackend(
        demo_result.incident,
        GitWorktreeSandbox(repository, tmp_path / "worktrees"),
    )
    candidate = CandidateVersion(
        candidate_id="untrusted",
        kind=CandidateKind.ROOT_CAUSE_FIX,
        base_commit=demo_result.incident.faulty_commit,
        patch_text=root_cause_patch(),
        rationale="not registered by the trusted demo scenario",
    )

    with pytest.raises(SandboxError, match="Docker sandbox"):
        backend.prepare(candidate)


def test_candidate_environment_does_not_inherit_application_secrets(monkeypatch) -> None:
    monkeypatch.setenv("VISIONDOCTOR_TEST_SECRET", "must-not-leak")

    environment = DatasetExecutionBackend._sanitized_environment()

    assert "VISIONDOCTOR_TEST_SECRET" not in environment
    assert environment["PYTHONNOUSERSITE"] == "1"
