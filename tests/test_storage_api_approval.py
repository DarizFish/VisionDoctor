from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from visiondoctor.api.app import ApiSettings, create_app
from visiondoctor.demo.scenario import run_demo
from visiondoctor.release import ApprovalService
from visiondoctor.schemas import WorkflowState
from visiondoctor.storage import SqliteRunRepository
from visiondoctor.workflow import DemoRunResult


def _git(repository: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repository), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def _copy_run(result: DemoRunResult, root: Path) -> DemoRunResult:
    copied = root / "workspace" / "runs" / result.incident.incident_id
    copied.parent.mkdir(parents=True)
    shutil.copytree(result.run_root, copied)
    return replace(result, run_root=str(copied))


def test_sqlite_import_history_and_artifact_hash_verification(
    demo_result: DemoRunResult, tmp_path: Path
) -> None:
    copied = _copy_run(demo_result, tmp_path)
    repository = SqliteRunRepository(tmp_path / "state.sqlite3")

    record = repository.import_completed_run("RUN-STORAGE", copied)

    assert record.state is WorkflowState.AWAITING_HUMAN_APPROVAL
    assert len(repository.history(record.run_id)) == len(demo_result.history)
    artifacts = repository.list_artifacts(record.run_id)
    assert len(artifacts) >= 10
    postmortem = next(item for item in artifacts if item.relative_path == "postmortem.md")
    path = Path(postmortem.absolute_path)
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"tampered")
        with pytest.raises(ValueError, match="hash has changed"):
            repository.get_artifact(record.run_id, postmortem.artifact_id)
    finally:
        path.write_bytes(original)


def test_approval_is_recoverable_and_never_merges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_gateway_factory
) -> None:
    result = run_demo(
        tmp_path / "approval-demo", model_gateway=model_gateway_factory()
    )
    repository = SqliteRunRepository(tmp_path / "approval.sqlite3")
    repository.import_completed_run("RUN-APPROVAL", result)
    service = ApprovalService(repository)
    target = Path(result.incident.repository.path)
    original_head = _git(target, "rev-parse", "HEAD").stdout.strip()
    release_path = Path(result.run_root) / "release_decision.json"
    original_release = release_path.read_bytes()
    original_record = repository.record_approval

    def fail_record(*args, **kwargs):
        raise RuntimeError("injected database failure")

    monkeypatch.setattr(repository, "record_approval", fail_record)
    with pytest.raises(RuntimeError, match="injected database failure"):
        service.act("RUN-APPROVAL", action="approve", actor="qa-owner")

    branch = f"visiondoctor/INC-001/{result.selected_candidate.candidate_id}"
    assert _git(target, "show-ref", "--verify", f"refs/heads/{branch}").returncode != 0
    assert release_path.read_bytes() == original_release
    assert not (Path(result.run_root) / "pr_materials.json").exists()
    assert repository.get_run("RUN-APPROVAL").state is WorkflowState.AWAITING_HUMAN_APPROVAL

    monkeypatch.setattr(repository, "record_approval", original_record)
    outcome = service.act(
        "RUN-APPROVAL",
        action="approve",
        actor="qa-owner",
        note="All deterministic gates reviewed.",
    )

    assert outcome.state is WorkflowState.PR_READY
    assert outcome.branch_name == branch
    assert outcome.candidate_commit
    assert _git(target, "rev-parse", "HEAD").stdout.strip() == original_head
    assert _git(target, "status", "--porcelain").stdout == ""
    assert _git(target, "rev-parse", branch).stdout.strip() == outcome.candidate_commit
    parent = _git(target, "rev-parse", f"{branch}^").stdout.strip()
    assert parent == result.incident.faulty_commit
    release = json.loads(release_path.read_text(encoding="utf-8"))
    assert release["automatic_merge"] is False
    assert release["human_action"] == "approve"
    assert (Path(result.run_root) / "pr_materials.md").is_file()


def test_api_exposes_verified_evidence_and_human_rejection(
    demo_result: DemoRunResult, tmp_path: Path
) -> None:
    copied = _copy_run(demo_result, tmp_path)
    settings = ApiSettings(
        data_root=tmp_path / "server",
        database_path=tmp_path / "server" / "state.sqlite3",
    )
    repository = SqliteRunRepository(settings.database_path)
    repository.import_completed_run("RUN-API", copied)
    client = TestClient(create_app(settings, repository))

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert client.get("/api/v1/runs").json()[0]["run_id"] == "RUN-API"
    detail = client.get("/api/v1/runs/RUN-API")
    assert detail.status_code == 200
    assert len(detail.json()["history"]) == len(demo_result.history)

    evidence = client.get("/api/v1/runs/RUN-API/evidence").json()
    assert len(evidence) == 50
    case_id = evidence[0]["case_id"]
    assert client.get(f"/api/v1/runs/RUN-API/evidence/{case_id}").status_code == 200
    rgb = client.get(f"/api/v1/runs/RUN-API/evidence/{case_id}/rgb")
    depth = client.get(f"/api/v1/runs/RUN-API/evidence/{case_id}/depth.png")
    assert rgb.headers["content-type"].startswith("image/png")
    assert depth.headers["content-type"].startswith("image/png")
    assert client.get("/api/v1/runs/RUN-API/diff").status_code == 200
    validation = client.get("/api/v1/runs/RUN-API/validation")
    assert validation.status_code == 200
    assert validation.json()["candidate_id"] == demo_result.selected_candidate.candidate_id
    assert validation.json()["case_count"] == 50

    artifacts = client.get("/api/v1/runs/RUN-API/artifacts").json()
    artifact_id = artifacts[0]["artifact_id"]
    assert client.get(f"/api/v1/runs/RUN-API/artifacts/{artifact_id}").status_code == 200

    response = client.post(
        "/api/v1/runs/RUN-API/approval",
        json={"action": "reject", "actor": "reviewer", "note": "Needs redesign."},
    )
    assert response.status_code == 200
    assert response.json()["state"] == "REJECTED_BY_HUMAN"
    assert response.json()["automatic_merge"] is False
    assert client.post(
        "/api/v1/runs/RUN-API/approval",
        json={"action": "approve", "actor": "reviewer"},
    ).status_code == 409
