from __future__ import annotations

import threading
import time
from pathlib import Path

from fastapi.testclient import TestClient

from visiondoctor.api.app import ApiSettings, create_app
from visiondoctor.storage import RunJobStatus, SqliteRunRepository
from visiondoctor.workflow import DemoRunResult


def _repository(tmp_path: Path) -> SqliteRunRepository:
    return SqliteRunRepository(tmp_path / "server" / "state.sqlite3")


def test_sqlite_queue_claim_and_interrupted_recovery(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    queued = repository.enqueue_run_job(
        job_id="JOB-RECOVERY",
        run_id="RUN-RECOVERY",
        workspace_path=tmp_path / "workspaces" / "RUN-RECOVERY",
        sandbox="docker",
        robot_backend="dataset",
    )

    assert queued.status is RunJobStatus.QUEUED
    claimed = repository.claim_next_run_job("worker-before-crash")
    assert claimed is not None
    assert claimed.status is RunJobStatus.RUNNING
    assert claimed.attempt_count == 1
    assert repository.claim_next_run_job("other-worker") is None

    recovered = repository.recover_interrupted_run_jobs(max_attempts=3)
    assert len(recovered) == 1
    assert recovered[0].status is RunJobStatus.QUEUED
    assert recovered[0].recovery_count == 1
    assert recovered[0].worker_id is None

    retried = repository.claim_next_run_job("worker-after-restart")
    assert retried is not None
    assert retried.attempt_count == 2
    failed = repository.mark_run_job_failed(
        retried.job_id, "worker-after-restart", "deterministic failure"
    )
    assert failed.status is RunJobStatus.FAILED
    assert failed.error == "deterministic failure"


def test_recovery_finalizes_job_when_run_was_already_imported(
    demo_result: DemoRunResult, tmp_path: Path
) -> None:
    repository = _repository(tmp_path)
    repository.enqueue_run_job(
        job_id="JOB-COMMIT-GAP",
        run_id="RUN-COMMIT-GAP",
        workspace_path=tmp_path / "workspaces" / "RUN-COMMIT-GAP",
        sandbox="local",
        robot_backend="dataset",
    )
    assert repository.claim_next_run_job("worker-that-stopped") is not None
    repository.import_completed_run("RUN-COMMIT-GAP", demo_result)

    recovered = repository.recover_interrupted_run_jobs()

    assert recovered[0].status is RunJobStatus.SUCCEEDED
    assert repository.get_run("RUN-COMMIT-GAP").selected_candidate
    assert repository.list_artifacts("RUN-COMMIT-GAP")


def test_interrupted_job_fails_after_recovery_limit(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    repository.enqueue_run_job(
        job_id="JOB-RETRY-LIMIT",
        run_id="RUN-RETRY-LIMIT",
        workspace_path=tmp_path / "workspaces" / "RUN-RETRY-LIMIT",
        sandbox="local",
        robot_backend="dataset",
    )
    for attempt in range(1, 4):
        claimed = repository.claim_next_run_job(f"worker-{attempt}")
        assert claimed is not None and claimed.attempt_count == attempt
        recovered = repository.recover_interrupted_run_jobs(max_attempts=3)[0]
        expected = RunJobStatus.FAILED if attempt == 3 else RunJobStatus.QUEUED
        assert recovered.status is expected

    assert recovered.recovery_count == 3
    assert "retry limit" in (recovered.error or "")
    assert repository.claim_next_run_job("worker-4") is None


def test_api_submits_run_without_waiting_and_worker_persists_result(
    demo_result: DemoRunResult, tmp_path: Path
) -> None:
    settings = ApiSettings(
        data_root=tmp_path / "server",
        database_path=tmp_path / "server" / "state.sqlite3",
    )
    repository = SqliteRunRepository(settings.database_path)
    started = threading.Event()
    release = threading.Event()
    workspaces: list[Path] = []

    def executor(
        workspace: Path, *, sandbox_mode: str, robot_backend: str
    ) -> DemoRunResult:
        workspaces.append(workspace)
        assert sandbox_mode == "local"
        assert robot_backend == "dataset"
        started.set()
        if not release.wait(timeout=10):
            raise TimeoutError("test did not release the queued run")
        return demo_result

    app = create_app(settings, repository, job_executor=executor)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/demo-runs",
            json={"sandbox": "local", "robot_backend": "dataset"},
        )
        assert response.status_code == 202
        submitted = response.json()
        assert submitted["status"] == "QUEUED"
        assert started.wait(timeout=5)
        assert not release.is_set()
        running = client.get(submitted["status_url"])
        assert running.status_code == 200
        assert running.json()["status"] == "RUNNING"
        assert client.get(submitted["run_url"]).status_code == 404

        release.set()
        deadline = time.monotonic() + 10
        payload: dict = {}
        while time.monotonic() < deadline:
            payload = client.get(submitted["status_url"]).json()
            if payload["status"] == "SUCCEEDED":
                break
            time.sleep(0.05)

        assert payload["status"] == "SUCCEEDED"
        assert payload["attempt_count"] == 1
        assert workspaces[0].name == "attempt-001"
        assert client.get(submitted["run_url"]).status_code == 200
        health = client.get("/health").json()
        assert health["jobs"]["SUCCEEDED"] == 1
        assert health["worker"]["running"] is True


def test_api_queues_a_caller_supplied_incident_without_demo_construction(
    demo_result: DemoRunResult, tmp_path: Path
) -> None:
    settings = ApiSettings(
        data_root=tmp_path / "external-server",
        database_path=tmp_path / "external-server" / "state.sqlite3",
    )
    repository = SqliteRunRepository(settings.database_path)
    received = threading.Event()
    release = threading.Event()
    captured: dict = {}

    def incident_executor(
        workspace: Path,
        incident,
        *,
        sandbox_mode: str,
        robot_backend: str,
    ) -> DemoRunResult:
        captured.update(
            {
                "workspace": workspace,
                "incident": incident,
                "sandbox": sandbox_mode,
                "robot_backend": robot_backend,
            }
        )
        received.set()
        if not release.wait(timeout=10):
            raise TimeoutError("test did not release the product job")
        return demo_result

    app = create_app(settings, repository, incident_executor=incident_executor)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/runs",
            json={
                "incident": demo_result.incident.model_dump(mode="json"),
                "sandbox": "docker",
                "robot_backend": "dataset",
            },
        )
        assert response.status_code == 202
        submitted = response.json()
        assert submitted["run_kind"] == "incident"
        assert received.wait(timeout=5)
        assert captured["incident"] == demo_result.incident
        assert captured["sandbox"] == "docker"
        assert captured["robot_backend"] == "dataset"
        queued = repository.get_run_job(submitted["job_id"])
        assert queued.request_json == demo_result.incident.model_dump_json()

        release.set()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if client.get(submitted["status_url"]).json()["status"] == "SUCCEEDED":
                break
            time.sleep(0.05)
        assert repository.get_run_job(submitted["job_id"]).status is RunJobStatus.SUCCEEDED
