from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from visiondoctor.demo.scenario import run_demo
from visiondoctor.product import run_incident
from visiondoctor.schemas import Incident
from visiondoctor.storage import RunJobRecord, SqliteRunRepository
from visiondoctor.workflow import DemoRunResult

RunExecutor = Callable[..., DemoRunResult]


class DurableRunWorker:
    """Single-process worker backed by an atomically claimed SQLite queue."""

    def __init__(
        self,
        repository: SqliteRunRepository,
        *,
        executor: RunExecutor = run_demo,
        incident_executor: RunExecutor = run_incident,
        poll_interval_s: float = 0.5,
        max_attempts: int = 3,
        concurrency: int = 2,
        worker_id: str | None = None,
    ) -> None:
        if poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be positive")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if concurrency < 1:
            raise ValueError("concurrency must be positive")
        self.repository = repository
        self.executor = executor
        self.incident_executor = incident_executor
        self.poll_interval_s = poll_interval_s
        self.max_attempts = max_attempts
        self.concurrency = concurrency
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:12]}"
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._threads: list[threading.Thread] = []

    @property
    def running(self) -> bool:
        return bool(self._threads) and all(thread.is_alive() for thread in self._threads)

    def start(self) -> None:
        if self.running:
            return
        self.repository.recover_interrupted_run_jobs(max_attempts=self.max_attempts)
        self._stop.clear()
        self._threads = [
            threading.Thread(
                target=self._run,
                args=(slot,),
                name=f"visiondoctor-{self.worker_id}-{slot}",
                daemon=True,
            )
            for slot in range(1, self.concurrency + 1)
        ]
        for thread in self._threads:
            thread.start()

    def stop(self, timeout_s: float = 5.0) -> bool:
        self._stop.set()
        self._wake.set()
        if not self._threads:
            return True
        deadline = time.monotonic() + timeout_s
        for thread in self._threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        stopped = not any(thread.is_alive() for thread in self._threads)
        if stopped:
            self._threads = []
        return stopped

    def notify(self) -> None:
        self._wake.set()

    def run_once(self, lease_id: str | None = None) -> bool:
        lease_id = lease_id or self.worker_id
        job = self.repository.claim_next_run_job(lease_id)
        if job is None:
            return False
        self._execute(job, lease_id)
        return True

    def _run(self, slot: int) -> None:
        lease_id = f"{self.worker_id}-{slot}"
        while not self._stop.is_set():
            if self.run_once(lease_id):
                continue
            self._wake.wait(self.poll_interval_s)
            self._wake.clear()

    def _execute(self, job: RunJobRecord, lease_id: str) -> None:
        workspace = Path(job.workspace_path) / f"attempt-{job.attempt_count:03d}"
        try:
            if job.run_kind == "incident":
                incident = Incident.model_validate_json(job.request_json)
                if job.sandbox != "docker" or job.robot_backend == "auto":
                    raise ValueError(
                        "product job requires explicit docker sandbox and dataset/gazebo backend"
                    )
                result = self.incident_executor(
                    workspace,
                    incident,
                    sandbox_mode=job.sandbox,
                    robot_backend=job.robot_backend,
                )
            else:
                result = self.executor(
                    workspace,
                    sandbox_mode=job.sandbox,
                    robot_backend=job.robot_backend,
                )
        except Exception as exc:
            self.repository.mark_run_job_failed(job.job_id, lease_id, str(exc))
            return
        try:
            self.repository.import_completed_run(job.run_id, result)
        except Exception as exc:
            self.repository.mark_run_job_failed(
                job.job_id,
                lease_id,
                f"completed workflow could not be persisted: {exc}",
            )
            return
        self.repository.mark_run_job_succeeded(job.job_id, lease_id)
