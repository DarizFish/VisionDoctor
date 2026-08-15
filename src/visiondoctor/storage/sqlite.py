from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict

from visiondoctor.schemas import WorkflowState
from visiondoctor.workflow.state_machine import ALLOWED_TRANSITIONS

if TYPE_CHECKING:
    from visiondoctor.workflow.orchestrator import DemoRunResult


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class StorageModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RunRecord(StorageModel):
    run_id: str
    incident_id: str
    workspace_path: str
    run_root: str
    state: WorkflowState
    selected_candidate: str | None
    created_at: datetime
    updated_at: datetime
    version: int


class ArtifactRecord(StorageModel):
    artifact_id: int
    run_id: str
    artifact_type: str
    relative_path: str
    absolute_path: str
    sha256: str
    size_bytes: int
    media_type: str
    created_at: datetime


class ApprovalRecord(StorageModel):
    approval_id: int
    run_id: str
    candidate_id: str
    action: Literal["approve", "reject", "additional_testing"]
    actor: str
    note: str
    branch_name: str | None
    candidate_commit: str | None
    created_at: datetime


class RunJobStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class RunJobRecord(StorageModel):
    job_id: str
    run_id: str
    workspace_path: str
    sandbox: Literal["local", "docker", "auto"]
    robot_backend: Literal["dataset", "gazebo", "auto"]
    run_kind: Literal["demo", "incident"]
    request_json: str
    status: RunJobStatus
    attempt_count: int
    recovery_count: int
    worker_id: str | None
    error: str | None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    completed_at: datetime | None


class SqliteRunRepository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path.resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    incident_id TEXT NOT NULL,
                    workspace_path TEXT NOT NULL,
                    run_root TEXT NOT NULL,
                    state TEXT NOT NULL,
                    selected_candidate TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1
                );
                CREATE INDEX IF NOT EXISTS idx_runs_updated_at ON runs(updated_at DESC);
                CREATE TABLE IF NOT EXISTS state_transitions (
                    transition_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                    from_state TEXT,
                    to_state TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                    artifact_type TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    absolute_path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    media_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(run_id, relative_path)
                );
                CREATE TABLE IF NOT EXISTS approvals (
                    approval_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                    candidate_id TEXT NOT NULL,
                    action TEXT NOT NULL CHECK(
                        action IN ('approve', 'reject', 'additional_testing')
                    ),
                    actor TEXT NOT NULL,
                    note TEXT NOT NULL,
                    branch_name TEXT,
                    candidate_commit TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS run_jobs (
                    job_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL UNIQUE,
                    workspace_path TEXT NOT NULL,
                    sandbox TEXT NOT NULL CHECK(sandbox IN ('local', 'docker', 'auto')),
                    robot_backend TEXT NOT NULL CHECK(
                        robot_backend IN ('dataset', 'gazebo', 'auto')
                    ),
                    run_kind TEXT NOT NULL DEFAULT 'demo' CHECK(
                        run_kind IN ('demo', 'incident')
                    ),
                    request_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL CHECK(
                        status IN ('QUEUED', 'RUNNING', 'SUCCEEDED', 'FAILED')
                    ),
                    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK(attempt_count >= 0),
                    recovery_count INTEGER NOT NULL DEFAULT 0 CHECK(recovery_count >= 0),
                    worker_id TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_run_jobs_status_created
                    ON run_jobs(status, created_at);
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(run_jobs)").fetchall()
            }
            if "run_kind" not in columns:
                connection.execute(
                    "ALTER TABLE run_jobs ADD COLUMN run_kind TEXT NOT NULL DEFAULT 'demo'"
                )
            if "request_json" not in columns:
                connection.execute(
                    "ALTER TABLE run_jobs ADD COLUMN request_json TEXT NOT NULL DEFAULT '{}'"
                )

    def enqueue_run_job(
        self,
        *,
        job_id: str,
        run_id: str,
        workspace_path: Path,
        sandbox: Literal["local", "docker", "auto"],
        robot_backend: Literal["dataset", "gazebo", "auto"],
        run_kind: Literal["demo", "incident"] = "demo",
        request_json: str = "{}",
    ) -> RunJobRecord:
        timestamp = _now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO run_jobs(
                    job_id, run_id, workspace_path, sandbox, robot_backend,
                    run_kind, request_json, status,
                    attempt_count, recovery_count, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'QUEUED', 0, 0, ?, ?)
                """,
                (
                    job_id,
                    run_id,
                    str(workspace_path.resolve()),
                    sandbox,
                    robot_backend,
                    run_kind,
                    request_json,
                    timestamp,
                    timestamp,
                ),
            )
        return RunJobRecord(
            job_id=job_id,
            run_id=run_id,
            workspace_path=str(workspace_path.resolve()),
            sandbox=sandbox,
            robot_backend=robot_backend,
            run_kind=run_kind,
            request_json=request_json,
            status=RunJobStatus.QUEUED,
            attempt_count=0,
            recovery_count=0,
            worker_id=None,
            error=None,
            created_at=timestamp,
            updated_at=timestamp,
            started_at=None,
            completed_at=None,
        )

    def get_run_job(self, job_id: str) -> RunJobRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM run_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        if row is None:
            raise KeyError(job_id)
        return RunJobRecord.model_validate(dict(row))

    def get_run_job_for_run(self, run_id: str) -> RunJobRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM run_jobs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise KeyError(run_id)
        return RunJobRecord.model_validate(dict(row))

    def list_run_jobs(self, limit: int = 100) -> tuple[RunJobRecord, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM run_jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return tuple(RunJobRecord.model_validate(dict(row)) for row in rows)

    def run_job_counts(self) -> dict[str, int]:
        counts = {status.value: 0 for status in RunJobStatus}
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM run_jobs GROUP BY status"
            ).fetchall()
        for row in rows:
            counts[str(row["status"])] = int(row["count"])
        return counts

    def claim_next_run_job(self, worker_id: str) -> RunJobRecord | None:
        timestamp = _now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT job_id FROM run_jobs
                WHERE status = 'QUEUED'
                ORDER BY created_at, job_id
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                return None
            job_id = str(row["job_id"])
            cursor = connection.execute(
                """
                UPDATE run_jobs SET
                    status = 'RUNNING', attempt_count = attempt_count + 1,
                    worker_id = ?, error = NULL, started_at = ?, completed_at = NULL,
                    updated_at = ?
                WHERE job_id = ? AND status = 'QUEUED'
                """,
                (worker_id, timestamp, timestamp, job_id),
            )
            if cursor.rowcount != 1:
                return None
            claimed = connection.execute(
                "SELECT * FROM run_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        if claimed is None:
            raise RuntimeError("claimed job disappeared")
        return RunJobRecord.model_validate(dict(claimed))

    def mark_run_job_succeeded(self, job_id: str, worker_id: str) -> RunJobRecord:
        timestamp = _now()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE run_jobs SET
                    status = 'SUCCEEDED', worker_id = NULL, error = NULL,
                    completed_at = ?, updated_at = ?
                WHERE job_id = ? AND status = 'RUNNING' AND worker_id = ?
                """,
                (timestamp, timestamp, job_id, worker_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("job completion lost its worker lease")
        return self.get_run_job(job_id)

    def mark_run_job_failed(
        self, job_id: str, worker_id: str, error: str
    ) -> RunJobRecord:
        timestamp = _now()
        normalized_error = error.strip()[:8000] or "run failed without an error message"
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE run_jobs SET
                    status = 'FAILED', worker_id = NULL, error = ?,
                    completed_at = ?, updated_at = ?
                WHERE job_id = ? AND status = 'RUNNING' AND worker_id = ?
                """,
                (normalized_error, timestamp, timestamp, job_id, worker_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("job failure lost its worker lease")
        return self.get_run_job(job_id)

    def recover_interrupted_run_jobs(self, *, max_attempts: int = 3) -> tuple[RunJobRecord, ...]:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        timestamp = _now()
        completed_run_ids: list[str] = []
        recovered_job_ids: list[str] = []
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT * FROM run_jobs WHERE status = 'RUNNING' ORDER BY created_at"
            ).fetchall()
            for row in rows:
                job_id = str(row["job_id"])
                run_id = str(row["run_id"])
                run_exists = connection.execute(
                    "SELECT 1 FROM runs WHERE run_id = ?", (run_id,)
                ).fetchone()
                if run_exists is not None:
                    connection.execute(
                        """
                        UPDATE run_jobs SET
                            status = 'SUCCEEDED', worker_id = NULL, error = NULL,
                            completed_at = ?, updated_at = ?
                        WHERE job_id = ?
                        """,
                        (timestamp, timestamp, job_id),
                    )
                    completed_run_ids.append(run_id)
                elif int(row["attempt_count"]) >= max_attempts:
                    connection.execute(
                        """
                        UPDATE run_jobs SET
                            status = 'FAILED', worker_id = NULL,
                            error = 'worker interrupted and retry limit was reached',
                            completed_at = ?, updated_at = ?,
                            recovery_count = recovery_count + 1
                        WHERE job_id = ?
                        """,
                        (timestamp, timestamp, job_id),
                    )
                else:
                    connection.execute(
                        """
                        UPDATE run_jobs SET
                            status = 'QUEUED', worker_id = NULL,
                            error = 'worker interrupted; queued for a fresh attempt',
                            started_at = NULL, completed_at = NULL, updated_at = ?,
                            recovery_count = recovery_count + 1
                        WHERE job_id = ?
                        """,
                        (timestamp, job_id),
                    )
                recovered_job_ids.append(job_id)
        for run_id in completed_run_ids:
            self.refresh_artifacts(run_id)
        return tuple(self.get_run_job(job_id) for job_id in recovered_job_ids)

    def import_completed_run(self, run_id: str, result: DemoRunResult) -> RunRecord:
        run_root = Path(result.run_root).resolve()
        workspace = run_root.parents[1]
        timestamp = _now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runs(
                    run_id, incident_id, workspace_path, run_root, state,
                    selected_candidate, created_at, updated_at, version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    run_id,
                    result.incident.incident_id,
                    str(workspace),
                    str(run_root),
                    result.state.value,
                    result.selected_candidate.candidate_id,
                    timestamp,
                    timestamp,
                ),
            )
            previous: WorkflowState | None = None
            for state in result.history:
                connection.execute(
                    """
                    INSERT INTO state_transitions(
                        run_id, from_state, to_state, metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        previous.value if previous else None,
                        state.value,
                        "{}",
                        timestamp,
                    ),
                )
                previous = state
        self.refresh_artifacts(run_id)
        return self.get_run(run_id)

    def refresh_artifacts(self, run_id: str) -> tuple[ArtifactRecord, ...]:
        run = self.get_run(run_id)
        run_root = Path(run.run_root).resolve()
        if not run_root.is_dir():
            raise FileNotFoundError(f"run root does not exist: {run_root}")
        timestamp = _now()
        with self._connect() as connection:
            for path in sorted(item for item in run_root.rglob("*") if item.is_file()):
                resolved = path.resolve()
                if run_root not in resolved.parents:
                    raise ValueError("artifact escaped run root")
                relative = resolved.relative_to(run_root).as_posix()
                connection.execute(
                    """
                    INSERT INTO artifacts(
                        run_id, artifact_type, relative_path, absolute_path,
                        sha256, size_bytes, media_type, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id, relative_path) DO UPDATE SET
                        artifact_type=excluded.artifact_type,
                        absolute_path=excluded.absolute_path,
                        sha256=excluded.sha256,
                        size_bytes=excluded.size_bytes,
                        media_type=excluded.media_type,
                        created_at=excluded.created_at
                    """,
                    (
                        run_id,
                        self._artifact_type(relative),
                        relative,
                        str(resolved),
                        _sha256(resolved),
                        resolved.stat().st_size,
                        self._media_type(resolved),
                        timestamp,
                    ),
                )
        return self.list_artifacts(run_id)

    def get_run(self, run_id: str) -> RunRecord:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return RunRecord.model_validate(dict(row))

    def list_runs(self, limit: int = 100) -> tuple[RunRecord, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM runs ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return tuple(RunRecord.model_validate(dict(row)) for row in rows)

    def list_artifacts(self, run_id: str) -> tuple[ArtifactRecord, ...]:
        self.get_run(run_id)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM artifacts WHERE run_id = ? ORDER BY relative_path", (run_id,)
            ).fetchall()
        return tuple(ArtifactRecord.model_validate(dict(row)) for row in rows)

    def get_artifact(self, run_id: str, artifact_id: int) -> ArtifactRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM artifacts WHERE run_id = ? AND artifact_id = ?",
                (run_id, artifact_id),
            ).fetchone()
        if row is None:
            raise KeyError((run_id, artifact_id))
        artifact = ArtifactRecord.model_validate(dict(row))
        path = Path(artifact.absolute_path)
        if not path.is_file() or _sha256(path) != artifact.sha256:
            raise ValueError("artifact is missing or its hash has changed")
        return artifact

    def history(self, run_id: str) -> tuple[dict[str, Any], ...]:
        self.get_run(run_id)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT from_state, to_state, metadata_json, created_at
                FROM state_transitions WHERE run_id = ? ORDER BY transition_id
                """,
                (run_id,),
            ).fetchall()
        return tuple(
            {
                "from_state": row["from_state"],
                "to_state": row["to_state"],
                "metadata": json.loads(row["metadata_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        )

    def record_approval(
        self,
        run_id: str,
        *,
        candidate_id: str,
        action: Literal["approve", "reject", "additional_testing"],
        actor: str,
        note: str,
        branch_name: str | None = None,
        candidate_commit: str | None = None,
    ) -> ApprovalRecord:
        target = {
            "approve": WorkflowState.PR_READY,
            "reject": WorkflowState.REJECTED_BY_HUMAN,
            "additional_testing": WorkflowState.ADDITIONAL_TESTING,
        }[action]
        timestamp = _now()
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(run_id)
            current = WorkflowState(row["state"])
            if target not in ALLOWED_TRANSITIONS.get(current, frozenset()):
                raise ValueError(f"illegal approval transition: {current} -> {target}")
            cursor = connection.execute(
                """
                UPDATE runs SET state = ?, updated_at = ?, version = version + 1
                WHERE run_id = ? AND version = ?
                """,
                (target.value, timestamp, run_id, row["version"]),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("run was updated concurrently")
            connection.execute(
                """
                INSERT INTO state_transitions(
                    run_id, from_state, to_state, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    current.value,
                    target.value,
                    json.dumps({"actor": actor, "action": action}, sort_keys=True),
                    timestamp,
                ),
            )
            cursor = connection.execute(
                """
                INSERT INTO approvals(
                    run_id, candidate_id, action, actor, note,
                    branch_name, candidate_commit, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    candidate_id,
                    action,
                    actor,
                    note,
                    branch_name,
                    candidate_commit,
                    timestamp,
                ),
            )
            approval_id = int(cursor.lastrowid)
        return self.get_approval(approval_id)

    def get_approval(self, approval_id: int) -> ApprovalRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM approvals WHERE approval_id = ?", (approval_id,)
            ).fetchone()
        if row is None:
            raise KeyError(approval_id)
        return ApprovalRecord.model_validate(dict(row))

    @staticmethod
    def _artifact_type(relative_path: str) -> str:
        name = Path(relative_path).name
        if "validation" in name:
            return "validation"
        if "external_gate" in name or name.endswith("_gazebo.log"):
            return "external_gate"
        if name.endswith((".diff", ".patch")):
            return "patch"
        if name == "evidence_bundle.json":
            return "evidence"
        if name in {"release_decision.json", "pr_materials.json", "pr_materials.md"}:
            return "release"
        if name == "trace.jsonl":
            return "trace"
        return "artifact"

    @staticmethod
    def _media_type(path: Path) -> str:
        return {
            ".json": "application/json",
            ".jsonl": "application/x-ndjson",
            ".md": "text/markdown",
            ".diff": "text/x-diff",
            ".patch": "text/x-diff",
            ".png": "image/png",
            ".npy": "application/x-npy",
        }.get(path.suffix.lower(), "application/octet-stream")
