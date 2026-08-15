from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from visiondoctor.adapters.gazebo import GazeboAdapter
from visiondoctor.demo.scenario import run_demo
from visiondoctor.intake import IntakeService, inspect_cases, inspect_repository
from visiondoctor.jobs import DurableRunWorker, RunExecutor
from visiondoctor.llm import ModelGatewayError
from visiondoctor.llm.settings import ModelConfigurationError, ModelSettings
from visiondoctor.multimodal import (
    OllamaVisionGateway,
    VisionConfigurationError,
    VisionSettings,
)
from visiondoctor.observability import find_job_run_root, read_agent_events
from visiondoctor.product import run_incident
from visiondoctor.release import ApprovalService
from visiondoctor.schemas import (
    AcceptanceCriteria,
    Incident,
    PatchPolicy,
    RepositoryRef,
    TaskSpecification,
    TestCaseRef,
    VisionProject,
)
from visiondoctor.sessions import DiagnosisSessionService
from visiondoctor.simulation import SimulationService
from visiondoctor.storage import SqliteRunRepository
from visiondoctor.tasks import supported_task_capabilities


@dataclass(frozen=True)
class ApiSettings:
    data_root: Path
    database_path: Path

    @classmethod
    def from_environment(cls) -> ApiSettings:
        data_root = Path(os.getenv("VISIONDOCTOR_DATA_ROOT", ".visiondoctor/server")).resolve()
        database_path = Path(
            os.getenv("VISIONDOCTOR_DATABASE", str(data_root / "visiondoctor.sqlite3"))
        ).resolve()
        return cls(data_root=data_root, database_path=database_path)


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateDemoRequest(ApiModel):
    sandbox: Literal["local", "docker", "auto"] = "auto"
    robot_backend: Literal["dataset", "gazebo", "auto"] = "auto"


class CreateRunRequest(ApiModel):
    incident: Incident
    sandbox: Literal["docker"] = "docker"
    robot_backend: Literal["dataset", "gazebo"] = "dataset"


class ApprovalRequest(ApiModel):
    action: Literal["approve", "reject", "additional_testing"]
    actor: str = Field(min_length=1, max_length=200)
    note: str = Field(default="", max_length=4000)


class CaseSourceRequest(ApiModel):
    case_id: str = Field(min_length=1, max_length=200)
    manifest_path: str = Field(min_length=1)
    reference_path: str = Field(min_length=1)


class IntakeTurnRequest(ApiModel):
    message: str = Field(min_length=1, max_length=8000)
    session_id: str | None = None
    repository_path: str | None = None
    baseline_commit: str | None = None
    faulty_commit: str | None = None
    runner_script: str = Field(default="runner.py", min_length=1)
    test_runner_script: str = Field(default="test_runner.py", min_length=1)
    robot_backend: Literal["dataset", "gazebo"] = "dataset"
    cases: tuple[CaseSourceRequest, ...] = ()
    acceptance_confirmed: bool = False
    patch_policy_confirmed: bool = False


class GuidedRunRequest(ApiModel):
    intake_session_id: str
    title: str = Field(min_length=1, max_length=500)
    description: str = Field(min_length=1, max_length=8000)
    repository_path: str = Field(min_length=1)
    baseline_commit: str = Field(min_length=7)
    faulty_commit: str = Field(min_length=7)
    runner_script: str = Field(default="runner.py", min_length=1)
    test_runner_script: str = Field(default="test_runner.py", min_length=1)
    robot_backend: Literal["dataset", "gazebo"] = "dataset"
    cases: tuple[CaseSourceRequest, ...] = Field(min_length=1)
    acceptance_criteria: AcceptanceCriteria = Field(default_factory=AcceptanceCriteria)
    patch_policy: PatchPolicy = Field(default_factory=PatchPolicy)
    acceptance_confirmed: bool
    patch_policy_confirmed: bool
    raw_log: str = Field(default="", max_length=12000)


class SimulationActionRequest(ApiModel):
    action: Literal[
        "start_gui",
        "run_motion",
        "run_project_observation",
        "capture_rgbd",
        "stop",
    ]
    case_id: str = Field(default="live-gazebo-001", min_length=1, max_length=200)
    session_id: str | None = Field(default=None, pattern=r"^SESSION-[a-f0-9]{12}$")


class SessionCreateRequest(ApiModel):
    repository_path: str | None = None


class SessionAttachmentRequest(ApiModel):
    name: str = Field(min_length=1, max_length=200)
    media_type: str = Field(min_length=1, max_length=100)
    content_base64: str = Field(min_length=1, max_length=14_000_000)


class SessionTurnRequest(ApiModel):
    message: str = Field(default="", max_length=12000)
    attachments: tuple[SessionAttachmentRequest, ...] = Field(default=(), max_length=8)


class SessionRepositoryRequest(ApiModel):
    repository_path: str = Field(min_length=1, max_length=2000)
    semantic_understanding: bool = True


class SessionProjectConfirmationRequest(ApiModel):
    ambiguity_id: str = Field(pattern=r"^AMB-[a-f0-9]{12}$")
    option_id: str = Field(min_length=1, max_length=300)


class SessionDatasetRequest(ApiModel):
    dataset_path: str = Field(min_length=1, max_length=2000)


class SessionRunRequest(ApiModel):
    validation_plan_confirmed: bool
    safe_change_scope_confirmed: bool


class SessionFeedbackRequest(ApiModel):
    message: str = Field(min_length=1, max_length=4000)


def _project_patch_globs(project: VisionProject | None) -> tuple[str, ...]:
    if project is None:
        return ("src/*.py", "src/**/*.py", "config/*.yaml", "tests/test_*.py")
    allowed: list[str] = []
    editable_suffixes = {
        ".py",
        ".c",
        ".cc",
        ".cpp",
        ".h",
        ".hpp",
        ".yaml",
        ".yml",
        ".json",
        ".toml",
        ".xml",
    }
    for component in project.components:
        allowed.extend(
            path
            for path in component.source_paths
            if Path(path).suffix.lower() in editable_suffixes
        )
    for asset in project.assets:
        if asset.kind.value in {"configuration", "calibration"}:
            allowed.append(asset.path)
    for path in project.validation.test_candidates:
        allowed.append(path)
        parent = str(Path(path).parent).replace("\\", "/")
        allowed.append(f"{parent}/test_*.py" if parent != "." else "test_*.py")
    normalized = tuple(dict.fromkeys(item.replace("\\", "/") for item in allowed))
    if not normalized:
        raise ValueError("项目结构图中没有可安全修改的源码或配置")
    return normalized[:500]


def _project_summary(project: VisionProject) -> dict[str, Any]:
    return {
        "project_id": project.project_id,
        "name": project.name,
        "source": project.source.model_dump(mode="json"),
        "component_count": len(project.components),
        "asset_count": len(project.assets),
        "relation_count": len(project.relations),
        "frameworks": list(project.runtime.frameworks),
        "understanding_status": project.understanding_status,
        "pending_question_count": sum(
            item.status.value == "pending" for item in project.ambiguities
        ),
        "incident_count": len(project.incident_ids),
        "revision": project.revision,
        "updated_at": project.updated_at,
    }


def _docker_status() -> dict[str, str | bool]:
    executable = shutil.which("docker")
    if executable is None:
        candidate = Path("C:/Program Files/Docker/Docker/resources/bin/docker.exe")
        executable = str(candidate) if candidate.is_file() else None
    if executable is None:
        return {"available": False, "reason": "docker CLI not found"}
    process = subprocess.run(
        [executable, "version", "--format", "{{.Server.Version}}"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
        check=False,
    )
    if process.returncode != 0:
        return {"available": False, "reason": process.stderr.strip() or "engine unavailable"}
    return {"available": True, "version": process.stdout.strip()}


def _verified_evidence_case(repository: SqliteRunRepository, run_id: str, case_id: str) -> dict:
    run = repository.get_run(run_id)
    evidence_path = Path(run.run_root) / "evidence_bundle.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    case = next((item for item in evidence["cases"] if item["case_id"] == case_id), None)
    if case is None:
        raise KeyError(case_id)
    artifacts = [case["manifest"]]
    artifacts.extend(
        case[key] for key in ("rgb", "depth") if isinstance(case.get(key), dict)
    )
    artifacts.extend(case.get("input_artifacts") or ())
    for artifact in artifacts:
        path = Path(artifact["path"])
        if not path.is_file():
            raise ValueError(f"evidence artifact is missing: {artifact['artifact_id']}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != artifact["sha256"]:
            raise ValueError(
                f"evidence artifact hash mismatch: {artifact['artifact_id']}"
            )
    return case


def create_app(
    settings: ApiSettings | None = None,
    repository: SqliteRunRepository | None = None,
    job_executor: RunExecutor = run_demo,
    incident_executor: RunExecutor = run_incident,
    intake_service: IntakeService | None = None,
    session_service: DiagnosisSessionService | None = None,
    simulation_service: SimulationService | None = None,
) -> FastAPI:
    settings = settings or ApiSettings.from_environment()
    settings.data_root.mkdir(parents=True, exist_ok=True)
    repository = repository or SqliteRunRepository(settings.database_path)
    approval_service = ApprovalService(repository)
    project_root = Path(__file__).resolve().parents[3]
    intake_service = intake_service or IntakeService(settings.data_root / "intake")
    session_service = session_service or DiagnosisSessionService(settings.data_root / "sessions")
    simulation_service = simulation_service or SimulationService(
        project_root, settings.data_root / "simulation"
    )
    job_worker = DurableRunWorker(
        repository,
        executor=job_executor,
        incident_executor=incident_executor,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        job_worker.start()
        try:
            yield
        finally:
            job_worker.stop()

    app = FastAPI(
        title="VisionDoctor API",
        version="0.3.0",
        description="Environment-decoupled machine-vision diagnosis and repair API",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.repository = repository
    app.state.job_worker = job_worker
    app.state.intake_service = intake_service
    app.state.session_service = session_service
    app.state.simulation_service = simulation_service

    @app.get("/health")
    def health() -> dict:
        gazebo = GazeboAdapter.availability()
        try:
            model: dict = ModelSettings.from_environment().public_summary()
        except ModelConfigurationError as exc:
            model = {"configured": False, "reason": str(exc)}
        try:
            vision_settings = VisionSettings.from_environment()
            vision_model: dict = OllamaVisionGateway(vision_settings).status()
        except VisionConfigurationError as exc:
            vision_model = {
                "configured": False,
                "available": False,
                "model_ready": False,
                "reason": str(exc),
            }
        return {
            "status": "ok",
            "database": str(settings.database_path),
            "jobs": repository.run_job_counts(),
            "worker": {
                "id": job_worker.worker_id,
                "running": job_worker.running,
                "capacity": job_worker.concurrency,
            },
            "docker": _docker_status(),
            "gazebo": {
                "available": gazebo.available,
                "reason": gazebo.reason,
                "runtime": gazebo.runtime,
                "image": gazebo.image,
            },
            "model": model,
            "vision_model": vision_model,
        }

    @app.get("/api/v1/capabilities")
    def capabilities() -> dict[str, Any]:
        return {
            "task_adapters": supported_task_capabilities(),
            "shared_workflow": [
                "model-driven diagnosis",
                "bounded model-generated patch",
                "isolated execution",
                "deterministic QA",
                "human approval",
            ],
            "fallback_enabled": False,
        }

    @app.post("/api/v1/runs", status_code=202, include_in_schema=False)
    def create_run(request: CreateRunRequest) -> dict:
        return enqueue_incident(request.incident, request.robot_backend)

    def enqueue_incident(incident: Incident, robot_backend: str) -> dict[str, Any]:
        run_id = f"RUN-{uuid.uuid4().hex[:12]}"
        job = repository.enqueue_run_job(
            job_id=f"JOB-{uuid.uuid4().hex[:12]}",
            run_id=run_id,
            workspace_path=settings.data_root / "workspaces" / run_id,
            sandbox="docker",
            robot_backend=robot_backend,
            run_kind="incident",
            request_json=incident.model_dump_json(),
        )
        job_worker.notify()
        return {
            **job.model_dump(mode="json"),
            "status_url": f"/api/v1/jobs/{job.job_id}",
            "run_url": f"/api/v1/runs/{run_id}",
        }

    def session_job_context(snapshot: dict[str, Any]) -> dict[str, Any]:
        jobs: list[dict[str, Any]] = []
        for job_id in snapshot.get("job_ids") or ():
            try:
                job = repository.get_run_job(str(job_id))
            except KeyError:
                continue
            value = job.model_dump(mode="json")
            root = find_job_run_root(Path(job.workspace_path), job.attempt_count)
            value["events"] = read_agent_events(root) if root else []
            if job.status == "SUCCEEDED":
                try:
                    run = repository.get_run(job.run_id)
                except KeyError:
                    pass
                else:
                    value["run_state"] = run.state
                    value["selected_candidate"] = run.selected_candidate
            jobs.append(value)
        latest = jobs[-1] if jobs else None
        return {"jobs": jobs, "latest_job": latest}

    def with_job_context(snapshot: dict[str, Any]) -> dict[str, Any]:
        job_context = session_job_context(snapshot)
        latest = job_context["latest_job"]
        phase = snapshot["phase"]
        if latest:
            if latest["status"] in {"QUEUED", "RUNNING"}:
                phase = "running"
            elif latest["status"] == "FAILED":
                phase = "needs_attention"
            elif latest.get("run_state") in {
                "AWAITING_HUMAN_APPROVAL",
                "AWAITING_TECHNICAL_REVIEW",
            }:
                phase = "review"
            else:
                phase = "completed"
        return {**snapshot, **job_context, "phase": phase}

    def public_session(session_id: str) -> dict[str, Any]:
        return with_job_context(session_service.get(session_id))

    def current_run_context(session_id: str) -> dict[str, Any]:
        snapshot = public_session(session_id)
        latest = snapshot.get("latest_job")
        if not latest:
            return {}
        context = {
            "status": latest.get("status"),
            "run_state": latest.get("run_state"),
            "error": latest.get("error"),
            "recent_activity": [
                {
                    "kind": item.get("kind"),
                    "title": item.get("title"),
                    "status": item.get("status"),
                }
                for item in (latest.get("events") or [])[-8:]
            ],
        }
        if latest.get("status") == "SUCCEEDED" and latest.get("run_id"):
            try:
                run = repository.get_run(str(latest["run_id"]))
                if run.selected_candidate:
                    path = Path(run.run_root) / f"{run.selected_candidate}_validation.json"
                    if path.is_file():
                        validation = json.loads(path.read_text(encoding="utf-8"))
                        context["validation"] = {
                            "decision": validation.get("decision"),
                            "passed_cases": validation.get("passed_cases"),
                            "failed_cases": validation.get("failed_cases"),
                            "failure_category": validation.get("failure_category"),
                            "retryable": validation.get("retryable"),
                        }
            except (KeyError, OSError, json.JSONDecodeError):
                pass
        return context

    @app.get("/api/v1/sessions")
    def list_sessions() -> list[dict]:
        return [with_job_context(dict(item)) for item in session_service.list()]

    @app.post("/api/v1/sessions", status_code=201)
    def create_session(request: SessionCreateRequest) -> dict:
        try:
            created = session_service.create(repository_path=request.repository_path)
            return public_session(str(created["session_id"]))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/v1/sessions/{session_id}")
    def get_session(session_id: str) -> dict:
        try:
            return public_session(session_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="diagnosis session not found") from exc

    @app.delete("/api/v1/sessions/{session_id}")
    def delete_session(session_id: str) -> dict[str, Any]:
        try:
            snapshot = public_session(session_id)
            if any(
                item.get("status") in {"QUEUED", "RUNNING"}
                for item in snapshot.get("jobs") or ()
            ):
                raise HTTPException(status_code=409, detail="诊断正在运行，暂时不能删除会话")
            return session_service.delete(session_id)
        except HTTPException:
            raise
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="diagnosis session not found") from exc

    @app.post("/api/v1/sessions/{session_id}/turn")
    def session_turn(session_id: str, request: SessionTurnRequest) -> dict:
        try:
            session_service.turn(
                session_id,
                message=request.message,
                attachments=tuple(item.model_dump() for item in request.attachments),
                simulation_status=simulation_service.status(),
                run_context=current_run_context(session_id),
            )
            return public_session(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="diagnosis session not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ModelGatewayError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/api/v1/sessions/{session_id}/repository")
    def connect_session_repository(session_id: str, request: SessionRepositoryRequest) -> dict:
        try:
            session_service.connect_repository(
                session_id,
                request.repository_path,
                understand=request.semantic_understanding,
            )
            return public_session(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="diagnosis session not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ModelGatewayError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/api/v1/sessions/{session_id}/project/understand")
    def understand_session_project(session_id: str) -> dict:
        try:
            session_service.understand_project(session_id)
            return public_session(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="diagnosis session not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ModelGatewayError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/api/v1/sessions/{session_id}/project/confirm")
    def confirm_session_project(
        session_id: str, request: SessionProjectConfirmationRequest
    ) -> dict:
        try:
            session_service.confirm_project_choice(
                session_id, request.ambiguity_id, request.option_id
            )
            return public_session(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="project question not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/v1/projects")
    def list_projects() -> list[dict[str, Any]]:
        return [_project_summary(item) for item in session_service.projects.list()]

    @app.get("/api/v1/projects/{project_id}")
    def get_project(project_id: str) -> dict[str, Any]:
        try:
            return session_service.projects.get(project_id).model_dump(mode="json")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc

    @app.post("/api/v1/sessions/{session_id}/dataset")
    def connect_session_dataset(session_id: str, request: SessionDatasetRequest) -> dict:
        try:
            session_service.connect_dataset(session_id, request.dataset_path)
            return public_session(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="diagnosis session not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/v1/sessions/{session_id}/simulation-capture")
    def attach_session_simulation(session_id: str) -> dict:
        latest = simulation_service.status().get("latest_operation") or {}
        if latest.get("action") != "capture_rgbd" or latest.get("status") != "SUCCEEDED":
            raise HTTPException(status_code=409, detail="请先完成一次仿真相机采集")
        if latest.get("case_id") != f"scene-{session_id[-6:]}":
            raise HTTPException(status_code=409, detail="最近的仿真采集不属于当前诊断会话")
        try:
            session_service.attach_simulation_capture(
                session_id,
                {
                    **(latest.get("result") or {}),
                    "operation_id": latest.get("operation_id"),
                },
            )
            return public_session(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="diagnosis session not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ModelGatewayError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.get("/api/v1/sessions/{session_id}/attachments/{attachment_id}")
    def download_session_attachment(session_id: str, attachment_id: str) -> FileResponse:
        try:
            path, media_type, filename = session_service.attachment_path(session_id, attachment_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="attachment not found") from exc
        return FileResponse(path, media_type=media_type, filename=filename)

    @app.post("/api/v1/sessions/{session_id}/feedback")
    def add_session_feedback(session_id: str, request: SessionFeedbackRequest) -> dict:
        try:
            session_service.add_user_feedback(session_id, request.message)
            return public_session(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="diagnosis session not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/v1/sessions/{session_id}/runs", status_code=202)
    def create_session_run(session_id: str, request: SessionRunRequest) -> dict:
        if not request.validation_plan_confirmed or not request.safe_change_scope_confirmed:
            raise HTTPException(status_code=422, detail="开始诊断前需要确认验证与修改范围")
        try:
            values = session_service.run_inputs(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="diagnosis session not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        cases = values["cases"]
        robot_backend = (
            "gazebo"
            if any("simulation" in str(item["manifest_path"]).lower() for item in cases)
            else "dataset"
        )
        project = (
            VisionProject.model_validate(values["project"])
            if values.get("project")
            else None
        )
        incident = Incident(
            incident_id=f"INC-{uuid.uuid4().hex[:10]}",
            title=values["title"],
            description=values["description"],
            repository=RepositoryRef(path=str(Path(values["repository_path"]).resolve())),
            project=project,
            baseline_commit=values["baseline_commit"],
            faulty_commit=values["current_commit"],
            case_set=tuple(
                TestCaseRef(
                    case_id=str(item["case_id"]),
                    manifest_path=str(item["manifest_path"]),
                    reference_path=str(item["reference_path"]),
                )
                for item in cases
            ),
            task=TaskSpecification.model_validate(values["task"]),
            acceptance_criteria=AcceptanceCriteria(),
            allowed_patch_scope=PatchPolicy(
                allowed_globs=_project_patch_globs(project)
            ),
            metadata={
                "simulation_only": True,
                "robot_backend": robot_backend,
                "diagnosis_session_id": session_id,
                "execution_contract": {
                    "runner_script": values["runner_script"],
                    "test_runner_script": values["test_runner_script"],
                },
                "evidence_sources": ["user", "repository", robot_backend],
            },
        )
        created = enqueue_incident(incident, robot_backend)
        session_service.record_incident(session_id, incident.incident_id)
        session_service.add_job(session_id, str(created["job_id"]))
        return {**created, "session_id": session_id}

    @app.post("/api/v1/intake/turn", include_in_schema=False)
    def intake_turn(request: IntakeTurnRequest) -> dict:
        try:
            return intake_service.turn(
                message=request.message,
                session_id=request.session_id,
                repository_path=request.repository_path,
                baseline_commit=request.baseline_commit,
                faulty_commit=request.faulty_commit,
                runner_script=request.runner_script,
                test_runner_script=request.test_runner_script,
                robot_backend=request.robot_backend,
                cases=tuple(item.model_dump() for item in request.cases),
                acceptance_confirmed=request.acceptance_confirmed,
                patch_policy_confirmed=request.patch_policy_confirmed,
                simulation_status=simulation_service.status(),
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ModelGatewayError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/api/v1/intake/runs", status_code=202, include_in_schema=False)
    def create_guided_run(request: GuidedRunRequest) -> dict:
        if not request.acceptance_confirmed or not request.patch_policy_confirmed:
            raise HTTPException(
                status_code=422,
                detail="acceptance criteria and patch policy require explicit human confirmation",
            )
        transcript = intake_service.transcript(request.intake_session_id)
        user_messages = [
            str(item.get("content", "")).strip()
            for item in transcript
            if item.get("role") == "user" and str(item.get("content", "")).strip()
        ]
        if not user_messages:
            raise HTTPException(status_code=422, detail="intake conversation is empty")
        source_cases = tuple(item.model_dump() for item in request.cases)
        repository_status = inspect_repository(
            request.repository_path,
            baseline_commit=request.baseline_commit,
            faulty_commit=request.faulty_commit,
            runner_script=request.runner_script,
            test_runner_script=request.test_runner_script,
        )
        case_status = inspect_cases(source_cases)
        contract = repository_status.get("execution_contract", {})
        if not (
            repository_status.get("available")
            and repository_status.get("baseline_valid")
            and repository_status.get("faulty_valid")
            and repository_status.get("commits_differ")
            and contract.get("runner_available")
            and contract.get("test_runner_available")
            and case_status.get("ready")
        ):
            raise HTTPException(
                status_code=422,
                detail={"repository": repository_status, "cases": case_status},
            )
        incident = Incident(
            incident_id=f"INC-{uuid.uuid4().hex[:10]}",
            title=request.title,
            description=(
                request.description.strip()
                + "\n\n补充的用户上下文：\n- "
                + "\n- ".join(user_messages)
            ),
            repository=RepositoryRef(path=str(Path(request.repository_path).resolve())),
            baseline_commit=request.baseline_commit,
            faulty_commit=request.faulty_commit,
            case_set=tuple(
                TestCaseRef(
                    case_id=item.case_id,
                    manifest_path=item.manifest_path,
                    reference_path=item.reference_path,
                )
                for item in request.cases
            ),
            acceptance_criteria=request.acceptance_criteria,
            allowed_patch_scope=request.patch_policy,
            metadata={
                "simulation_only": True,
                "robot_backend": request.robot_backend,
                "raw_log": request.raw_log,
                "intake_session_id": request.intake_session_id,
                "execution_contract": {
                    "runner_script": request.runner_script,
                    "test_runner_script": request.test_runner_script,
                },
                "evidence_sources": [
                    "user",
                    "repository",
                    *sorted(
                        {
                            "gazebo" if "gazebo" in item.manifest_path.lower() else "dataset"
                            for item in request.cases
                        }
                    ),
                ],
            },
        )
        return enqueue_incident(incident, request.robot_backend)

    @app.post("/api/v1/demo-runs", status_code=202, include_in_schema=False)
    def create_demo(request: CreateDemoRequest) -> dict:
        run_id = f"RUN-{uuid.uuid4().hex[:12]}"
        job = repository.enqueue_run_job(
            job_id=f"JOB-{uuid.uuid4().hex[:12]}",
            run_id=run_id,
            workspace_path=settings.data_root / "workspaces" / run_id,
            sandbox=request.sandbox,
            robot_backend=request.robot_backend,
        )
        job_worker.notify()
        return {
            **job.model_dump(mode="json"),
            "status_url": f"/api/v1/jobs/{job.job_id}",
            "run_url": f"/api/v1/runs/{run_id}",
        }

    @app.get("/api/v1/jobs")
    def list_jobs(limit: int = Query(default=100, ge=1, le=500)) -> list[dict]:
        return [record.model_dump(mode="json") for record in repository.list_run_jobs(limit)]

    @app.get("/api/v1/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        try:
            record = repository.get_run_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc
        return record.model_dump(mode="json")

    @app.get("/api/v1/jobs/{job_id}/agent-events")
    def job_agent_events(job_id: str) -> dict:
        try:
            record = repository.get_run_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc
        root = find_job_run_root(Path(record.workspace_path), record.attempt_count)
        return {
            "job_id": job_id,
            "status": record.status,
            "events": read_agent_events(root) if root else [],
        }

    @app.get("/api/v1/runs")
    def list_runs(limit: int = Query(default=100, ge=1, le=500)) -> list[dict]:
        return [record.model_dump(mode="json") for record in repository.list_runs(limit)]

    @app.get("/api/v1/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        try:
            record = repository.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        return {
            **record.model_dump(mode="json"),
            "history": repository.history(run_id),
        }

    @app.get("/api/v1/runs/{run_id}/agent-events")
    def run_agent_events(run_id: str) -> dict:
        try:
            run = repository.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        return {"run_id": run_id, "events": read_agent_events(Path(run.run_root))}

    @app.get("/api/v1/simulation")
    def simulation_status() -> dict:
        return simulation_service.status()

    @app.post("/api/v1/simulation/actions", status_code=202)
    def simulation_action(request: SimulationActionRequest) -> dict:
        try:
            project_observation = None
            if request.action == "run_project_observation":
                if request.session_id is None:
                    raise ValueError("project observation requires a diagnosis session")
                session = session_service.get(request.session_id)
                if not session["repository"].get("available"):
                    raise ValueError("connect a project before running an observation")
                captured_cases = [
                    item
                    for item in session["evidence"].get("items") or ()
                    if "simulation" in str(item.get("manifest_path", "")).lower()
                ]
                if not captured_cases:
                    raise ValueError("capture RGB-D evidence in this session first")
                case = captured_cases[-1]
                project_observation = {
                    "repository_path": str(session["repository"]["path"]),
                    "runner_script": str(session["execution_contract"]["runner_script"]),
                    "case_id": str(case["case_id"]),
                    "manifest_path": str(case["manifest_path"]),
                }
            if project_observation is None:
                return simulation_service.start(request.action, case_id=request.case_id)
            return simulation_service.start(
                request.action,
                case_id=request.case_id,
                project_observation=project_observation,
            )
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/v1/runs/{run_id}/artifacts")
    def list_artifacts(run_id: str) -> list[dict]:
        try:
            records = repository.list_artifacts(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        return [
            {
                "artifact_id": record.artifact_id,
                "artifact_type": record.artifact_type,
                "relative_path": record.relative_path,
                "sha256": record.sha256,
                "size_bytes": record.size_bytes,
                "media_type": record.media_type,
            }
            for record in records
        ]

    @app.get("/api/v1/runs/{run_id}/artifacts/{artifact_id}")
    def download_artifact(run_id: str, artifact_id: int) -> FileResponse:
        try:
            artifact = repository.get_artifact(run_id, artifact_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="artifact not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return FileResponse(
            artifact.absolute_path,
            media_type=artifact.media_type,
            filename=Path(artifact.relative_path).name,
        )

    @app.get("/api/v1/runs/{run_id}/evidence")
    def evidence_index(run_id: str) -> list[dict]:
        try:
            run = repository.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        evidence = json.loads((Path(run.run_root) / "evidence_bundle.json").read_text("utf-8"))
        return [
            {
                "case_id": case["case_id"],
                "task_kind": case.get("task_kind", "rgbd_pose"),
                "expected_pixel": case.get("expected_pixel"),
                "depth_valid_ratio": case.get("depth_valid_ratio"),
                "structured_input": case.get("structured_input"),
                "input_artifacts": [
                    {
                        "artifact_id": artifact["artifact_id"],
                        "media_type": artifact["media_type"],
                    }
                    for artifact in case.get("input_artifacts") or ()
                ],
                "source": case["source"],
            }
            for case in evidence["cases"]
        ]

    @app.get("/api/v1/runs/{run_id}/evidence/{case_id}")
    def evidence_case(run_id: str, case_id: str) -> dict:
        try:
            return _verified_evidence_case(repository, run_id, case_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="run or case not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/v1/runs/{run_id}/evidence/{case_id}/rgb")
    def evidence_rgb(run_id: str, case_id: str) -> FileResponse:
        try:
            case = _verified_evidence_case(repository, run_id, case_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="run or case not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if not isinstance(case.get("rgb"), dict):
            raise HTTPException(status_code=404, detail="case has no RGB artifact")
        return FileResponse(case["rgb"]["path"], media_type="image/png")

    @app.get("/api/v1/runs/{run_id}/evidence/{case_id}/artifacts/{artifact_id}")
    def evidence_input_artifact(run_id: str, case_id: str, artifact_id: str) -> FileResponse:
        try:
            case = _verified_evidence_case(repository, run_id, case_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="run or case not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        artifact = next(
            (
                item
                for item in case.get("input_artifacts") or ()
                if item.get("artifact_id") == artifact_id
            ),
            None,
        )
        if artifact is None:
            raise HTTPException(status_code=404, detail="input artifact not found")
        return FileResponse(
            artifact["path"],
            media_type=artifact["media_type"],
            filename=Path(artifact["path"]).name,
        )

    @app.get("/api/v1/runs/{run_id}/evidence/{case_id}/depth.png")
    def evidence_depth(run_id: str, case_id: str) -> StreamingResponse:
        try:
            case = _verified_evidence_case(repository, run_id, case_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="run or case not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if not isinstance(case.get("depth"), dict):
            raise HTTPException(status_code=404, detail="case has no depth artifact")
        depth = np.load(case["depth"]["path"], allow_pickle=False)
        valid = np.isfinite(depth) & (depth > 0)
        normalized = np.zeros(depth.shape, dtype=np.uint8)
        if valid.any():
            low, high = np.percentile(depth[valid], [1, 99])
            if high <= low:
                high = low + 1e-6
            normalized[valid] = np.clip((depth[valid] - low) / (high - low) * 255, 0, 255)
        image = Image.fromarray(normalized)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="image/png")

    @app.get("/api/v1/runs/{run_id}/diff", response_class=PlainTextResponse)
    def selected_diff(run_id: str) -> str:
        try:
            run = repository.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        if not run.selected_candidate:
            raise HTTPException(status_code=409, detail="run has no selected candidate")
        path = Path(run.run_root) / f"{run.selected_candidate}.diff"
        return path.read_text(encoding="utf-8")

    @app.get("/api/v1/runs/{run_id}/validation")
    def selected_validation(run_id: str) -> dict:
        try:
            run = repository.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        if not run.selected_candidate:
            raise HTTPException(status_code=409, detail="run has no selected candidate")
        path = Path(run.run_root) / f"{run.selected_candidate}_validation.json"
        if not path.is_file():
            raise HTTPException(status_code=409, detail="selected validation is unavailable")
        return json.loads(path.read_text(encoding="utf-8"))

    @app.post("/api/v1/runs/{run_id}/approval")
    def approval(run_id: str, request: ApprovalRequest) -> dict:
        try:
            outcome = approval_service.act(
                run_id,
                action=request.action,
                actor=request.actor,
                note=request.note,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "run_id": outcome.run_id,
            "state": outcome.state,
            "approval": outcome.approval.model_dump(mode="json"),
            "branch_name": outcome.branch_name,
            "candidate_commit": outcome.candidate_commit,
            "automatic_merge": False,
        }

    return app
