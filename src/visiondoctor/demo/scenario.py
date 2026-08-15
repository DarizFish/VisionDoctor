from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Literal

from visiondoctor.adapters import (
    AdapterUnavailableError,
    DatasetEvidenceProvider,
    DatasetReferenceProvider,
    GazeboAdapter,
    GazeboFixedMotionGate,
)
from visiondoctor.adapters.dataset_execution import DatasetExecutionBackend
from visiondoctor.agents import DiagnosisAgent, PatchAgent
from visiondoctor.demo.dataset_builder import build_demo_dataset
from visiondoctor.demo.repository_builder import (
    build_demo_repository,
)
from visiondoctor.evidence import EvidenceStore
from visiondoctor.llm import ModelGateway, ModelSettings, OpenAICompatibleGateway
from visiondoctor.sandbox import (
    DockerPythonRunner,
    GitWorktreeSandbox,
    LocalPythonRunner,
    PythonSandboxRunner,
)
from visiondoctor.schemas import AcceptanceCriteria, Incident, PatchPolicy, RepositoryRef
from visiondoctor.workflow import DemoRunResult, Orchestrator


def _git_diff(repository: Path, baseline: str, faulty: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), "diff", baseline, faulty, "--"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout


def _sandbox_runner(mode: Literal["local", "docker", "auto"]) -> PythonSandboxRunner:
    if mode not in {"local", "docker", "auto"}:
        raise ValueError(f"unknown sandbox mode: {mode}")
    if mode == "local":
        return LocalPythonRunner()
    if mode in {"docker", "auto"}:
        if not DockerPythonRunner.available():
            raise AdapterUnavailableError(
                "Docker sandbox is required for model-generated code and is unavailable"
            )
        runner = DockerPythonRunner()
        project_root = Path(__file__).resolve().parents[3]
        runner.ensure_image(project_root / "docker" / "sandbox.Dockerfile", project_root)
        return runner
    return LocalPythonRunner()


def run_demo(
    workspace: Path,
    sandbox_mode: Literal["local", "docker", "auto"] = "docker",
    robot_backend: Literal["dataset", "gazebo", "auto"] = "dataset",
    *,
    model_gateway: ModelGateway | None = None,
) -> DemoRunResult:
    workspace = workspace.resolve()
    if workspace.exists() and any(workspace.iterdir()):
        raise FileExistsError(f"demo workspace must be new or empty: {workspace}")
    workspace.mkdir(parents=True, exist_ok=True)
    if model_gateway is None:
        settings = ModelSettings.from_environment()
        model_gateway = OpenAICompatibleGateway(
            settings, audit_path=workspace / "runs" / "INC-001" / "model_audit.jsonl"
        )
    runner = _sandbox_runner(sandbox_mode)
    if robot_backend not in {"dataset", "gazebo", "auto"}:
        raise ValueError(f"unknown robot backend: {robot_backend}")
    gazebo_status = GazeboAdapter.availability() if robot_backend != "dataset" else None
    use_gazebo = bool(
        gazebo_status and gazebo_status.available and gazebo_status.runtime == "docker"
    )
    if robot_backend == "gazebo" and not use_gazebo:
        reason = gazebo_status.reason if gazebo_status else "Gazebo status was not checked"
        raise AdapterUnavailableError(reason)
    dataset_root = workspace / "dataset"
    repository = build_demo_repository(workspace / "target-repository")
    cases = build_demo_dataset(dataset_root, scene_count=50)
    gazebo_rgbd_payload = None
    if use_gazebo:
        live_case = next(case for case in cases if case.case_id == "scene-049")
        capture = GazeboAdapter.run_rgbd_capture_contract(
            Path(__file__).resolve().parents[3],
            Path(live_case.manifest_path).parent,
            case_id=live_case.case_id,
        )
        if not capture.success:
            raise AdapterUnavailableError(
                str(capture.payload.get("error", "Gazebo RGB-D capture failed"))
            )
        captured_case = live_case.model_copy(
            update={
                "manifest_path": capture.payload["manifest_path"],
                "reference_path": capture.payload["reference_path"],
            }
        )
        cases = tuple(
            captured_case if case.case_id == live_case.case_id else case for case in cases
        )
        gazebo_rgbd_payload = {
            "case_id": live_case.case_id,
            "translation_error_m": capture.payload["rgbd_translation_error_m"],
            "rotation_error_rad": capture.payload["rgbd_rotation_error_rad"],
            "depth_valid_ratio": capture.payload["depth_valid_ratio"],
            "manifest_path": capture.payload["manifest_path"],
        }
    incident = Incident(
        incident_id="INC-001",
        title="Coordinate transform composition regression",
        description=(
            "The faulty vision commit completes the whitelisted fixed validation motion, "
            "but the UR5e dataset-simulated TCP reaches a pose inconsistent with RGB-D and "
            "the trusted reference signal."
        ),
        repository=RepositoryRef(path=str(repository.path)),
        baseline_commit=repository.baseline_commit,
        faulty_commit=repository.faulty_commit,
        case_set=cases,
        acceptance_criteria=AcceptanceCriteria(),
        allowed_patch_scope=PatchPolicy(),
        metadata={
            "simulation_only": True,
            "seed": 20260805,
            "sandbox": "docker" if runner.secure_for_untrusted else "local",
            "robot_backend": "gazebo" if use_gazebo else "dataset",
            "rgbd_backend": "gazebo" if use_gazebo else "dataset",
            "gazebo_rgbd_capture": gazebo_rgbd_payload,
        },
    )
    code_diff = _git_diff(repository.path, repository.baseline_commit, repository.faulty_commit)
    evidence_provider = DatasetEvidenceProvider(dataset_root, code_diff=code_diff)
    reference_provider = DatasetReferenceProvider()
    sandbox = GitWorktreeSandbox(repository.path, workspace / "worktrees")
    execution_backend = DatasetExecutionBackend(
        incident,
        sandbox,
        runner=runner,
    )
    store = EvidenceStore(workspace / "runs" / incident.incident_id)
    orchestrator = Orchestrator(
        evidence_provider=evidence_provider,
        execution_backend=execution_backend,
        reference_provider=reference_provider,
        store=store,
        diagnosis_agent=DiagnosisAgent(model_gateway),
        patch_agent=PatchAgent(model_gateway),
        release_gate=(
            GazeboFixedMotionGate(
                Path(__file__).resolve().parents[3],
                case_id="scene-049" if use_gazebo else "scene-main",
            )
            if use_gazebo
            else None
        ),
    )
    return orchestrator.run(incident)
