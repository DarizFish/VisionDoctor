from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Literal

from visiondoctor.adapters import (
    AdapterUnavailableError,
    DatasetEvidenceProvider,
    DatasetReferenceProvider,
    GazeboFixedMotionGate,
)
from visiondoctor.adapters.dataset_execution import DatasetExecutionBackend
from visiondoctor.agents import DiagnosisAgent, PatchAgent
from visiondoctor.evidence import EvidenceStore
from visiondoctor.llm import ModelGateway, ModelSettings, OpenAICompatibleGateway
from visiondoctor.multimodal import OllamaVisionGateway, VisionSettings
from visiondoctor.sandbox import DockerPythonRunner, GitWorktreeSandbox
from visiondoctor.schemas import Incident, TaskKind
from visiondoctor.tasks import get_task_adapter
from visiondoctor.workflow import DemoRunResult, Orchestrator


def load_incident(path: Path) -> Incident:
    return Incident.model_validate_json(path.read_text(encoding="utf-8"))


def run_incident(
    workspace: Path,
    incident: Incident,
    *,
    sandbox_mode: Literal["docker"] = "docker",
    robot_backend: Literal["dataset", "gazebo"] = "dataset",
    model_gateway: ModelGateway | None = None,
    max_patch_attempts: int = 3,
) -> DemoRunResult:
    """Run a caller-supplied incident. No demo data, repository, or patch is synthesized."""

    if sandbox_mode != "docker":
        raise ValueError("product runs require the Docker sandbox; no local fallback is allowed")
    if robot_backend not in {"dataset", "gazebo"}:
        raise ValueError("robot_backend must be explicitly dataset or gazebo")
    workspace = workspace.resolve()
    repository = Path(incident.repository.path).resolve()
    _validate_product_inputs(workspace, repository, incident)
    task_adapter = get_task_adapter(incident.task.kind)
    if robot_backend == "gazebo" and not task_adapter.supports_fixed_motion:
        raise ValueError(
            f"Gazebo fixed motion is not applicable to task {incident.task.kind}; "
            "select the dataset backend explicitly"
        )
    workspace.mkdir(parents=True, exist_ok=True)

    runner = DockerPythonRunner()
    if not runner.available():
        raise AdapterUnavailableError(
            "Docker is required to execute model-generated patches and is unavailable"
        )
    project_root = Path(__file__).resolve().parents[2]
    runner.ensure_image(project_root / "docker" / "sandbox.Dockerfile", project_root)

    if model_gateway is None:
        settings = ModelSettings.from_environment()
        model_gateway = OpenAICompatibleGateway(
            settings,
            audit_path=workspace
            / "runs"
            / incident.incident_id
            / "model_audit.jsonl",
        )

    visual_tasks = {TaskKind.DETECTION, TaskKind.OCR, TaskKind.SEGMENTATION}
    vision_gateway = (
        OllamaVisionGateway(
            VisionSettings.from_environment(),
            audit_path=workspace
            / "runs"
            / incident.incident_id
            / "vision_model_audit.jsonl",
        )
        if incident.task.kind in visual_tasks
        else None
    )

    diff = _git_diff(repository, incident.baseline_commit, incident.faulty_commit)
    dataset_root = Path(
        os.path.commonpath(
            [str(Path(case.manifest_path).resolve().parent) for case in incident.case_set]
        )
    )
    evidence_provider = DatasetEvidenceProvider(
        dataset_root,
        code_diff=diff,
        raw_log=str(incident.metadata.get("raw_log", "")),
    )
    reference_provider = DatasetReferenceProvider(incident.task.kind)
    execution_backend = DatasetExecutionBackend(
        incident,
        GitWorktreeSandbox(repository, workspace / "worktrees"),
        runner=runner,
    )
    presentation_case = str(
        incident.metadata.get("presentation_case_id", incident.case_set[0].case_id)
    )
    release_gate = (
        GazeboFixedMotionGate(project_root, case_id=presentation_case)
        if robot_backend == "gazebo"
        else None
    )
    store = EvidenceStore(workspace / "runs" / incident.incident_id)
    orchestrator = Orchestrator(
        evidence_provider=evidence_provider,
        execution_backend=execution_backend,
        reference_provider=reference_provider,
        store=store,
        diagnosis_agent=DiagnosisAgent(
            model_gateway,
            vision_gateway=vision_gateway,
        ),
        patch_agent=PatchAgent(model_gateway),
        release_gate=release_gate,
        max_patch_attempts=max_patch_attempts,
    )
    return orchestrator.run(incident)


def _validate_product_inputs(workspace: Path, repository: Path, incident: Incident) -> None:
    if workspace.exists() and any(workspace.iterdir()):
        raise FileExistsError(f"run workspace must be new or empty: {workspace}")
    if not repository.is_dir():
        raise FileNotFoundError(f"repository does not exist: {repository}")
    paths_overlap = (
        workspace == repository
        or repository in workspace.parents
        or workspace in repository.parents
    )
    if paths_overlap:
        raise ValueError("run workspace and target repository must not contain one another")
    if incident.project is not None:
        project_repository = Path(
            incident.project.source.repository_path
        ).expanduser().resolve()
        if project_repository != repository:
            raise ValueError("incident project snapshot belongs to a different repository")
        if incident.project.source.head_commit != incident.faulty_commit:
            raise ValueError(
                "incident faulty commit does not match the canonical project snapshot"
            )
        if any(item.status.value == "pending" for item in incident.project.ambiguities):
            raise ValueError(
                "incident project still has unresolved production relationships"
            )
    inside = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        raise ValueError(f"not a Git repository: {repository}")
    for commit in (incident.baseline_commit, incident.faulty_commit):
        check = subprocess.run(
            ["git", "-C", str(repository), "cat-file", "-e", f"{commit}^{{commit}}"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if check.returncode != 0:
            raise ValueError(f"incident commit is unavailable: {commit}")
    for case in incident.case_set:
        evidence_path = Path(case.manifest_path).resolve()
        reference_path = Path(case.reference_path).resolve()
        if not evidence_path.is_file() or not reference_path.is_file():
            raise FileNotFoundError(f"case files are missing for {case.case_id}")
        if evidence_path == reference_path:
            raise ValueError(f"case evidence/reference separation violated: {case.case_id}")


def _git_diff(repository: Path, baseline: str, faulty: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(repository), "diff", "--no-ext-diff", baseline, faulty, "--"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if process.returncode != 0:
        raise ValueError("could not diff baseline and faulty commits")
    if not process.stdout.strip():
        raise ValueError("baseline and faulty commits have no source diff")
    return process.stdout
