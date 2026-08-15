from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from visiondoctor.adapters import DatasetEvidenceProvider, DatasetReferenceProvider
from visiondoctor.adapters.dataset_execution import DatasetExecutionBackend
from visiondoctor.agents import QAAgent
from visiondoctor.sandbox import GitWorktreeSandbox, LocalPythonRunner
from visiondoctor.schemas import (
    AcceptanceCriteria,
    CandidateKind,
    CandidateVersion,
    Decision,
    Incident,
    PatchPolicy,
    RepositoryRef,
    TaskKind,
    TaskSpecification,
)
from visiondoctor.schemas import TestCaseRef as CaseRef
from visiondoctor.sessions import DiagnosisSessionService
from visiondoctor.tasks import get_task_adapter, supported_task_capabilities

_RUNNER = """from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from preprocess import summarize

payload = json.load(sys.stdin)
results = []
for case in payload["cases"]:
    results.append({
        "case_id": case["case_id"],
        "output": summarize(case["input"]),
        "latency_s": 0.001,
    })
print(json.dumps({
    "results": results,
    "calibration_latency_s": 0.001,
    "benchmark_samples_s": [0.001] * 15,
    "calibration_samples_s": [0.001] * 15,
}))
"""

_TEST_RUNNER = """import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from preprocess import summarize
result = summarize({"pixels": [0, 0]})
assert result["mean"] == 0.0
print("Ran 1 test")
"""

_BASELINE = """def summarize(value):
    pixels = value["pixels"]
    normalized = [pixel / 255.0 for pixel in pixels]
    return {
        "mean": sum(normalized) / len(normalized),
        "bright_count": sum(item >= 0.5 for item in normalized),
        "label": "mixed",
    }
"""

_FAULTY = _BASELINE.replace("pixel / 255.0", "pixel / 256.0")


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Incident, Path]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-b", "main")
    _git(repository, "config", "user.email", "tests@visiondoctor.local")
    _git(repository, "config", "user.name", "VisionDoctor Tests")
    _write(repository / "src" / "__init__.py", "")
    _write(repository / "src" / "preprocess.py", _BASELINE)
    _write(repository / "runner.py", _RUNNER)
    _write(repository / "test_runner.py", _TEST_RUNNER)
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "known good preprocessing")
    baseline = _git(repository, "rev-parse", "HEAD")
    _write(repository / "src" / "preprocess.py", _FAULTY)
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "regress normalization divisor")
    faulty = _git(repository, "rev-parse", "HEAD")

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "frame.bin").write_bytes(bytes([0, 255]))
    captured_at = "2026-08-09T00:00:00Z"
    manifest = dataset / "case.json"
    reference = dataset / "case.reference.json"
    _write(
        manifest,
        json.dumps(
            {
                "task_kind": "structured_output",
                "case_id": "normalize-001",
                "source": "dataset",
                "captured_at": captured_at,
                "input": {"pixels": [0, 255]},
                "artifacts": [
                    {
                        "artifact_id": "frame",
                        "path": "frame.bin",
                        "media_type": "application/octet-stream",
                    }
                ],
            }
        ),
    )
    _write(
        reference,
        json.dumps(
            {
                "task_kind": "structured_output",
                "case_id": "normalize-001",
                "source_type": "dataset_label",
                "captured_at": captured_at,
                "expected_output": {
                    "mean": 0.5,
                    "bright_count": 1,
                    "label": "mixed",
                },
            }
        ),
    )
    incident = Incident(
        incident_id="INC-STRUCTURED",
        title="preprocessing normalization regression",
        description="Intensity normalization changed after a dependency update.",
        repository=RepositoryRef(path=str(repository)),
        baseline_commit=baseline,
        faulty_commit=faulty,
        case_set=(
            CaseRef(
                case_id="normalize-001",
                manifest_path=str(manifest),
                reference_path=str(reference),
            ),
        ),
        task=TaskSpecification(
            kind=TaskKind.STRUCTURED_OUTPUT,
            display_name="Image preprocessing summary",
            case_pass_rate=1.0,
            numeric_absolute_tolerance=1e-9,
            numeric_relative_tolerance=1e-9,
        ),
        acceptance_criteria=AcceptanceCriteria(latency_growth_ratio=0.1),
        allowed_patch_scope=PatchPolicy(allowed_globs=("src/*.py", "tests/test_*.py")),
        metadata={
            "execution_contract": {
                "runner_script": "runner.py",
                "test_runner_script": "test_runner.py",
            }
        },
    )
    return incident, dataset


def _execute(
    backend: DatasetExecutionBackend, incident: Incident, candidate: CandidateVersion
):
    handle = backend.prepare(candidate)
    for case in incident.case_set:
        backend.run_case(handle, case)
    execution = backend.finalize(handle)
    checks = backend.policy_checks(handle)
    assert backend.cleanup(handle, rollback=False)
    return execution, checks


def test_structured_task_runs_without_pose_or_rgbd_contract(tmp_path: Path) -> None:
    incident, dataset = _fixture(tmp_path)
    provider = DatasetEvidenceProvider(dataset, code_diff="normalization divisor changed")
    context = provider.prepare_case(incident)
    evidence = provider.collect(context)
    reference = DatasetReferenceProvider(incident.task.kind).get_reference(
        incident.case_set[0]
    )

    assert evidence.cases[0].task_kind is TaskKind.STRUCTURED_OUTPUT
    assert evidence.cases[0].rgb is None
    assert evidence.cases[0].depth is None
    assert evidence.cases[0].structured_input is not None
    assert len(evidence.cases[0].input_artifacts) == 1
    prepared = get_task_adapter(incident.task.kind).prepare_execution(incident.case_set[0])
    assert prepared.runner_payload["artifacts"][0]["content_base64"] == "AP8="
    assert "expected_output" not in json.dumps(prepared.runner_payload)

    backend = DatasetExecutionBackend(
        incident,
        GitWorktreeSandbox(Path(incident.repository.path), tmp_path / "worktrees"),
        runner=LocalPythonRunner(),
    )
    baseline_execution, baseline_checks = _execute(
        backend,
        incident,
        CandidateVersion(
            candidate_id="baseline",
            kind=CandidateKind.BASELINE,
            base_commit=incident.baseline_commit,
            rationale="known good",
        ),
    )
    faulty_execution, faulty_checks = _execute(
        backend,
        incident,
        CandidateVersion(
            candidate_id="faulty",
            kind=CandidateKind.FAULTY,
            base_commit=incident.faulty_commit,
            rationale="current version",
        ),
    )
    qa = QAAgent()
    baseline_validation = qa.validate(
        incident,
        evidence,
        baseline_execution,
        (reference,),
        baseline_execution,
        baseline_checks,
    )
    faulty_validation = qa.validate(
        incident,
        evidence,
        faulty_execution,
        (reference,),
        baseline_execution,
        faulty_checks,
    )

    assert baseline_validation.decision is Decision.PASS
    assert faulty_validation.decision is Decision.REJECTED
    assert faulty_validation.failed_cases["normalize-001"] == (
        "$.mean: numeric value outside tolerance",
    )
    assert faulty_execution.case_results[0].robot_outputs is None
    assert faulty_execution.case_results[0].vision_outputs is None


def test_capability_registry_exposes_distinct_real_task_contracts() -> None:
    capabilities = {item["kind"]: item for item in supported_task_capabilities()}

    assert set(capabilities) == {
        "rgbd_pose",
        "detection",
        "ocr",
        "segmentation",
        "structured_output",
    }
    assert capabilities["rgbd_pose"]["gazebo_fixed_motion"] is True
    assert capabilities["detection"]["order_independent"] is True
    assert capabilities["ocr"]["output_contract"] == "{text: string}"
    assert "boundary F1" in capabilities["segmentation"]["deterministic_checks"][-1]
    assert capabilities["structured_output"]["gazebo_fixed_motion"] is False
    assert capabilities["structured_output"]["legacy"] is True
    assert {
        item["automatic_execution"] for item in capabilities.values()
    } == {"python_contract_v1"}
    assert {
        item["runner_protocol"] for item in capabilities.values()
    } == {"JSON stdin/stdout"}


def test_dataset_backend_rejects_non_python_execution_contract(tmp_path: Path) -> None:
    incident, _dataset = _fixture(tmp_path)
    incident = incident.model_copy(
        update={
            "metadata": {
                "execution_contract": {
                    "runner_script": "launch.sh",
                    "test_runner_script": "test_runner.py",
                }
            }
        }
    )
    backend = DatasetExecutionBackend(
        incident,
        GitWorktreeSandbox(Path(incident.repository.path), tmp_path / "worktrees"),
        runner=LocalPythonRunner(),
    )

    with pytest.raises(ValueError, match="supports only Python contract scripts"):
        backend._contract_script("runner_script", "runner.py")


def test_conversation_can_discover_structured_dataset_without_incident_json(
    tmp_path: Path,
) -> None:
    _incident, dataset = _fixture(tmp_path)
    service = DiagnosisSessionService(tmp_path / "sessions")
    session = service.create()

    connected = service.connect_dataset(session["session_id"], str(dataset))

    assert connected["task"]["kind"] == "structured_output"
    assert connected["evidence"]["ready"] is True
    assert connected["evidence"]["count"] == 1
    assert "expected_output" not in json.dumps(connected, ensure_ascii=False)
