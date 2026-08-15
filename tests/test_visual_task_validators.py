from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from visiondoctor.adapters import DatasetEvidenceProvider, DatasetReferenceProvider
from visiondoctor.agents import DiagnosisAgent, QAAgent
from visiondoctor.llm.tools import EvidenceInspector
from visiondoctor.schemas import (
    AcceptanceCriteria,
    ArtifactRef,
    BackendType,
    CaseEvidence,
    CaseExecutionResult,
    Decision,
    EvidenceBundle,
    ExecutionResult,
    ExecutionStatus,
    Incident,
    PatchPolicy,
    PolicyCheck,
    RepositoryRef,
    StructuredCaseInput,
    StructuredOutputs,
    TaskKind,
    TaskSpecification,
    UnitTestResult,
)
from visiondoctor.schemas import TestCaseRef as CaseRef
from visiondoctor.tasks.validators import get_validator_plugin


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _image_evidence(
    tmp_path: Path, kind: TaskKind, *, width: int = 8, height: int = 8
) -> tuple[CaseEvidence, ArtifactRef]:
    image_path = tmp_path / f"{kind.value}.png"
    Image.new("RGB", (width, height), color=(30, 90, 150)).save(image_path)
    digest = _sha256(image_path)
    image = ArtifactRef(
        artifact_id="case-1:input:image",
        path=str(image_path),
        sha256=digest,
        media_type="image/png",
    )
    manifest_path = tmp_path / f"{kind.value}.json"
    manifest_path.write_text("{}", encoding="utf-8")
    manifest = ArtifactRef(
        artifact_id="case-1:manifest",
        path=str(manifest_path),
        sha256=_sha256(manifest_path),
        media_type="application/json",
    )
    evidence = CaseEvidence(
        case_id="case-1",
        task_kind=kind,
        manifest=manifest,
        structured_input=StructuredCaseInput(
            value={"image_artifact_id": "image", "width": width, "height": height}
        ),
        input_artifacts=(image,),
        source="dataset",
        captured_at=datetime.now(UTC),
    )
    return evidence, image


def test_detection_is_order_independent_and_includes_exact_iou_boundary(
    tmp_path: Path,
) -> None:
    evidence, _image = _image_evidence(tmp_path, TaskKind.DETECTION)
    plugin = get_validator_plugin(TaskKind.DETECTION)
    specification = TaskSpecification(
        kind=TaskKind.DETECTION,
        detection_iou_threshold=1.0,
        detection_precision_threshold=1.0,
        detection_recall_threshold=1.0,
    )
    expected = {
        "objects": [
            {"label": "bolt", "bbox_xyxy": [1, 1, 3, 3]},
            {"label": "nut", "bbox_xyxy": [4, 4, 7, 7]},
        ]
    }
    actual = {
        "detections": [
            {"label": "nut", "bbox_xyxy": [4, 4, 7, 7], "score": 0.9},
            {"label": "bolt", "bbox_xyxy": [1, 1, 3, 3], "score": 0.8},
        ]
    }

    evaluation = plugin.evaluate_case(
        "case-1", actual, expected, evidence, specification
    )

    assert evaluation.contract_valid is True
    assert evaluation.passed is True
    assert evaluation.measurements == {
        "precision": 1.0,
        "recall": 1.0,
        "mean_iou": 1.0,
    }


def test_detection_rejects_out_of_bounds_output_contract(tmp_path: Path) -> None:
    evidence, _image = _image_evidence(tmp_path, TaskKind.DETECTION)
    evaluation = get_validator_plugin(TaskKind.DETECTION).evaluate_case(
        "case-1",
        {
            "detections": [
                {"label": "bolt", "bbox_xyxy": [1, 1, 9, 3], "score": 0.9}
            ]
        },
        {"objects": [{"label": "bolt", "bbox_xyxy": [1, 1, 3, 3]}]},
        evidence,
        TaskSpecification(kind=TaskKind.DETECTION),
    )

    assert evaluation.contract_valid is False
    assert "outside image dimensions" in evaluation.failures[0]


def test_ocr_normalization_and_exact_error_rate_boundary(tmp_path: Path) -> None:
    evidence, _image = _image_evidence(tmp_path, TaskKind.OCR)
    plugin = get_validator_plugin(TaskKind.OCR)
    specification = TaskSpecification(
        kind=TaskKind.OCR,
        ocr_case_sensitive=False,
        ocr_normalize_whitespace=True,
        ocr_max_character_error_rate=1 / 3,
        ocr_max_word_error_rate=1.0,
    )

    normalized = plugin.evaluate_case(
        "case-1",
        {"text": "Ａ  B"},
        {"text": "a b"},
        evidence,
        specification,
    )
    boundary = plugin.evaluate_case(
        "case-1",
        {"text": "axc"},
        {"text": "abc"},
        evidence,
        specification,
    )
    invalid = plugin.evaluate_case(
        "case-1", {"text": 42}, {"text": "42"}, evidence, specification
    )

    assert normalized.passed is True
    assert boundary.measurements["character_error_rate"] == pytest.approx(1 / 3)
    assert boundary.passed is True
    assert invalid.contract_valid is False


def test_segmentation_boundary_tolerance_and_shape_contract(tmp_path: Path) -> None:
    evidence, _image = _image_evidence(
        tmp_path, TaskKind.SEGMENTATION, width=6, height=6
    )
    plugin = get_validator_plugin(TaskKind.SEGMENTATION)
    reference = np.zeros((6, 6), dtype=int)
    reference[2:4, 2:4] = 1
    shifted = np.zeros((6, 6), dtype=int)
    shifted[2:4, 3:5] = 1
    common = {
        "kind": TaskKind.SEGMENTATION,
        "segmentation_mean_iou_threshold": 0.0,
        "segmentation_pixel_accuracy_threshold": 0.0,
        "segmentation_boundary_f1_threshold": 1.0,
    }

    tolerant = plugin.evaluate_case(
        "case-1",
        {"mask": shifted.tolist()},
        {"mask": reference.tolist()},
        evidence,
        TaskSpecification(**common, segmentation_boundary_tolerance_px=1),
    )
    strict = plugin.evaluate_case(
        "case-1",
        {"mask": shifted.tolist()},
        {"mask": reference.tolist()},
        evidence,
        TaskSpecification(**common, segmentation_boundary_tolerance_px=0),
    )
    invalid = plugin.evaluate_case(
        "case-1",
        {"mask": [[0, 0], [0, 0]]},
        {"mask": reference.tolist()},
        evidence,
        TaskSpecification(**common, segmentation_boundary_tolerance_px=1),
    )

    assert tolerant.measurements["boundary_f1"] == pytest.approx(1.0)
    assert tolerant.passed is True
    assert strict.measurements["boundary_f1"] < 1.0
    assert strict.passed is False
    assert invalid.contract_valid is False


def _write_detection_case(root: Path, case_id: str) -> CaseRef:
    image_path = root / f"{case_id}.png"
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(image_path)
    captured_at = "2026-08-10T00:00:00Z"
    manifest = root / f"{case_id}.json"
    reference = root / f"{case_id}.reference.json"
    manifest.write_text(
        json.dumps(
            {
                "task_kind": "detection",
                "case_id": case_id,
                "source": "dataset",
                "captured_at": captured_at,
                "input": {
                    "image_artifact_id": "image",
                    "width": 8,
                    "height": 8,
                },
                "artifacts": [
                    {
                        "artifact_id": "image",
                        "path": image_path.name,
                        "media_type": "image/png",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    reference.write_text(
        json.dumps(
            {
                "task_kind": "detection",
                "case_id": case_id,
                "source_type": "dataset_label",
                "captured_at": captured_at,
                "expected_output": {
                    "objects": [{"label": "part", "bbox_xyxy": [1, 1, 4, 4]}]
                },
            }
        ),
        encoding="utf-8",
    )
    return CaseRef(
        case_id=case_id,
        manifest_path=str(manifest),
        reference_path=str(reference),
    )


def _execution(
    candidate_id: str,
    outputs: list[tuple[str, dict[str, Any]]],
) -> ExecutionResult:
    now = datetime.now(UTC)
    return ExecutionResult(
        execution_id=f"EXEC-{candidate_id}",
        candidate_id=candidate_id,
        backend_type=BackendType.DATASET,
        case_results=tuple(
            CaseExecutionResult(
                case_id=case_id,
                status=ExecutionStatus.SUCCESS,
                structured_outputs=StructuredOutputs(value=value),
                latency_s=0.01,
            )
            for case_id, value in outputs
        ),
        unit_tests=UnitTestResult(
            passed=True,
            tests_run=1,
            failures=0,
            errors=0,
            duration_s=0.01,
        ),
        status=ExecutionStatus.SUCCESS,
        started_at=now,
        completed_at=now,
    )


def test_multi_case_partial_pass_and_shuffled_execution_are_deterministic(
    tmp_path: Path,
) -> None:
    cases = (_write_detection_case(tmp_path, "case-a"), _write_detection_case(tmp_path, "case-b"))
    incident = Incident(
        incident_id="INC-DETECTION",
        title="missed part",
        description="One image loses its part detection.",
        repository=RepositoryRef(path=str(tmp_path)),
        baseline_commit="1234567",
        faulty_commit="7654321",
        case_set=cases,
        task=TaskSpecification(
            kind=TaskKind.DETECTION,
            case_pass_rate=0.5,
            detection_iou_threshold=1.0,
            detection_precision_threshold=1.0,
            detection_recall_threshold=1.0,
        ),
        acceptance_criteria=AcceptanceCriteria(latency_growth_ratio=0.1),
        allowed_patch_scope=PatchPolicy(),
    )
    evidence = DatasetEvidenceProvider(tmp_path).collect(
        DatasetEvidenceProvider(tmp_path).prepare_case(incident)
    )
    references = tuple(
        DatasetReferenceProvider(TaskKind.DETECTION).get_reference(case) for case in cases
    )
    passed_output = {
        "detections": [
            {"label": "part", "bbox_xyxy": [1, 1, 4, 4], "score": 0.99}
        ]
    }
    baseline = _execution(
        "baseline", [("case-b", passed_output), ("case-a", passed_output)]
    )
    candidate = _execution(
        "candidate", [("case-b", {"detections": []}), ("case-a", passed_output)]
    )

    qa = QAAgent()
    baseline_report = qa.validate(
        incident,
        evidence,
        baseline,
        tuple(reversed(references)),
        baseline,
        (PolicyCheck(name="scope", passed=True, details="allowed"),),
    )
    report = qa.validate(
        incident,
        evidence,
        candidate,
        tuple(reversed(references)),
        baseline,
        (PolicyCheck(name="scope", passed=True, details="allowed"),),
    )

    metrics = {item.name: item for item in report.metric_results}
    assert report.decision is Decision.REJECTED
    assert report.passed_cases == ("case-a",)
    assert "case-b" in report.failed_cases
    assert metrics["task_output_contract"].passed is True
    assert metrics["case_pass_rate"].value == pytest.approx(0.5)
    assert metrics["case_pass_rate"].passed is True
    assert metrics["detection_recall"].value == pytest.approx(0.5)
    payload = DiagnosisAgent._task_payload(
        incident,
        evidence,
        candidate,
        baseline_report,
        report,
    )
    assert payload["faulty_execution"]["failed_execution_cases"] == []
    assert payload["faulty_validation"]["failed_cases"] == [
        {
            "case_id": "case-b",
            "reasons": list(report.failed_cases["case-b"]),
        }
    ]
    artifact_cases = {
        item["case_id"]: item["artifacts"]
        for item in payload["evidence_summary"]["available_input_artifacts"]
    }
    assert artifact_cases["case-b"][0]["artifact_id"] == "case-b:input:image"


@pytest.mark.parametrize(
    "kind",
    [
        TaskKind.RGBD_POSE,
        TaskKind.DETECTION,
        TaskKind.OCR,
        TaskKind.SEGMENTATION,
        TaskKind.STRUCTURED_OUTPUT,
    ],
)
def test_default_dataset_log_matches_task_kind(kind: TaskKind) -> None:
    summary = DatasetEvidenceProvider._default_log(kind)

    if kind is TaskKind.RGBD_POSE:
        assert "TCP target" in summary
    else:
        assert "TCP" not in summary
        assert (
            kind.value.split("_")[0].casefold() in summary.casefold()
            or kind is TaskKind.STRUCTURED_OUTPUT
        )


class _VisionGatewayDouble:
    model = "vision-test"

    def __init__(self) -> None:
        self.paths: list[Path] = []

    def assess(
        self,
        image_path: Path,
        *,
        attachment_id: str,
        visible_name: str,
        user_context: str,
    ) -> dict[str, Any]:
        del attachment_id, visible_name, user_context
        self.paths.append(image_path)
        return {
            "observations": ["visible part"],
            "diagnostic_relevance": "possible missed detection",
            "limitations": [],
            "confidence": 0.8,
            "model": self.model,
        }


def test_evidence_tools_observe_pixels_and_never_accept_qa_artifacts(
    tmp_path: Path,
) -> None:
    case, image = _image_evidence(tmp_path, TaskKind.DETECTION)
    cloud_path = tmp_path / "cloud.npy"
    np.save(cloud_path, np.asarray([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]]))
    cloud = ArtifactRef(
        artifact_id="case-1:input:cloud",
        path=str(cloud_path),
        sha256=_sha256(cloud_path),
        media_type="application/x-npy",
    )
    case = case.model_copy(update={"input_artifacts": (image, cloud)})
    evidence = EvidenceBundle(
        evidence_id="EVID-1",
        incident_id="INC-1",
        issue_text="missed part",
        raw_log="",
        code_diff="diff",
        cases=(case,),
        artifact_hashes={
            case.manifest.artifact_id: case.manifest.sha256,
            image.artifact_id: image.sha256,
            cloud.artifact_id: cloud.sha256,
        },
    )
    gateway = _VisionGatewayDouble()
    inspector = EvidenceInspector(
        evidence, vision_gateway=gateway, user_context="missed part"
    )

    observed = inspector.execute(
        "observe_evidence_image",
        {"case_id": "case-1", "artifact_id": image.artifact_id},
    )
    metadata = inspector.execute(
        "inspect_evidence_metadata",
        {"case_id": "case-1", "artifact_id": image.artifact_id},
    )
    summary = inspector.execute(
        "summarize_point_cloud",
        {"case_id": "case-1", "artifact_id": cloud.artifact_id},
    )

    assert gateway.paths == [Path(image.path)]
    assert observed["artifact_id"] == image.artifact_id
    assert inspector.observed_cases == {"case-1"}
    assert metadata["image"]["width"] == 8
    assert "path" not in json.dumps(metadata)
    assert summary["point_count"] == 2
    assert "points" not in summary
    with pytest.raises(ValueError, match="not an allowed input"):
        inspector.execute(
            "inspect_evidence_metadata",
            {"case_id": "case-1", "artifact_id": "case-1:qa-reference"},
        )


def test_evidence_tools_fail_closed_after_artifact_tampering(tmp_path: Path) -> None:
    case, image = _image_evidence(tmp_path, TaskKind.DETECTION)
    evidence = EvidenceBundle(
        evidence_id="EVID-1",
        incident_id="INC-1",
        issue_text="missed part",
        raw_log="",
        code_diff="diff",
        cases=(case,),
        artifact_hashes={
            case.manifest.artifact_id: case.manifest.sha256,
            image.artifact_id: image.sha256,
        },
    )
    Path(image.path).write_bytes(b"tampered")
    inspector = EvidenceInspector(
        evidence, vision_gateway=_VisionGatewayDouble(), user_context="missed part"
    )

    with pytest.raises(ValueError, match="hash verification failed"):
        inspector.execute(
            "observe_evidence_image",
            {"case_id": "case-1", "artifact_id": image.artifact_id},
        )
