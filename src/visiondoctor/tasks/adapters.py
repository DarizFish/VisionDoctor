from __future__ import annotations

import base64
import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from visiondoctor.schemas import (
    ArtifactRef,
    CaseEvidence,
    CaseExecutionResult,
    ExecutionStatus,
    PoseTransform,
    ReferenceSignal,
    RobotOutputs,
    StructuredCaseInput,
    StructuredOutputs,
    StructuredReferenceOutput,
    TaskKind,
    TestCaseRef,
    VisionOutputs,
)
from visiondoctor.tasks.validators import (
    get_validator_plugin,
    public_validator_capabilities,
)
from visiondoctor.vision import DeterministicRgbdPoseEstimator

_MAX_TASK_ARTIFACT_BYTES = 10 * 1024 * 1024
_MAX_TASK_ARTIFACT_TOTAL_BYTES = 20 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_ref(path: Path, artifact_id: str, media_type: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=artifact_id,
        path=str(path.resolve()),
        sha256=sha256_file(path),
        media_type=media_type,
    )


def _load_json(path: Path, case_id: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"task manifest must be an object for {case_id}")
    if data.get("case_id") != case_id:
        raise ValueError(f"manifest case_id mismatch for {case_id}")
    return data


def load_manifest(case: TestCaseRef) -> dict[str, Any]:
    return _load_json(Path(case.manifest_path), case.case_id)


def load_reference_manifest(case: TestCaseRef) -> dict[str, Any]:
    return _load_json(Path(case.reference_path), case.case_id)


def _manifest_kind(data: dict[str, Any]) -> TaskKind:
    return TaskKind(str(data.get("task_kind", TaskKind.RGBD_POSE)))


def _validate_json_value(value: Any, *, label: str) -> None:
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite JSON data") from exc


@dataclass(frozen=True)
class PreparedTaskCase:
    runner_payload: dict[str, Any]
    runtime_context: dict[str, Any]


class DatasetTaskAdapter(ABC):
    kind: TaskKind
    supports_fixed_motion: bool = False

    def _require_kind(self, data: dict[str, Any], case_id: str) -> None:
        actual = _manifest_kind(data)
        if actual is not self.kind:
            raise ValueError(
                f"task kind mismatch for {case_id}: expected {self.kind}, got {actual}"
            )

    @abstractmethod
    def collect_evidence(self, case: TestCaseRef) -> CaseEvidence: ...

    @abstractmethod
    def get_reference(self, case: TestCaseRef) -> ReferenceSignal: ...

    @abstractmethod
    def prepare_execution(self, case: TestCaseRef) -> PreparedTaskCase: ...

    @abstractmethod
    def parse_execution(
        self,
        case: TestCaseRef,
        output: dict[str, Any],
        prepared: PreparedTaskCase,
        fixed_motion: Callable[[str, np.ndarray], RobotOutputs],
    ) -> CaseExecutionResult: ...


class RgbdPoseTaskAdapter(DatasetTaskAdapter):
    kind = TaskKind.RGBD_POSE
    supports_fixed_motion = True

    def collect_evidence(self, case: TestCaseRef) -> CaseEvidence:
        manifest_path = Path(case.manifest_path).resolve()
        manifest = load_manifest(case)
        self._require_kind(manifest, case.case_id)
        if "reference_t_base_object" in manifest:
            raise ValueError(f"QA reference leaked into evidence for {case.case_id}")
        rgb_path = (manifest_path.parent / str(manifest["rgb_path"])).resolve()
        depth_path = (manifest_path.parent / str(manifest["depth_path"])).resolve()
        depth = np.load(depth_path, allow_pickle=False)
        valid = np.isfinite(depth) & (depth > 0)
        source = str(manifest.get("source"))
        if source not in {"dataset", "gazebo"}:
            raise ValueError(f"unsupported evidence source for {case.case_id}: {source}")
        estimator = DeterministicRgbdPoseEstimator()
        return CaseEvidence(
            case_id=case.case_id,
            task_kind=self.kind,
            rgb=artifact_ref(rgb_path, f"{case.case_id}:rgb", "image/png"),
            depth=artifact_ref(
                depth_path, f"{case.case_id}:depth", "application/x-npy"
            ),
            manifest=artifact_ref(
                manifest_path, f"{case.case_id}:manifest", "application/json"
            ),
            t_base_camera=PoseTransform.model_validate(manifest["t_base_camera"]),
            t_camera_object=estimator.estimate(
                rgb_path,
                depth_path,
                np.asarray(manifest["camera_matrix"], dtype=float),
            ),
            camera_matrix=tuple(tuple(row) for row in manifest["camera_matrix"]),
            expected_pixel=tuple(manifest["expected_pixel"]),
            depth_valid_ratio=float(valid.mean()),
            source=source,  # type: ignore[arg-type]
            captured_at=manifest["captured_at"],
        )

    def get_reference(self, case: TestCaseRef) -> ReferenceSignal:
        manifest = load_reference_manifest(case)
        self._require_kind(manifest, case.case_id)
        if "reference_t_base_object" not in manifest:
            raise ValueError(f"reference pose missing for {case.case_id}")
        path = Path(case.reference_path).resolve()
        return ReferenceSignal(
            reference_id=f"REF-{case.case_id}",
            case_id=case.case_id,
            task_kind=self.kind,
            pose=PoseTransform.model_validate(manifest["reference_t_base_object"]),
            source_type=manifest.get("source_type", "dataset_label"),
            provenance=f"sha256:{sha256_file(path)}",
            artifact=artifact_ref(path, f"{case.case_id}:qa-reference", "application/json"),
            captured_at=manifest["captured_at"],
        )

    def prepare_execution(self, case: TestCaseRef) -> PreparedTaskCase:
        manifest_path = Path(case.manifest_path).resolve()
        manifest = load_manifest(case)
        self._require_kind(manifest, case.case_id)
        estimated = DeterministicRgbdPoseEstimator().estimate(
            manifest_path.parent / str(manifest["rgb_path"]),
            manifest_path.parent / str(manifest["depth_path"]),
            np.asarray(manifest["camera_matrix"], dtype=float),
        )
        return PreparedTaskCase(
            runner_payload={
                "case_id": case.case_id,
                "T_base_camera": manifest["t_base_camera"]["matrix"],
                "T_camera_object": estimated.matrix,
            },
            runtime_context={"estimated_pose": estimated},
        )

    def parse_execution(
        self,
        case: TestCaseRef,
        output: dict[str, Any],
        prepared: PreparedTaskCase,
        fixed_motion: Callable[[str, np.ndarray], RobotOutputs],
    ) -> CaseExecutionResult:
        predicted = np.asarray(output["T_base_object"], dtype=float)
        predicted_pose = PoseTransform.from_array("base", "object", predicted)
        estimated = prepared.runtime_context["estimated_pose"]
        if not isinstance(estimated, PoseTransform):
            raise TypeError("RGB-D execution context lost its estimated pose")
        return CaseExecutionResult(
            case_id=case.case_id,
            status=ExecutionStatus.SUCCESS,
            vision_outputs=VisionOutputs(
                t_camera_object=estimated,
                t_base_object=predicted_pose,
                confidence=1.0,
            ),
            robot_outputs=fixed_motion(case.case_id, predicted),
            latency_s=float(output["latency_s"]),
            stdout="candidate runner completed",
        )


class StructuredOutputTaskAdapter(DatasetTaskAdapter):
    """Bounded JSON/artifact transport; task semantics live in validator plugins."""

    kind = TaskKind.STRUCTURED_OUTPUT

    def __init__(self, kind: TaskKind = TaskKind.STRUCTURED_OUTPUT) -> None:
        if kind is TaskKind.RGBD_POSE:
            raise ValueError("RGB-D pose requires its dedicated adapter")
        self.kind = kind
        self.validator = get_validator_plugin(kind)

    def collect_evidence(self, case: TestCaseRef) -> CaseEvidence:
        manifest_path = Path(case.manifest_path).resolve()
        manifest = load_manifest(case)
        self._require_kind(manifest, case.case_id)
        forbidden = {"expected_output", "reference", "assertions", "thresholds"}
        leaked = forbidden.intersection(manifest)
        if leaked:
            raise ValueError(f"QA fields leaked into evidence for {case.case_id}: {sorted(leaked)}")
        if "input" not in manifest:
            raise ValueError(f"structured input missing for {case.case_id}")
        _validate_json_value(manifest["input"], label=f"input for {case.case_id}")
        input_artifacts = self._input_artifacts(case, manifest, include_content=False)
        self.validator.validate_input(
            manifest["input"], tuple(item[0] for item in input_artifacts)
        )
        source = str(manifest.get("source", "dataset"))
        if source not in {"dataset", "gazebo"}:
            raise ValueError(f"unsupported evidence source for {case.case_id}: {source}")
        return CaseEvidence(
            case_id=case.case_id,
            task_kind=self.kind,
            manifest=artifact_ref(
                manifest_path, f"{case.case_id}:manifest", "application/json"
            ),
            structured_input=StructuredCaseInput(value=manifest["input"]),
            input_artifacts=tuple(item[0] for item in input_artifacts),
            source=source,  # type: ignore[arg-type]
            captured_at=manifest["captured_at"],
        )

    def get_reference(self, case: TestCaseRef) -> ReferenceSignal:
        manifest = load_reference_manifest(case)
        self._require_kind(manifest, case.case_id)
        if "expected_output" not in manifest:
            raise ValueError(f"expected output missing for {case.case_id}")
        _validate_json_value(
            manifest["expected_output"], label=f"expected output for {case.case_id}"
        )
        self.validator.validate_reference(manifest["expected_output"])
        path = Path(case.reference_path).resolve()
        return ReferenceSignal(
            reference_id=f"REF-{case.case_id}",
            case_id=case.case_id,
            task_kind=self.kind,
            structured_output=StructuredReferenceOutput(expected=manifest["expected_output"]),
            source_type=manifest.get("source_type", "dataset_label"),
            provenance=f"sha256:{sha256_file(path)}",
            artifact=artifact_ref(path, f"{case.case_id}:qa-reference", "application/json"),
            captured_at=manifest["captured_at"],
        )

    def prepare_execution(self, case: TestCaseRef) -> PreparedTaskCase:
        manifest = load_manifest(case)
        self._require_kind(manifest, case.case_id)
        if "input" not in manifest:
            raise ValueError(f"structured input missing for {case.case_id}")
        artifacts = self._input_artifacts(case, manifest, include_content=True)
        self.validator.validate_input(
            manifest["input"], tuple(item[0] for item in artifacts)
        )
        runner_artifacts = [content for _ref, content in artifacts]
        return PreparedTaskCase(
            runner_payload={
                "case_id": case.case_id,
                "input": manifest["input"],
                "artifacts": runner_artifacts,
            },
            runtime_context={},
        )

    def parse_execution(
        self,
        case: TestCaseRef,
        output: dict[str, Any],
        prepared: PreparedTaskCase,
        fixed_motion: Callable[[str, np.ndarray], RobotOutputs],
    ) -> CaseExecutionResult:
        del prepared, fixed_motion
        if "output" not in output:
            raise ValueError("structured candidate result omitted output")
        _validate_json_value(output["output"], label=f"candidate output for {case.case_id}")
        return CaseExecutionResult(
            case_id=case.case_id,
            status=ExecutionStatus.SUCCESS,
            structured_outputs=StructuredOutputs(value=output["output"]),
            latency_s=float(output["latency_s"]),
            stdout="candidate runner completed",
        )

    @staticmethod
    def _input_artifacts(
        case: TestCaseRef,
        manifest: dict[str, Any],
        *,
        include_content: bool,
    ) -> list[tuple[ArtifactRef, dict[str, Any]]]:
        manifest_root = Path(case.manifest_path).resolve().parent
        values = manifest.get("artifacts", [])
        if not isinstance(values, list):
            raise ValueError(f"artifacts must be an array for {case.case_id}")
        total = 0
        seen: set[str] = set()
        result: list[tuple[ArtifactRef, dict[str, Any]]] = []
        for index, value in enumerate(values):
            if not isinstance(value, dict):
                raise ValueError(f"artifact {index} must be an object for {case.case_id}")
            artifact_id = str(value.get("artifact_id", "")).strip()
            relative = str(value.get("path", "")).strip()
            media_type = str(value.get("media_type", "application/octet-stream")).strip()
            if not artifact_id or artifact_id in seen:
                raise ValueError(f"artifact ids must be non-empty and unique for {case.case_id}")
            path = (manifest_root / relative).resolve()
            if manifest_root != path.parent and manifest_root not in path.parents:
                raise ValueError(f"artifact path escapes case directory for {case.case_id}")
            if not path.is_file():
                raise FileNotFoundError(f"input artifact missing for {case.case_id}: {relative}")
            size = path.stat().st_size
            total += size
            if size > _MAX_TASK_ARTIFACT_BYTES or total > _MAX_TASK_ARTIFACT_TOTAL_BYTES:
                raise ValueError(f"input artifacts exceed bounded size for {case.case_id}")
            seen.add(artifact_id)
            ref = artifact_ref(path, f"{case.case_id}:input:{artifact_id}", media_type)
            content: dict[str, Any] = {
                "artifact_id": artifact_id,
                "media_type": media_type,
                "sha256": ref.sha256,
            }
            if include_content:
                content["content_base64"] = base64.b64encode(path.read_bytes()).decode("ascii")
            result.append((ref, content))
        return result


_ADAPTERS: dict[TaskKind, DatasetTaskAdapter] = {
    TaskKind.RGBD_POSE: RgbdPoseTaskAdapter(),
    TaskKind.STRUCTURED_OUTPUT: StructuredOutputTaskAdapter(),
    TaskKind.DETECTION: StructuredOutputTaskAdapter(TaskKind.DETECTION),
    TaskKind.OCR: StructuredOutputTaskAdapter(TaskKind.OCR),
    TaskKind.SEGMENTATION: StructuredOutputTaskAdapter(TaskKind.SEGMENTATION),
}


def get_task_adapter(kind: TaskKind | str) -> DatasetTaskAdapter:
    normalized = TaskKind(kind)
    try:
        return _ADAPTERS[normalized]
    except KeyError as exc:  # defensive if TaskKind grows without an implementation
        raise ValueError(f"task adapter is not implemented: {normalized}") from exc


def supported_task_capabilities() -> tuple[dict[str, Any], ...]:
    execution = {
        "automatic_execution": "python_contract_v1",
        "runner_runtime": "Python 3.11",
        "runner_protocol": "JSON stdin/stdout",
        "other_project_stacks": "understanding only unless a dedicated adapter is available",
    }
    legacy = {
        "kind": TaskKind.STRUCTURED_OUTPUT.value,
        "legacy": True,
        "inputs": ["finite JSON", "bounded binary artifacts"],
        "outputs": ["finite JSON"],
        "deterministic_checks": [
            "recursive typed output comparison",
            "numeric absolute/relative tolerance",
            "artifact integrity",
            "case pass rate",
            "latency",
        ],
        "gazebo_fixed_motion": False,
        **execution,
    }
    visual = tuple(
        {**capability, "gazebo_fixed_motion": False, **execution}
        for capability in public_validator_capabilities()
    )
    return (
        {
            "kind": TaskKind.RGBD_POSE.value,
            "inputs": ["RGB", "depth", "camera intrinsics", "T_base_camera"],
            "outputs": ["T_base_object"],
            "deterministic_checks": [
                "translation",
                "rotation",
                "reprojection",
                "TCP fixed motion",
                "latency",
            ],
            "gazebo_fixed_motion": True,
            **execution,
        },
        *visual,
        legacy,
    )
