from __future__ import annotations

from pathlib import Path

from visiondoctor.adapters.base import CaseContext
from visiondoctor.schemas import (
    EvidenceBundle,
    Incident,
    ReferenceSignal,
    TaskKind,
    TestCaseRef,
)
from visiondoctor.tasks import get_task_adapter
from visiondoctor.tasks.adapters import load_reference_manifest


class DatasetEvidenceProvider:
    def __init__(
        self, dataset_root: Path, *, code_diff: str = "", raw_log: str = ""
    ) -> None:
        self.dataset_root = dataset_root.resolve()
        self.code_diff = code_diff
        self.raw_log = raw_log

    def prepare_case(self, incident: Incident) -> CaseContext:
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(f"dataset root does not exist: {self.dataset_root}")
        for case in incident.case_set:
            if not Path(case.manifest_path).is_file():
                raise FileNotFoundError(f"missing manifest for case {case.case_id}")
            if not Path(case.reference_path).is_file():
                raise FileNotFoundError(f"missing QA reference for case {case.case_id}")
        return CaseContext(incident=incident, dataset_root=self.dataset_root)

    def collect(self, context: CaseContext) -> EvidenceBundle:
        adapter = get_task_adapter(context.incident.task.kind)
        cases = []
        hashes: dict[str, str] = {}
        for case_ref in context.incident.case_set:
            manifest_path = Path(case_ref.manifest_path).resolve()
            if manifest_path == Path(case_ref.reference_path).resolve():
                raise ValueError(
                    f"evidence and QA reference must be separate files for {case_ref.case_id}"
                )
            collected = adapter.collect_evidence(case_ref)
            for ref in (
                collected.manifest,
                collected.rgb,
                collected.depth,
                *collected.input_artifacts,
            ):
                if ref is not None:
                    hashes[ref.artifact_id] = ref.sha256
            cases.append(collected)
        return EvidenceBundle(
            evidence_id=f"EVID-{context.incident.incident_id}",
            incident_id=context.incident.incident_id,
            issue_text=context.incident.description,
            raw_log=self.raw_log or self._default_log(context.incident.task.kind),
            code_diff=self.code_diff,
            cases=tuple(cases),
            artifact_hashes=hashes,
        )

    @staticmethod
    def _default_log(kind: TaskKind) -> str:
        return {
            TaskKind.RGBD_POSE: (
                "faulty version completes fixed motion but reaches the wrong TCP target"
            ),
            TaskKind.DETECTION: (
                "faulty version completes inference but one or more detection cases fail "
                "deterministic object matching"
            ),
            TaskKind.OCR: (
                "faulty version completes inference but one or more OCR cases fail text "
                "accuracy checks"
            ),
            TaskKind.SEGMENTATION: (
                "faulty version completes inference but one or more segmentation cases fail "
                "mask quality checks"
            ),
            TaskKind.STRUCTURED_OUTPUT: (
                "faulty version completes execution but one or more structured output cases "
                "fail deterministic comparison"
            ),
        }[kind]


class DatasetReferenceProvider:
    def __init__(self, task_kind: TaskKind | str | None = None) -> None:
        self.task_kind = TaskKind(task_kind) if task_kind is not None else None

    def get_reference(self, case: TestCaseRef) -> ReferenceSignal:
        manifest = load_reference_manifest(case)
        kind = self.task_kind or TaskKind(
            str(manifest.get("task_kind", TaskKind.RGBD_POSE))
        )
        return get_task_adapter(kind).get_reference(case)
