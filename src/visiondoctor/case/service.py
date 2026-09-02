"""One place that owns cases, so an API and a console see the same thing."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from visiondoctor.environment import FileBundleAdapter, ObservationBundle
from visiondoctor.repair import ProjectBinding, Recheck, recheck

from .case import Case, Evidence
from .chain import SEGMENT_SCOPE
from .gates import approval_gate, diagnosis_gate
from .repair import ApprovalRecord, RepairPlan


@dataclass
class CaseRecord:
    case: Case
    adapter: FileBundleAdapter | None = None
    bundle: ObservationBundle | None = None
    turns: list[Any] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    uploads: dict[str, Path] = field(default_factory=dict)
    approvals: dict[str, ApprovalRecord] = field(default_factory=dict)
    applied_at: datetime | None = None
    recheck: Recheck | None = None


class CaseService:
    """Create cases, feed them observations, push them one turn at a time."""

    def __init__(self) -> None:
        self.records: dict[str, CaseRecord] = {}

    # ---- lifecycle ---------------------------------------------------------------

    def create(self, title: str, *, guided_motion: bool = True) -> str:
        case_id = f"CASE-{len(self.records) + 1:03d}"
        self.records[case_id] = CaseRecord(Case(case_id, title, guided_motion=guided_motion))
        return case_id

    def listing(self) -> list[dict[str, Any]]:
        return [
            {
                "case_id": key,
                "title": record.case.title,
                "project": (
                    str(record.case.project.repository) if record.case.project else None
                ),
                "evidence": len(record.case.evidence),
                "observations": len(record.case.observations),
                "gate": diagnosis_gate(record.case).passed,
            }
            for key, record in self.records.items()
        ]

    def record(self, case_id: str) -> CaseRecord:
        if case_id not in self.records:
            raise KeyError(f"没有这个案件：{case_id}")
        return self.records[case_id]

    # ---- inputs ------------------------------------------------------------------

    def attach_observation(self, case_id: str, directory: Path) -> dict[str, Any]:
        record = self.record(case_id)
        adapter = FileBundleAdapter(Path(directory))
        bundle = adapter.collect()
        admitted = record.case.admit(bundle)
        record.adapter = adapter
        record.bundle = bundle
        record.messages.append(
            {
                "role": "source",
                "content": f"接入观察证据包 {bundle.run_id}",
                "detail": f"{len(bundle.artifacts)} 件工件 · 收入 {len(admitted)} 条证据",
            }
        )
        return {"run_id": bundle.run_id, "admitted": len(admitted)}

    def attach_files(
        self, case_id: str, files: list[dict[str, str]], root: Path | None = None
    ) -> dict[str, Any]:
        """Take material a person handed in directly and admit it as evidence."""

        record = self.record(case_id)
        folder = (root or Path(".runtime/vd-uploads")) / case_id
        folder.mkdir(parents=True, exist_ok=True)
        admitted = []
        for item in files:
            name = Path(str(item["name"])).name
            payload = base64.b64decode(item["content_base64"])
            path = folder / f"{len(record.uploads) + 1:03d}-{name}"
            path.write_bytes(payload)
            evidence = record.case.add_evidence(
                Evidence(
                    evidence_id=record.case.next_evidence_id(),
                    bundle_id="handed-in",
                    reference=f"uploads/{name}",
                    media_type=str(item.get("media_type") or "application/octet-stream"),
                    captured_at=datetime.now(UTC),
                    clock_domain="handed_in",
                    summary=f"{name}，由人工提供",
                )
            )
            # Handed-in material counts as delivered only once it is read.
            record.case.examined.discard(evidence.evidence_id)
            record.uploads[evidence.evidence_id] = path
            admitted.append({"evidence_id": evidence.evidence_id, "name": name})
        record.messages.append(
            {
                "role": "source",
                "content": "补充材料 " + "、".join(item["name"] for item in admitted),
                "detail": f"{len(admitted)} 份，已登记为证据待查阅",
            }
        )
        return {"admitted": admitted}

    def bind_project(
        self,
        case_id: str,
        repository: Path,
        revision: str,
        replay_command: tuple[str, ...],
        test_command: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        record = self.record(case_id)
        binding = record.case.bind_project(
            ProjectBinding(
                repository=Path(repository).resolve(),
                revision=revision,
                replay_command=tuple(replay_command),
                test_command=tuple(test_command) if test_command else None,
            )
        )
        return {"repository": str(binding.repository), "revision": binding.revision}

    # ---- pushing it forward ------------------------------------------------------

    def run_turn(self, case_id: str, prompt: str) -> dict[str, Any]:
        from visiondoctor.investigation import investigate
        from visiondoctor.llm import ModelSettings, OpenAICompatibleGateway

        record = self.record(case_id)
        if record.bundle is None and not record.uploads:
            raise ValueError("这个案件还没有任何材料：先附上证据包或上传图片、日志")
        turn = investigate(
            case=record.case,
            adapter=record.adapter,
            bundle=record.bundle,
            prompt=prompt,
            gateway=OpenAICompatibleGateway(ModelSettings.from_environment()),
            vision=_vision_gateway(),
            uploads=record.uploads,
        )
        record.turns.append(turn)
        if record.case.title == "新的诊断":
            record.case.title = prompt[:24]
        record.messages.append({"role": "user", "content": prompt})
        record.messages.append(
            {
                "role": "assistant",
                "content": turn.next_step,
                "turn_id": turn.turn_id,
                "findings": [
                    f"{item.segment.value} → {item.status.value}"
                    for item in turn.findings
                    if item.status.value != "untested"
                ],
                "hypotheses": [item.statement for item in turn.hypotheses],
                "calls": [
                    {"name": call.name, "delivered": list(call.delivered),
                     "arguments": {
                         key: (value if len(str(value)) < 200 else str(value)[:200] + "…")
                         for key, value in call.arguments.items()
                     }}
                    for call in turn.calls
                ],
            }
        )
        return turn.model_dump(mode="json")

    def approve(
        self, case_id: str, plan_id: str, approver: str, *, approved: bool
    ) -> dict[str, Any]:
        record = self.record(case_id)
        plan = self._plan(record, plan_id)
        decision = ApprovalRecord(
            plan_id=plan.plan_id,
            frozen_hash=plan.frozen_hash,
            approved=approved,
            approver=approver,
            decided_at=datetime.now(UTC),
        )
        record.approvals[plan_id] = decision
        record.messages.append(
            {
                "role": "source",
                "content": ("已批准 " if approved else "已退回 ") + plan.plan_id,
                "detail": f"{approver} · 冻结 {plan.frozen_hash[:16]}",
            }
        )
        return approval_gate(plan, decision).model_dump(mode="json")

    def mark_applied(self, case_id: str) -> datetime:
        record = self.record(case_id)
        record.applied_at = datetime.now(UTC)
        return record.applied_at

    def run_recheck(self, case_id: str, directory: Path) -> dict[str, Any]:
        record = self.record(case_id)
        if record.bundle is None:
            raise ValueError("没有可比对的原始观察")
        after = FileBundleAdapter(Path(directory)).collect()
        record.case.admit(after)
        outcome = recheck(before=record.bundle, after=after, applied_at=record.applied_at)
        record.recheck = outcome
        record.messages.append(
            {
                "role": "source",
                "content": f"现场复核 {after.run_id}",
                "detail": outcome.scope,
            }
        )
        return outcome.as_dict()

    # ---- what a viewer sees ------------------------------------------------------

    def view(self, case_id: str) -> dict[str, Any]:
        record = self.record(case_id)
        case = record.case
        verdict = diagnosis_gate(case)
        return {
            "case_id": case.case_id,
            "title": case.title,
            "observation": self._observation(record),
            "project": self._project(case),
            "chain": self._chain(case),
            "messages": record.messages,
            "hypotheses": [item.model_dump(mode="json") for item in case.hypotheses],
            "evidence": [
                {
                    "evidence_id": item.evidence_id,
                    "reference": item.reference,
                    "media_type": item.media_type,
                    "captured_at": item.captured_at.isoformat(),
                    "examined": item.evidence_id in case.examined,
                }
                for item in case.evidence
            ],
            "gates": {
                "diagnosis": verdict.model_dump(mode="json"),
                "source_visible": bool(case.project and verdict.passed),
            },
            "plans": self._plans(record),
            "turns": [turn.model_dump(mode="json") for turn in record.turns],
            "applied_at": record.applied_at.isoformat() if record.applied_at else None,
            "recheck": record.recheck.as_dict() if record.recheck else None,
        }

    @staticmethod
    def _observation(record: CaseRecord) -> dict[str, Any] | None:
        if record.bundle is None:
            return None
        return {
            "run_id": record.bundle.run_id,
            "collected_at": record.bundle.created_at.isoformat(),
            "revision": record.bundle.project_revision,
            "results": [item.model_dump(mode="json") for item in record.bundle.results],
            "artifacts": len(record.bundle.artifacts),
        }

    @staticmethod
    def _project(case: Case) -> dict[str, Any] | None:
        if case.project is None:
            return None
        return {
            "repository": str(case.project.repository),
            "revision": case.project.revision,
        }

    @staticmethod
    def _chain(case: Case) -> list[dict[str, Any]]:
        findings = {item.segment: item for item in case.findings}
        rows = []
        for segment, status in case.demarcation().items():
            finding = findings.get(segment)
            rows.append(
                {
                    "segment": segment.value,
                    "status": status.value,
                    "scope": SEGMENT_SCOPE[segment],
                    "note": finding.note if finding else "",
                    "evidence_ids": list(finding.evidence_ids) if finding else [],
                }
            )
        return rows

    def _plans(self, record: CaseRecord) -> list[dict[str, Any]]:
        rows = []
        for plan in record.case.repair_plans:
            decision = record.approvals.get(plan.plan_id)
            rows.append(
                {
                    "plan_id": plan.plan_id,
                    "target_segment": plan.target_segment.value,
                    "frozen_hash": plan.frozen_hash,
                    "diff": plan.diff,
                    "approved": decision.approved if decision else None,
                    "approver": decision.approver if decision else None,
                    "gate": approval_gate(plan, decision).model_dump(mode="json"),
                }
            )
        return rows

    @staticmethod
    def _plan(record: CaseRecord, plan_id: str) -> RepairPlan:
        for plan in record.case.repair_plans:
            if plan.plan_id == plan_id:
                return plan
        raise KeyError(f"没有这个候选：{plan_id}")


def _vision_gateway():
    """The image observer, when it is configured; images stay unread otherwise."""

    from visiondoctor.multimodal import (
        OpenAIVisionGateway,
        VisionConfigurationError,
        VisionSettings,
    )

    try:
        return OpenAIVisionGateway(VisionSettings.from_environment())
    except VisionConfigurationError:
        return None
