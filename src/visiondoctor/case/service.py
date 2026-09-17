"""One place that owns cases, so an API and a console see the same thing."""

from __future__ import annotations

import base64
import threading
from dataclasses import dataclass, field
from dataclasses import replace as _replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from visiondoctor.environment import FileBundleAdapter, ObservationBundle
from visiondoctor.repair import ProjectBinding, Recheck, land, recheck

from . import store
from .case import Case, Evidence
from .chain import SEGMENT_NAME, SEGMENT_SCOPE
from .gates import approval_gate, software_localizations, source_layer_gate
from .graph import consuming_node
from .isolation import isolation_view, node_standing
from .repair import ApprovalRecord, RepairPlan


@dataclass
class CaseRecord:
    case: Case
    adapter: FileBundleAdapter | None = None
    bundle: ObservationBundle | None = None
    turns: list[Any] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    uploads: dict[str, Path] = field(default_factory=dict)
    #: Export directories, kept instead of their contents so a restart can
    #: collect the bundles again rather than copy them.
    observation_dirs: list[str] = field(default_factory=list)
    #: The turn in flight, so a viewer can watch it work.
    running: dict[str, Any] | None = None
    approvals: dict[str, ApprovalRecord] = field(default_factory=dict)
    #: Where an approved plan was written in the repository, by plan id.
    landings: dict[str, dict[str, Any]] = field(default_factory=dict)
    applied_at: datetime | None = None
    recheck: Recheck | None = None


class CaseService:
    """Create cases, feed them observations, push them one turn at a time."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else store.ROOT
        self.records: dict[str, CaseRecord] = store.load_all(CaseRecord, self.root)

    def keep(self, case_id: str) -> None:
        """Write the case down.  Called after anything that changed it."""

        store.save(self.records[case_id], self.root)

    # ---- lifecycle ---------------------------------------------------------------

    def create(self, title: str, *, guided_motion: bool = True) -> str:
        taken = [int(key.rsplit("-", maxsplit=1)[-1]) for key in self.records]
        case_id = f"CASE-{max(taken, default=0) + 1:03d}"
        self.records[case_id] = CaseRecord(Case(case_id, title, guided_motion=guided_motion))
        self.keep(case_id)
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
                "gate": source_layer_gate(record.case).passed,
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
        # The first observation is the one under diagnosis; later ones, such as
        # a normal reference run, are compared against it.
        if record.bundle is None:
            record.adapter, record.bundle = adapter, bundle
        record.observation_dirs.append(str(Path(directory)))
        record.messages.append(
            {
                "role": "source",
                "content": f"接入观察证据包 {bundle.run_id}",
                "detail": f"{len(bundle.artifacts)} 件工件 · 收入 {len(admitted)} 条证据",
            }
        )
        self._realign_project(record)
        self.keep(case_id)
        return {"run_id": bundle.run_id, "admitted": len(admitted)}

    def _realign_project(self, record: CaseRecord) -> None:
        """A project bound before the observation arrived is bound to the wrong thing.

        Connecting first pins the repository at its tip, which is whatever the
        working copy happens to be today.  Once the bundle says which commit was
        running, that answer wins.  A repository that does not hold that commit
        did not produce this run, so the binding is dropped rather than left
        quietly pointing somewhere else.
        """

        case = record.case
        if case.project is None or record.bundle is None:
            return
        wanted = _observed_revision(record)
        if wanted == "HEAD" or wanted == case.project.revision:
            return
        try:
            resolved = _resolve_revision(case.project.repository, wanted)
        except ValueError as exc:
            repository = case.project.repository
            case.project = None
            record.messages.append(
                {
                    "role": "source",
                    "content": "已断开项目仓库连接",
                    "detail": f"{repository} 里没有这次运行的提交（{exc}）。请重新连接正确的仓库。",
                }
            )
            return
        case.project = _replace(case.project, revision=resolved)
        record.messages.append(
            {
                "role": "source",
                "content": "项目提交已按观察包对齐",
                "detail": f"源码改按 {resolved[:12]} 读取",
            }
        )

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
            admitted.append(
                {
                    "evidence_id": evidence.evidence_id,
                    "name": name,
                    "media_type": evidence.media_type,
                }
            )
        record.messages.append(
            {
                "role": "source",
                "content": "补充材料",
                "detail": f"{len(admitted)} 份，已登记为证据待查阅",
                "files": admitted,
            }
        )
        self.keep(case_id)
        return {"admitted": admitted}

    def evidence_content(self, case_id: str, evidence_id: str) -> dict[str, Any]:
        """Hand back a handed-in file so the conversation can show it again."""

        record = self.record(case_id)
        path = record.uploads.get(evidence_id)
        if path is None or not path.is_file():
            raise KeyError(f"没有可显示的材料：{evidence_id}")
        return {
            "evidence_id": evidence_id,
            "content_base64": base64.b64encode(path.read_bytes()).decode("ascii"),
        }

    def bind_project(
        self,
        case_id: str,
        repository: Path,
        revision: str = "",
        replay_command: tuple[str, ...] = (),
        test_command: tuple[str, ...] | None = None,
        source_readable: bool = True,
    ) -> dict[str, Any]:
        """Bind the repository at the revision that produced the observation.

        ``source_readable=False`` stands for a site that holds only the released
        program: it can be re-run on recorded inputs, its source stays unseen.

        The bundle already says which commit was running when the evidence was
        taken, so nobody is asked to type it.  Reading whatever the working tree
        holds today would let the model reason about code that never produced
        this run -- and say nothing about it having done so.
        """

        record = self.record(case_id)
        repository = Path(repository).resolve()
        wanted = revision.strip() or _observed_revision(record)
        resolved = _resolve_revision(repository, wanted)
        binding = record.case.bind_project(
            ProjectBinding(
                repository=repository,
                revision=resolved,
                replay_command=tuple(replay_command),
                test_command=tuple(test_command) if test_command else None,
                source_readable=source_readable,
            )
        )
        self.keep(case_id)
        return {"repository": str(binding.repository), "revision": binding.revision,
                "runnable": binding.runnable, "source_readable": binding.source_readable}

    # ---- pushing it forward ------------------------------------------------------

    def run_turn(self, case_id: str, prompt: str) -> dict[str, Any]:
        """Start a turn and return at once; the work is watched, not waited on."""

        record = self.record(case_id)
        if record.bundle is None and not record.uploads:
            raise ValueError("这个案件还没有任何材料：先附上证据包或上传图片、日志")
        if record.running is not None:
            raise ValueError("这个案件正在推进中，等这一轮结束再说下一句")
        if record.case.title == "新的诊断":
            record.case.title = prompt[:24]
        record.messages.append({"role": "user", "content": prompt})
        record.running = {"prompt": prompt, "calls": [], "error": None}
        self.keep(case_id)
        thread = threading.Thread(
            target=self._work, args=(case_id, prompt), daemon=True
        )
        thread.start()
        return {"started": True, "prompt": prompt}

    def _work(self, case_id: str, prompt: str) -> None:
        from visiondoctor.investigation import investigate

        record = self.record(case_id)
        try:
            turn = investigate(
                case=record.case,
                adapter=record.adapter,
                bundle=record.bundle,
                prompt=prompt,
                gateway=_model_gateway(),
                vision=_vision_gateway(),
                uploads=record.uploads,
                observation_dirs=tuple(Path(path) for path in record.observation_dirs),
                observer=lambda call: record.running["calls"].append(
                    _call_summary(call)
                ),
            )
        except Exception as exc:  # noqa: BLE001 - the viewer needs the reason
            record.running = None
            record.messages.append(
                {"role": "assistant", "content": f"这一轮没有走完：{exc}", "calls": []}
            )
            self.keep(case_id)
            return
        record.turns.append(turn)
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
                "calls": [_call_summary(call) for call in turn.calls],
            }
        )
        record.running = None
        self.keep(case_id)

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
        verdict = approval_gate(plan, decision)
        if verdict.passed:
            self._land(record, plan, approver)
        self.keep(case_id)
        return verdict.model_dump(mode="json")

    def _land(self, record: CaseRecord, plan: RepairPlan, approver: str) -> None:
        """Write the approved bytes into the repository they were raised against.

        Copying a reviewed diff by hand is where an unreviewed character gets in,
        so the host writes it -- and writes it where someone opening the file
        will see it.  Putting it on the cell is still a human act.
        """

        binding = record.case.project
        if binding is None:
            return
        try:
            landed = land(
                binding=binding,
                plan_id=plan.plan_id,
                frozen_hash=plan.frozen_hash,
                diff=plan.diff,
                case_id=record.case.case_id,
                approver=approver,
            )
        except Exception as exc:  # noqa: BLE001 - the person needs the reason
            record.landings[plan.plan_id] = {"error": f"{type(exc).__name__}: {exc}"}
            record.messages.append(
                {
                    "role": "source",
                    "content": f"{plan.plan_id} 没能写入仓库",
                    "detail": str(exc),
                }
            )
            return
        record.landings[plan.plan_id] = landed
        record.messages.append(
            {
                "role": "source",
                "content": f"{plan.plan_id} 已提交到 {landed['branch']}",
                "detail": f"提交 {landed['commit'][:12]}　{'、'.join(landed['files'])}",
            }
        )

    def mark_applied(self, case_id: str) -> datetime:
        record = self.record(case_id)
        record.applied_at = datetime.now(UTC)
        self.keep(case_id)
        return record.applied_at

    def run_recheck(self, case_id: str, directory: Path) -> dict[str, Any]:
        record = self.record(case_id)
        if record.bundle is None:
            raise ValueError("没有可比对的原始观察")
        after = FileBundleAdapter(Path(directory)).collect()
        record.case.admit(after)
        record.observation_dirs.append(str(Path(directory)))
        outcome = recheck(before=record.bundle, after=after, applied_at=record.applied_at)
        record.recheck = outcome
        self._recheck_structure(record, Path(directory), after)
        record.messages.append(
            {
                "role": "source",
                "content": f"现场复核 {after.run_id}",
                "detail": outcome.scope,
            }
        )
        self.keep(case_id)
        return outcome.as_dict()

    def _recheck_structure(
        self, record: CaseRecord, directory: Path, after: ObservationBundle,
    ) -> None:
        """Run the same structural tests again, on the same records of the new run.

        The recheck verdict stays with the task results; this only shows whether
        the constraints that failed before hold now.
        """

        from visiondoctor.investigation import Toolbox

        case = record.case
        by_reference = {
            item.reference: item.evidence_id for item in case.evidence
            if item.bundle_id == after.run_id
        }
        toolbox = Toolbox(
            case, FileBundleAdapter(directory), after,
            uploads=record.uploads,
            observation_dirs=tuple(Path(path) for path in record.observation_dirs),
        )
        for row in isolation_view(case):
            if not row["diagnosed_run"]:
                continue
            bindings, carried = {}, True
            for name, evidence_id in row["bindings"].items():
                source = next(item for item in case.evidence if item.evidence_id == evidence_id)
                if source.bundle_id != row["run_id"]:
                    bindings[name] = evidence_id  # a reference run stays the reference
                elif source.reference in by_reference:
                    bindings[name] = by_reference[source.reference]
                else:
                    carried = False
            if not carried:
                continue
            try:
                toolbox.structural_diagnose(
                    template_ids=row["templates"], part_id=row["part_id"], bindings=bindings,
                )
            except (KeyError, ValueError, TypeError) as exc:
                record.messages.append({
                    "role": "source", "content": f"复核核算未完成（工件 {row['part_id']}）",
                    "detail": str(exc),
                })

    # ---- what a viewer sees ------------------------------------------------------

    def view(self, case_id: str) -> dict[str, Any]:
        from visiondoctor.knowledge import catalogue

        from .graph import graph_view

        record = self.record(case_id)
        case = record.case
        verdict = source_layer_gate(case)
        layers = {item.evidence_id: item.layer for item in case.evidence}
        return {
            "case_id": case.case_id,
            "title": case.title,
            "observation": self._observation(record),
            "project": self._project(case),
            "chain": self._chain(case),
            "graph": graph_view(case.findings, layers),
            "knowledge_catalogue": catalogue(),
            "messages": record.messages,
            "hypotheses": [
                {
                    **item.model_dump(mode="json"),
                    "segment_name": SEGMENT_NAME[item.target_segment],
                }
                for item in case.hypotheses
            ],
            "evidence": [
                {
                    "evidence_id": item.evidence_id,
                    "bundle_id": item.bundle_id,
                    "reference": item.reference,
                    "media_type": item.media_type,
                    "captured_at": item.captured_at.isoformat(),
                    "phase": item.phase,
                    "layer": item.layer,
                    "summary": item.summary,
                    "examined": item.evidence_id in case.examined,
                }
                for item in case.evidence
            ],
            "access": {
                "observation": bool(case.observations),
                "runnable": bool(case.project and case.project.runnable),
                "source_readable": bool(case.project and case.project.source_readable),
            },
            "gates": {
                "source_layer": {
                    **verdict.model_dump(mode="json"),
                    "basis": [
                        {"target_id": item.target_id, "note": item.note,
                         "evidence_ids": list(item.evidence_ids)}
                        for item in software_localizations(case)
                    ],
                },
            },
            "isolation": isolation_view(case),
            "node_standing": node_standing(case),
            "divergence": _divergence(case),
            "plans": self._plans(record),
            "turns": [turn.model_dump(mode="json") for turn in record.turns],
            "running": record.running,
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
            "runnable": case.project.runnable,
            "source_readable": case.project.source_readable,
        }

    @staticmethod
    def _chain(case: Case) -> list[dict[str, Any]]:
        rows = []
        for segment, status in case.demarcation().items():
            findings = [item for item in case.findings if item.segment is segment]
            rows.append(
                {
                    "segment": segment.value,
                    "name": SEGMENT_NAME[segment],
                    "status": status.value,
                    "scope": SEGMENT_SCOPE[segment],
                    "note": "；".join(item.note for item in findings),
                    "evidence_ids": sorted({key for item in findings for key in item.evidence_ids}),
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
                    "target_segment": SEGMENT_NAME[plan.target_segment],
                    "frozen_hash": plan.frozen_hash,
                    "diff": plan.diff,
                    "approved": decision.approved if decision else None,
                    "approver": decision.approver if decision else None,
                    "gate": approval_gate(plan, decision).model_dump(mode="json"),
                    "landed": record.landings.get(plan.plan_id),
                }
            )
        return rows

    @staticmethod
    def _plan(record: CaseRecord, plan_id: str) -> RepairPlan:
        for plan in record.case.repair_plans:
            if plan.plan_id == plan_id:
                return plan
        raise KeyError(f"没有这个候选：{plan_id}")


def _divergence(case: Case) -> list[dict[str, str]]:
    """Where the model's verdict and the host's isolation say different things."""

    standing = node_standing(case)
    rows = []
    for finding in case.findings:
        # Handoffs are judged on their own; only a verdict on the node itself is compared.
        if consuming_node(finding.target_id) != finding.target_id:
            continue
        host = standing.get(str(finding.target_id))
        model = finding.status.value
        if (model, host) in {("cleared", "candidate"), ("suspect", "exonerated"),
                             ("cleared", "unexamined")}:
            rows.append({"target_id": str(finding.target_id), "model": model, "host": host})
    return rows


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


def _observed_revision(record: CaseRecord) -> str:
    """The commit the observation says was running, or the tip when none said."""

    if record.bundle is None:
        return "HEAD"
    return str(record.bundle.project_revision.get("commit") or "HEAD")


def _resolve_revision(repository: Path, revision: str) -> str:
    """Pin the revision to a commit that this repository actually holds.

    A commit the repository has never heard of means the observation and the
    source belong to different runs.  Say so here, where it can still be fixed.
    """

    import subprocess

    if not (repository / ".git").exists():
        raise ValueError(f"{repository} 不是一个 Git 仓库")
    found = subprocess.run(
        ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    if found.returncode != 0:
        raise ValueError(
            f"这个仓库里没有 {revision} 这个提交——观察包和源码很可能不是同一次运行。"
            f"请确认仓库路径：{repository}"
        )
    return found.stdout.strip()


def _call_summary(call: Any) -> dict[str, Any]:
    return {
        "name": call.name,
        "requested": list(call.requested),
        "delivered": list(call.delivered),
        "failure": call.failure,
        "arguments": {
            key: (value if len(str(value)) < 200 else str(value)[:200] + "…")
            for key, value in call.arguments.items()
        },
    }


def _model_gateway():
    from visiondoctor.llm import ModelSettings, OpenAICompatibleGateway

    return OpenAICompatibleGateway(ModelSettings.from_environment())
