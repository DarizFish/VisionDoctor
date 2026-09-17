"""Keep cases across a restart.

A diagnosis is worked out over several turns of real model time, and losing it
because a process stopped means doing that work again.  So the ledger is written
to a file per case after every change and read back at start-up.

Observations are not copied -- the export directory is already on disk, so only
its path is kept and the bundle is collected again.  A turn that was in flight
is not restored: its thread is gone, and pretending otherwise would show work
that nobody is doing.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from visiondoctor.environment import FileBundleAdapter
from visiondoctor.repair import ProjectBinding, Recheck

from .case import Case, Evidence
from .chain import Hypothesis, SegmentFinding
from .repair import ApprovalRecord, RepairPlan

ROOT = Path(".runtime/vd-cases")


def _binding(binding: ProjectBinding | None) -> dict[str, Any] | None:
    if binding is None:
        return None
    return {
        "repository": str(binding.repository),
        "revision": binding.revision,
        "replay_command": list(binding.replay_command),
        "test_command": list(binding.test_command) if binding.test_command else None,
        "source_readable": binding.source_readable,
    }


def _recheck(outcome: Recheck | None) -> dict[str, Any] | None:
    if outcome is None:
        return None
    return {
        "before_run_id": outcome.before_run_id,
        "after_run_id": outcome.after_run_id,
        "after_created_at": outcome.after_created_at.isoformat(),
        "applied_at": outcome.applied_at.isoformat() if outcome.applied_at else None,
        "before": outcome.before,
        "after": outcome.after,
        "observation_started_at": (
            outcome.observation_started_at.isoformat() if outcome.observation_started_at else None
        ),
        "context_matches": outcome.context_matches,
    }


def save(record: Any, root: Path = ROOT) -> None:
    """Write one case out.  Called after anything that changes it."""

    case = record.case
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "case": {
            "case_id": case.case_id,
            "title": case.title,
            "guided_motion": case.guided_motion,
            "project": _binding(case.project),
            "evidence": [item.model_dump(mode="json") for item in case.evidence],
            "findings": [item.model_dump(mode="json") for item in case.findings],
            "hypotheses": [item.model_dump(mode="json") for item in case.hypotheses],
            "repair_plans": [item.model_dump(mode="json") for item in case.repair_plans],
            "examined": sorted(case.examined),
            "transcript": case.transcript,
        },
        "observations": list(record.observation_dirs),
        "turns": [turn.model_dump(mode="json") for turn in record.turns],
        "messages": record.messages,
        "uploads": {key: str(value) for key, value in record.uploads.items()},
        "approvals": {
            key: value.model_dump(mode="json") for key, value in record.approvals.items()
        },
        "landings": record.landings,
        "applied_at": record.applied_at.isoformat() if record.applied_at else None,
        "recheck": _recheck(record.recheck),
    }
    target = root / f"{case.case_id}.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def load_all(make_record: Any, root: Path = ROOT) -> dict[str, Any]:
    """Read every case back.  A file that will not parse is skipped, not fatal."""

    from visiondoctor.investigation.turn import DecisionTurn

    records: dict[str, Any] = {}
    for path in sorted(root.glob("CASE-*.json")) if root.is_dir() else ():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        try:
            record = _load_one(payload, make_record, DecisionTurn)
        except (KeyError, TypeError, ValueError):
            continue
        records[record.case.case_id] = record
    return records


def _load_one(payload: dict[str, Any], make_record: Any, turn_type: Any) -> Any:
    """Rebuild one record; a schema mismatch raises and the file is skipped."""

    held = payload["case"]
    case = Case(held["case_id"], held["title"], guided_motion=held["guided_motion"])
    if held["project"]:
        case.bind_project(
            ProjectBinding(
                repository=Path(held["project"]["repository"]),
                revision=held["project"]["revision"],
                replay_command=tuple(held["project"]["replay_command"]),
                test_command=(
                    tuple(held["project"]["test_command"])
                    if held["project"]["test_command"]
                    else None
                ),
                source_readable=held["project"].get("source_readable", True),
            )
        )
    case.evidence = [Evidence(**item) for item in held["evidence"]]
    case.findings = [SegmentFinding(**item) for item in held["findings"]]
    case.hypotheses = [Hypothesis(**item) for item in held["hypotheses"]]
    case.repair_plans = [RepairPlan(**item) for item in held["repair_plans"]]
    case.examined = set(held["examined"])
    case.transcript = held["transcript"]

    record = make_record(case)
    record.observation_dirs = list(payload["observations"])
    for directory in record.observation_dirs:
        adapter = FileBundleAdapter(Path(directory))
        bundle = adapter.collect()
        case.observations.append(bundle)
        if record.bundle is None:
            record.adapter, record.bundle = adapter, bundle
    record.turns = [turn_type(**item) for item in payload["turns"]]
    record.messages = payload["messages"]
    record.uploads = {key: Path(value) for key, value in payload["uploads"].items()}
    record.approvals = {
        key: ApprovalRecord(**value) for key, value in payload["approvals"].items()
    }
    record.landings = payload["landings"]
    record.applied_at = (
        datetime.fromisoformat(payload["applied_at"]) if payload["applied_at"] else None
    )
    if payload["recheck"]:
        held_recheck = dict(payload["recheck"])
        held_recheck["after_created_at"] = datetime.fromisoformat(
            held_recheck["after_created_at"]
        )
        held_recheck["applied_at"] = (
            datetime.fromisoformat(held_recheck["applied_at"])
            if held_recheck["applied_at"]
            else None
        )
        if held_recheck.get("observation_started_at"):
            held_recheck["observation_started_at"] = datetime.fromisoformat(
                held_recheck["observation_started_at"]
            )
        record.recheck = Recheck(**held_recheck)
    return record
