from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Literal

from visiondoctor.demo.scenario import run_demo
from visiondoctor.schemas import Decision, WorkflowState


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout


def run_reliability_gate(
    root: Path,
    *,
    runs: int,
    sandbox_mode: Literal["local", "docker", "auto"],
    robot_backend: Literal["dataset", "gazebo", "auto"],
) -> dict:
    if runs < 1:
        raise ValueError("runs must be positive")
    root = root.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"reliability root must be new or empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for index in range(1, runs + 1):
        started = time.perf_counter()
        record: dict = {"run": index, "passed": False}
        try:
            result = run_demo(
                root / f"run-{index:02d}",
                sandbox_mode=sandbox_mode,
                robot_backend=robot_backend,
            )
            selected_validation = next(
                report
                for report in result.candidate_validations
                if report.candidate_id == result.selected_candidate.candidate_id
            )
            rejected = [
                report
                for report in result.candidate_validations
                if report.decision is Decision.REJECTED
            ]
            repository = Path(result.incident.repository.path)
            release = json.loads(
                (Path(result.run_root) / "release_decision.json").read_text(encoding="utf-8")
            )
            external_gates = [
                {
                    "name": gate.name,
                    "passed": gate.passed,
                    "infrastructure_error": gate.infrastructure_error,
                    "translation_error_m": gate.payload.get("tcp_translation_error_m"),
                    "rotation_error_rad": gate.payload.get("tcp_rotation_error_rad"),
                }
                for gate in result.external_gate_results
            ]
            required_external_gate = robot_backend == "gazebo" or (
                robot_backend == "auto" and bool(external_gates)
            )
            passed = (
                result.state is WorkflowState.AWAITING_HUMAN_APPROVAL
                and all(report.rollback_result == "SUCCESS" for report in rejected)
                and selected_validation.decision is Decision.PASS
                and len(selected_validation.passed_cases) == len(result.incident.case_set)
                and (
                    not required_external_gate
                    or (
                        bool(external_gates)
                        and all(gate["passed"] for gate in external_gates)
                    )
                )
                and release["automatic_merge"] is False
                and _git(repository, "rev-parse", "HEAD").strip()
                == result.incident.faulty_commit
                and _git(repository, "status", "--porcelain") == ""
            )
            record.update(
                {
                    "passed": passed,
                    "state": result.state.value,
                    "candidate_attempts": len(result.candidate_validations),
                    "rejected_candidates": len(rejected),
                    "selected_candidate": result.selected_candidate.candidate_id,
                    "selected_passed_cases": len(selected_validation.passed_cases),
                    "external_gates": external_gates,
                    "automatic_merge": release["automatic_merge"],
                    "run_root": result.run_root,
                }
            )
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["duration_s"] = time.perf_counter() - started
        records.append(record)
        _write_summary(
            root / "reliability_summary.json",
            runs=runs,
            sandbox_mode=sandbox_mode,
            robot_backend=robot_backend,
            records=records,
        )
    return _write_summary(
        root / "reliability_summary.json",
        runs=runs,
        sandbox_mode=sandbox_mode,
        robot_backend=robot_backend,
        records=records,
    )


def _write_summary(
    path: Path,
    *,
    runs: int,
    sandbox_mode: str,
    robot_backend: str,
    records: list[dict],
) -> dict:
    passed = sum(bool(record["passed"]) for record in records)
    summary = {
        "requested_runs": runs,
        "completed_runs": len(records),
        "passed_runs": passed,
        "all_passed": len(records) == runs and passed == runs,
        "sandbox_mode": sandbox_mode,
        "robot_backend": robot_backend,
        "runs": records,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return summary
