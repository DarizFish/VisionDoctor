"""Re-run the project on what it was actually fed, with the candidate applied.

REPLAY proves one thing and refuses to imply more: on the inputs this cell really
produced, the changed code now computes what the chain intended.  It says nothing
about the cell recovering -- only new evidence collected after the change is
applied can say that.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from visiondoctor.sandbox.git_worktree import GitWorktreeSandbox
from visiondoctor.schemas.models import CandidateKind, CandidateVersion

from .binding import ProjectBinding


@dataclass(frozen=True)
class CommandOutcome:
    command: tuple[str, ...]
    exit_code: int
    output_tail: str

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "command": list(self.command),
            "exit_code": self.exit_code,
            "succeeded": self.succeeded,
            "output_tail": self.output_tail,
        }


@dataclass(frozen=True)
class ReplayOutcome:
    """What an isolated re-run produced.  A verdict is left to whoever reads it."""

    base_revision: str
    changed_files: tuple[str, ...]
    project_tests: CommandOutcome | None
    replays: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": "isolated_replay",
            "base_revision": self.base_revision,
            "changed_files": list(self.changed_files),
            "project_tests": self.project_tests.as_dict() if self.project_tests else None,
            "replays": list(self.replays),
        }


def _run(command: tuple[str, ...], cwd: Path, timeout_s: float) -> CommandOutcome:
    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s,
        check=False,
        env={**_clean_environment(), "PYTHONPATH": str(cwd)},
    )
    tail = (result.stdout + result.stderr).strip()
    return CommandOutcome(command=command, exit_code=result.returncode, output_tail=tail[-1200:])


def _clean_environment() -> dict[str, str]:
    import os

    keep = ("PATH", "SYSTEMROOT", "COMSPEC", "TEMP", "TMP", "LANG", "PYTHONIOENCODING")
    return {name: os.environ[name] for name in keep if name in os.environ}


def replay(
    *,
    binding: ProjectBinding,
    patch_text: str,
    candidate_id: str,
    sandbox_root: Path,
    inputs: dict[str, bytes],
    timeout_s: float = 60.0,
) -> ReplayOutcome:
    """Apply the candidate in a detached worktree and feed it the recorded inputs."""

    sandbox = GitWorktreeSandbox(binding.repository, sandbox_root)
    handle = sandbox.create(
        CandidateVersion(
            candidate_id=candidate_id,
            kind=CandidateKind.ROOT_CAUSE_FIX,
            base_commit=binding.revision,
            patch_text=patch_text,
            rationale="candidate raised by the investigation",
        )
    )
    try:
        worktree = handle.worktree
        changed = subprocess.run(
            ["git", "diff", "--name-only", binding.revision, "--"],
            cwd=worktree,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.split()
        tests = _run(binding.test_command, worktree, timeout_s) if binding.test_command else None
        replays: list[dict[str, Any]] = []
        for name, payload in inputs.items():
            scratch = worktree / ".replay"
            scratch.mkdir(exist_ok=True)
            source = scratch / f"{name}-input.json"
            target = scratch / f"{name}-output.json"
            source.write_bytes(payload)
            outcome = _run(
                binding.replay_command_for(source, target, scratch / f"{name}-log.jsonl"),
                worktree,
                timeout_s,
            )
            produced = (
                json.loads(target.read_text(encoding="utf-8")) if target.is_file() else None
            )
            replays.append({"input": name, "run": outcome.as_dict(), "output": produced})
        return ReplayOutcome(
            base_revision=binding.revision,
            changed_files=tuple(sorted(set(changed))),
            project_tests=tests,
            replays=tuple(replays),
        )
    finally:
        sandbox.cleanup(handle, rollback=True)
