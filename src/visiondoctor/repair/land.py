"""Put an approved patch into the repository, on a branch of its own.

Retyping a diff by hand is the one step in this path where a person can
silently introduce something nobody reviewed, so the approved bytes are written
by the host rather than copied by the reader.  What is written is a branch and
nothing else: the current checkout does not move, no remote is touched, and
deploying the change to the cell stays a human act.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from visiondoctor.sandbox.git_worktree import GitWorktreeSandbox
from visiondoctor.schemas.models import CandidateKind, CandidateVersion

from .binding import ProjectBinding

AUTHOR = ("-c", "user.name=VisionDoctor", "-c", "user.email=visiondoctor@localhost")


def _git(cwd: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments], cwd=cwd, capture_output=True, text=True, check=False
    )


def land(
    *,
    binding: ProjectBinding,
    plan_id: str,
    frozen_hash: str,
    diff: str,
    case_id: str,
    approver: str,
    sandbox_root: Path,
) -> dict[str, Any]:
    """Commit the frozen patch onto a new branch and report where it went."""

    branch = f"visiondoctor/{plan_id.lower()}-{frozen_hash[:8]}"
    if _git(binding.repository, "rev-parse", "--verify", branch).returncode == 0:
        return {"branch": branch, "commit": "", "note": "这个分支已经存在，没有重复写入"}
    sandbox = GitWorktreeSandbox(binding.repository, sandbox_root)
    handle = sandbox.create(
        CandidateVersion(
            candidate_id=f"land-{plan_id}",
            kind=CandidateKind.ROOT_CAUSE_FIX,
            base_commit=binding.revision,
            patch_text=diff,
            rationale=f"approved by {approver}",
        )
    )
    try:
        worktree = handle.worktree
        _git(worktree, "add", "-A", "--", ".")
        message = (
            f"{case_id} {plan_id}: apply the approved repair\n\n"
            f"Approved by {approver}.\n"
            f"Frozen {frozen_hash}\n"
            f"Base {binding.revision}\n"
        )
        made = _git(worktree, *AUTHOR, "commit", "-m", message)
        if made.returncode != 0:
            raise RuntimeError(made.stderr.strip() or made.stdout.strip())
        commit = _git(worktree, "rev-parse", "HEAD").stdout.strip()
        named = _git(binding.repository, "branch", branch, commit)
        if named.returncode != 0:
            raise RuntimeError(named.stderr.strip())
    finally:
        sandbox.cleanup(handle, rollback=True)
    return {"branch": branch, "commit": commit, "note": ""}
