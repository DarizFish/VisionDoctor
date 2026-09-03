"""Put an approved patch into the repository it was raised against.

Retyping a diff by hand is the one step in this path where a person can
silently introduce something nobody reviewed, so the approved bytes are written
by the host rather than copied by the reader.  The commit lands on the branch
that is checked out, which is what someone looking at the files expects to see.
Deploying it to the cell stays a human act.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .binding import ProjectBinding

AUTHOR = ("-c", "user.name=VisionDoctor", "-c", "user.email=visiondoctor@localhost")


def _git(
    repository: Path, *arguments: str, feed: bytes | None = None
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *arguments], cwd=repository, capture_output=True, input=feed, check=False
    )


def _said(result: subprocess.CompletedProcess[bytes]) -> str:
    return (result.stderr or result.stdout).decode("utf-8", errors="replace").strip()


def land(
    *,
    binding: ProjectBinding,
    plan_id: str,
    frozen_hash: str,
    diff: str,
    case_id: str,
    approver: str,
) -> dict[str, Any]:
    """Apply the frozen patch and commit it, on whatever branch is checked out."""

    repository = binding.repository
    #: --index applies and stages exactly the paths the diff names, so the commit
    #: carries the reviewed change and nothing else the tree happened to hold.
    applied = _git(repository, "apply", "--index", "--whitespace=nowarn", "-", feed=diff.encode())
    if applied.returncode != 0:
        raise RuntimeError(_said(applied) or "补丁没能应用到这个仓库")
    message = (
        f"{case_id} {plan_id}: apply the approved repair\n\n"
        f"Approved by {approver}.\n"
        f"Frozen {frozen_hash}\n"
        f"Base {binding.revision}\n"
    )
    made = _git(repository, *AUTHOR, "commit", "-m", message)
    if made.returncode != 0:
        _git(repository, "reset", "--hard", "HEAD")
        raise RuntimeError(_said(made) or "提交失败")
    commit = _git(repository, "rev-parse", "HEAD").stdout.decode().strip()
    branch = _git(repository, "rev-parse", "--abbrev-ref", "HEAD").stdout.decode().strip()
    changed = _git(repository, "show", "--name-only", "--format=", commit)
    return {
        "branch": branch,
        "commit": commit,
        "files": _said(changed).split(),
    }
