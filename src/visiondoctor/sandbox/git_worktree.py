from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from visiondoctor.adapters.base import RuntimeHandle
from visiondoctor.schemas import CandidateVersion


class SandboxError(RuntimeError):
    pass


def _git(
    repository: Path,
    *args: str,
    check: bool = True,
    input_text: str | None = None,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[str]:
    if input_text is not None and input_bytes is not None:
        raise ValueError("git input must be text or bytes, not both")
    if input_bytes is not None:
        process = subprocess.run(
            ["git", "-C", str(repository), *args],
            check=check,
            input=input_bytes,
            capture_output=True,
        )
        return subprocess.CompletedProcess(
            process.args,
            process.returncode,
            process.stdout.decode("utf-8", errors="replace"),
            process.stderr.decode("utf-8", errors="replace"),
        )
    return subprocess.run(
        ["git", "-C", str(repository), *args],
        check=check,
        input=input_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


class GitWorktreeSandbox:
    """Candidate isolation using real detached Git worktrees."""

    def __init__(self, repository: Path, sandbox_root: Path) -> None:
        self.repository = repository.resolve()
        self.sandbox_root = sandbox_root.resolve()
        self.sandbox_root.mkdir(parents=True, exist_ok=True)
        if not (self.repository / ".git").exists():
            raise SandboxError(f"not a Git repository: {self.repository}")
        self._repository_head = _git(self.repository, "rev-parse", "HEAD").stdout.strip()
        self._repository_status = _git(self.repository, "status", "--porcelain").stdout

    def create(self, candidate: CandidateVersion) -> RuntimeHandle:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", candidate.candidate_id)
        worktree = (self.sandbox_root / safe_name).resolve()
        if self.sandbox_root not in worktree.parents:
            raise SandboxError("resolved worktree escaped sandbox root")
        if worktree.exists():
            raise SandboxError(f"worktree already exists: {worktree}")
        add = _git(
            self.repository,
            "worktree",
            "add",
            "--detach",
            str(worktree),
            candidate.base_commit,
            check=False,
        )
        if add.returncode != 0:
            raise SandboxError(add.stderr.strip() or add.stdout.strip())
        if candidate.patch_text:
            apply_result = _git(
                worktree,
                "apply",
                "--whitespace=nowarn",
                "-",
                check=False,
                input_bytes=candidate.patch_text.encode("utf-8"),
            )
            if apply_result.returncode != 0:
                self.remove_path(worktree)
                raise SandboxError(
                    f"candidate patch could not be applied: {apply_result.stderr.strip()}"
                )
            # Make newly created files visible to diff/numstat without staging their content.
            intent = _git(worktree, "add", "--intent-to-add", "--", ".", check=False)
            if intent.returncode != 0:
                self.remove_path(worktree)
                raise SandboxError(
                    f"candidate files could not be registered for audit: {intent.stderr.strip()}"
                )
        return RuntimeHandle(
            handle_id=f"HANDLE-{candidate.candidate_id}",
            candidate=candidate,
            worktree=worktree,
        )

    def diff(self, handle: RuntimeHandle) -> str:
        return _git(handle.worktree, "diff", "--binary", handle.candidate.base_commit, "--").stdout

    def cleanup(self, handle: RuntimeHandle, *, rollback: bool) -> bool:
        self.remove_path(handle.worktree)
        current_head = _git(self.repository, "rev-parse", "HEAD").stdout.strip()
        current_status = _git(self.repository, "status", "--porcelain").stdout
        repository_unchanged = (
            current_head == self._repository_head and current_status == self._repository_status
        )
        return (not handle.worktree.exists()) and repository_unchanged

    def remove_path(self, worktree: Path) -> None:
        resolved = worktree.resolve()
        if self.sandbox_root not in resolved.parents:
            raise SandboxError("refusing to remove a path outside the sandbox root")
        _git(
            self.repository,
            "worktree",
            "remove",
            "--force",
            str(resolved),
            check=False,
        )
        if resolved.exists():
            shutil.rmtree(resolved)
        _git(self.repository, "worktree", "prune", check=False)
