from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from visiondoctor.schemas import WorkflowState
from visiondoctor.storage import ApprovalRecord, SqliteRunRepository


@dataclass(frozen=True)
class ApprovalOutcome:
    run_id: str
    state: WorkflowState
    approval: ApprovalRecord
    branch_name: str | None
    candidate_commit: str | None


def _git(repository: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repository), *args],
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


class ApprovalService:
    def __init__(self, repository: SqliteRunRepository) -> None:
        self.repository = repository

    def act(
        self,
        run_id: str,
        *,
        action: Literal["approve", "reject", "additional_testing"],
        actor: str,
        note: str = "",
    ) -> ApprovalOutcome:
        run = self.repository.get_run(run_id)
        if run.state is not WorkflowState.AWAITING_HUMAN_APPROVAL:
            raise ValueError(f"run is not awaiting approval: {run.state}")
        run_root = Path(run.run_root).resolve()
        release_path = run_root / "release_decision.json"
        original_release = release_path.read_bytes()
        release = json.loads(release_path.read_text(encoding="utf-8"))
        candidate_id = str(release["candidate_id"])
        branch_name: str | None = None
        candidate_commit: str | None = None
        generated_paths = (run_root / "pr_materials.json", run_root / "pr_materials.md")
        originals = {path: path.read_bytes() if path.exists() else None for path in generated_paths}
        try:
            if action == "approve":
                branch_name, candidate_commit = self._prepare_branch(
                    run_root, candidate_id, actor
                )
                self._write_pr_materials(
                    run_root,
                    candidate_id=candidate_id,
                    branch_name=branch_name,
                    candidate_commit=candidate_commit,
                    actor=actor,
                    note=note,
                )
            release.update(
                {
                    "human_action": action,
                    "human_actor": actor,
                    "human_note": note,
                    "branch_name": branch_name,
                    "candidate_commit": candidate_commit,
                    "automatic_merge": False,
                }
            )
            self._atomic_json(release_path, release)
            approval = self.repository.record_approval(
                run_id,
                candidate_id=candidate_id,
                action=action,
                actor=actor,
                note=note,
                branch_name=branch_name,
                candidate_commit=candidate_commit,
            )
        except Exception:
            release_path.write_bytes(original_release)
            for path, content in originals.items():
                if content is None:
                    path.unlink(missing_ok=True)
                else:
                    path.write_bytes(content)
            if branch_name:
                incident = json.loads((run_root / "incident.json").read_text(encoding="utf-8"))
                self._delete_branch(Path(incident["repository"]["path"]), branch_name)
            raise
        self.repository.refresh_artifacts(run_id)
        return ApprovalOutcome(
            run_id=run_id,
            state=self.repository.get_run(run_id).state,
            approval=approval,
            branch_name=branch_name,
            candidate_commit=candidate_commit,
        )

    def _prepare_branch(self, run_root: Path, candidate_id: str, actor: str) -> tuple[str, str]:
        incident = json.loads((run_root / "incident.json").read_text(encoding="utf-8"))
        repository = Path(incident["repository"]["path"]).resolve()
        faulty_commit = str(incident["faulty_commit"])
        original_head = _git(repository, "rev-parse", "HEAD").stdout.strip()
        original_status = _git(repository, "status", "--porcelain").stdout
        if original_status:
            raise ValueError("target repository is not clean")
        safe_incident = re.sub(r"[^A-Za-z0-9_.-]+", "-", incident["incident_id"])
        safe_candidate = re.sub(r"[^A-Za-z0-9_.-]+", "-", candidate_id)
        branch_name = f"visiondoctor/{safe_incident}/{safe_candidate}"
        exists = _git(
            repository,
            "show-ref",
            "--verify",
            f"refs/heads/{branch_name}",
            check=False,
        )
        if exists.returncode == 0:
            raise ValueError(f"candidate branch already exists: {branch_name}")
        workspace = run_root.parents[1].resolve()
        worktree_root = (workspace / "approval-worktrees").resolve()
        worktree_root.mkdir(parents=True, exist_ok=True)
        worktree = (worktree_root / safe_candidate).resolve()
        if worktree_root not in worktree.parents or worktree.exists():
            raise ValueError("approval worktree path is unsafe or already exists")
        patch_path = run_root / f"{candidate_id}.diff"
        branch_created = False
        try:
            try:
                _git(
                    repository,
                    "worktree",
                    "add",
                    "-b",
                    branch_name,
                    str(worktree),
                    faulty_commit,
                )
                branch_created = True
                applied = subprocess.run(
                    ["git", "-C", str(worktree), "apply", "--whitespace=nowarn", "-"],
                    input=patch_path.read_bytes(),
                    capture_output=True,
                    check=False,
                )
                if applied.returncode != 0:
                    error = applied.stderr.decode("utf-8", errors="replace").strip()
                    raise RuntimeError(f"approval patch apply failed: {error}")
                _git(worktree, "add", "--all")
                _git(
                    worktree,
                    "-c",
                    "user.name=VisionDoctor Approval",
                    "-c",
                    "user.email=visiondoctor@example.invalid",
                    "commit",
                    "-m",
                    f"fix: {candidate_id} approved by {actor}",
                )
                candidate_commit = _git(worktree, "rev-parse", "HEAD").stdout.strip()
            finally:
                _git(repository, "worktree", "remove", "--force", str(worktree), check=False)
                _git(repository, "worktree", "prune", check=False)
            if _git(repository, "rev-parse", "HEAD").stdout.strip() != original_head:
                raise RuntimeError("approval changed the checked-out target branch")
            if _git(repository, "status", "--porcelain").stdout != original_status:
                raise RuntimeError("approval dirtied the target repository")
        except Exception:
            if branch_created:
                _git(repository, "branch", "-D", branch_name, check=False)
            raise
        return branch_name, candidate_commit

    @staticmethod
    def _delete_branch(repository: Path, branch_name: str) -> None:
        _git(repository.resolve(), "branch", "-D", branch_name, check=False)

    @staticmethod
    def _write_pr_materials(
        run_root: Path,
        *,
        candidate_id: str,
        branch_name: str,
        candidate_commit: str,
        actor: str,
        note: str,
    ) -> None:
        materials = {
            "title": "Fix target-source transform composition order",
            "candidate_id": candidate_id,
            "branch_name": branch_name,
            "candidate_commit": candidate_commit,
            "approved_by": actor,
            "approval_note": note,
            "merge_performed": False,
            "validation_report": f"{candidate_id}_validation.json",
            "diff": f"{candidate_id}.diff",
            "rollback": "revert the candidate commit or delete the unmerged branch",
        }
        ApprovalService._atomic_json(run_root / "pr_materials.json", materials)
        markdown = (
            f"# {materials['title']}\n\n"
            f"- Candidate: `{candidate_id}`\n"
            f"- Branch: `{branch_name}`\n"
            f"- Commit: `{candidate_commit}`\n"
            f"- Approved by: `{actor}`\n"
            "- Automatic merge: **No**\n\n"
            f"Approval note: {note or 'None'}\n\n"
            f"Validation: `{materials['validation_report']}`\n"
        )
        temporary = run_root / "pr_materials.md.tmp"
        temporary.write_text(markdown, encoding="utf-8")
        temporary.replace(run_root / "pr_materials.md")

    @staticmethod
    def _atomic_json(path: Path, value: dict) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
