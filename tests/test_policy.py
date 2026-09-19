from __future__ import annotations

from pathlib import Path


from tests.support import root_cause_patch
from visiondoctor.demo.repository_builder import build_demo_repository
from visiondoctor.sandbox import GitWorktreeSandbox, PatchPolicyEvaluator
from visiondoctor.schemas import CandidateKind, CandidateVersion, PatchPolicy

def _check(checks, name: str):
    return next(check for check in checks if check.name == name)

def test_policy_counts_new_regression_test_and_preserves_repository(tmp_path: Path) -> None:
    repository = build_demo_repository(tmp_path / "repo")
    sandbox = GitWorktreeSandbox(repository.path, tmp_path / "worktrees")
    candidate = CandidateVersion(
        candidate_id="candidate-b",
        kind=CandidateKind.ROOT_CAUSE_FIX,
        base_commit=repository.faulty_commit,
        patch_text=root_cause_patch(),
        rationale="root cause",
        expected_changed_files=(
            "src/pose_transformer.py",
            "tests/test_transform_order_regression.py",
        ),
    )
    handle = sandbox.create(candidate)
    checks = PatchPolicyEvaluator().evaluate(
        handle.worktree, repository.faulty_commit, PatchPolicy()
    )

    assert _check(checks, "changed_file_count").passed
    assert "2 changed" in _check(checks, "changed_file_count").details
    assert "tests/test_transform_order_regression.py" in sandbox.diff(handle)
    assert sandbox.cleanup(handle, rollback=False)

def test_policy_rejects_skipped_tests(tmp_path: Path) -> None:
    repository = build_demo_repository(tmp_path / "repo")
    sandbox = GitWorktreeSandbox(repository.path, tmp_path / "worktrees")
    candidate = CandidateVersion(
        candidate_id="candidate-skip",
        kind=CandidateKind.ROOT_CAUSE_FIX,
        base_commit=repository.faulty_commit,
        patch_text=root_cause_patch(),
        rationale="malicious skip",
    )
    handle = sandbox.create(candidate)
    test_path = handle.worktree / "tests" / "test_transform_order_regression.py"
    source = test_path.read_text(encoding="utf-8")
    test_path.write_text(
        source.replace(
            "    def test_non_commuting",
            '    @unittest.skip("hide failure")\n    def test_non_commuting',
        ),
        encoding="utf-8",
    )
    checks = PatchPolicyEvaluator().evaluate(
        handle.worktree, repository.faulty_commit, PatchPolicy()
    )

    assert not _check(checks, "tests_not_skipped").passed
    assert sandbox.cleanup(handle, rollback=True)

