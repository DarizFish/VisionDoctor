from __future__ import annotations

import ast
import fnmatch
import re
import subprocess
from pathlib import Path

from visiondoctor.schemas import PatchPolicy, PolicyCheck


def _git(worktree: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(worktree), *args],
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _matches(path: str, patterns: tuple[str, ...]) -> bool:
    normalized = path.replace("\\", "/")
    return any(fnmatch.fnmatchcase(normalized, pattern) for pattern in patterns)


def _test_names(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
            "test"
        ):
            names.add(node.name)
    return names


class PatchPolicyEvaluator:
    SKIP_PATTERN = re.compile(
        r"@(unittest\.)?skip|pytest\.mark\.(skip|skipif|xfail)|\.skipTest\(",
        re.IGNORECASE,
    )

    def evaluate(
        self, worktree: Path, base_commit: str, policy: PatchPolicy
    ) -> tuple[PolicyCheck, ...]:
        changed_output = _git(worktree, "diff", "--name-only", base_commit, "--").stdout
        changed_files = tuple(
            line.strip().replace("\\", "/") for line in changed_output.splitlines() if line.strip()
        )
        numstat = _git(worktree, "diff", "--numstat", base_commit, "--").stdout
        changed_lines = 0
        for line in numstat.splitlines():
            added, deleted, _path = line.split("\t", maxsplit=2)
            if added == "-" or deleted == "-":
                changed_lines = policy.max_changed_lines + 1
                break
            changed_lines += int(added) + int(deleted)

        allowed = bool(changed_files) and all(
            _matches(path, policy.allowed_globs) for path in changed_files
        )
        forbidden = [path for path in changed_files if _matches(path, policy.forbidden_globs)]
        deleted = tuple(
            line.strip().replace("\\", "/")
            for line in _git(
                worktree, "diff", "--diff-filter=D", "--name-only", base_commit, "--"
            ).stdout.splitlines()
            if line.strip()
        )
        deleted_tests = [path for path in deleted if path.startswith("tests/")]

        removed_test_names: list[str] = []
        skipped_tests: list[str] = []
        for path in changed_files:
            if not (path.startswith("tests/") and path.endswith(".py")):
                continue
            current_path = worktree / path
            current_source = (
                current_path.read_text(encoding="utf-8") if current_path.exists() else ""
            )
            baseline = _git(worktree, "show", f"{base_commit}:{path}", check=False)
            baseline_source = baseline.stdout if baseline.returncode == 0 else ""
            removed = _test_names(baseline_source) - _test_names(current_source)
            removed_test_names.extend(f"{path}:{name}" for name in sorted(removed))
            if self.SKIP_PATTERN.search(current_source):
                skipped_tests.append(path)

        diff_check = _git(worktree, "diff", "--check", base_commit, "--", check=False)
        return (
            PolicyCheck(
                name="changed_file_count",
                passed=0 < len(changed_files) <= policy.max_files,
                details=f"{len(changed_files)} changed; limit {policy.max_files}: {changed_files}",
            ),
            PolicyCheck(
                name="changed_line_count",
                passed=changed_lines <= policy.max_changed_lines,
                details=f"{changed_lines} added/deleted lines; limit {policy.max_changed_lines}",
            ),
            PolicyCheck(
                name="allowed_paths",
                passed=allowed,
                details=f"changed paths: {changed_files}",
            ),
            PolicyCheck(
                name="forbidden_paths",
                passed=not forbidden,
                details="none" if not forbidden else f"forbidden changes: {forbidden}",
            ),
            PolicyCheck(
                name="failed_tests_not_deleted",
                passed=not policy.forbid_test_removal
                or (not deleted_tests and not removed_test_names),
                details=(
                    "no tests removed"
                    if not deleted_tests and not removed_test_names
                    else f"deleted={deleted_tests}; removed={removed_test_names}"
                ),
            ),
            PolicyCheck(
                name="tests_not_skipped",
                passed=not policy.forbid_test_skips or not skipped_tests,
                details="no skip markers"
                if not skipped_tests
                else f"skip markers: {skipped_tests}",
            ),
            PolicyCheck(
                name="git_diff_check",
                passed=diff_check.returncode == 0,
                details=diff_check.stderr.strip() or diff_check.stdout.strip() or "clean",
            ),
        )
