from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tests.support import root_cause_patch
from visiondoctor.adapters.dataset_execution import DatasetExecutionBackend
from visiondoctor.sandbox import DockerPythonRunner, GitWorktreeSandbox
from visiondoctor.schemas import CandidateKind, CandidateVersion
from visiondoctor.workflow import DemoRunResult


def _runner_or_skip() -> DockerPythonRunner:
    if not DockerPythonRunner.available():
        pytest.skip("Docker engine is unavailable")
    runner = DockerPythonRunner()
    project_root = Path(__file__).resolve().parents[1]
    runner.ensure_image(project_root / "docker" / "sandbox.Dockerfile", project_root)
    return runner


def test_docker_runner_enforces_runtime_boundaries(tmp_path: Path) -> None:
    runner = _runner_or_skip()
    script = tmp_path / "probe.py"
    script.write_text(
        """
import json
import os
import pathlib
import socket

def attempt(callback):
    try:
        callback()
    except Exception as exc:
        return type(exc).__name__
    return "SUCCEEDED"

result = {
    "uid": os.getuid(),
    "workspace_write": attempt(lambda: pathlib.Path("blocked.txt").write_text("x")),
    "root_write": attempt(lambda: pathlib.Path("/blocked.txt").write_text("x")),
    "network": attempt(lambda: socket.create_connection(("1.1.1.1", 53), timeout=0.5)),
    "memory_max": pathlib.Path("/sys/fs/cgroup/memory.max").read_text().strip(),
    "pids_max": pathlib.Path("/sys/fs/cgroup/pids.max").read_text().strip(),
    "cpu_max": pathlib.Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
}
print(json.dumps(result))
""".strip(),
        encoding="utf-8",
    )

    completed = runner.run_script(tmp_path, "probe.py", input_text=None, timeout_s=10)

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["uid"] == 65532
    assert result["workspace_write"] != "SUCCEEDED"
    assert result["root_write"] != "SUCCEEDED"
    assert result["network"] != "SUCCEEDED"
    assert result["memory_max"] == str(512 * 1024 * 1024)
    assert result["pids_max"] == "64"
    quota, period = (int(value) for value in result["cpu_max"].split())
    assert quota / period == 1.0


def test_docker_runner_kills_timed_out_container(tmp_path: Path) -> None:
    runner = _runner_or_skip()
    (tmp_path / "sleep.py").write_text("import time; time.sleep(30)", encoding="utf-8")

    completed = runner.run_script(tmp_path, "sleep.py", input_text=None, timeout_s=0.5)

    assert completed.timed_out
    process = subprocess.run(
        [
            runner.docker_executable,
            "ps",
            "--all",
            "--filter",
            "label=visiondoctor.component=candidate-sandbox",
            "--format",
            "{{.Names}}",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        env=runner._docker_environment(runner.docker_executable),
    )
    assert process.returncode == 0
    assert process.stdout.strip() == ""


def test_secure_runner_accepts_an_unregistered_candidate(
    demo_result: DemoRunResult, tmp_path: Path
) -> None:
    runner = _runner_or_skip()
    repository = Path(demo_result.incident.repository.path)
    backend = DatasetExecutionBackend(
        demo_result.incident,
        GitWorktreeSandbox(repository, tmp_path / "worktrees"),
        runner=runner,
    )
    candidate = CandidateVersion(
        candidate_id="docker-unregistered",
        kind=CandidateKind.ROOT_CAUSE_FIX,
        base_commit=demo_result.incident.faulty_commit,
        patch_text=root_cause_patch(),
        expected_changed_files=(
            "src/pose_transformer.py",
            "tests/test_transform_order_regression.py",
        ),
        rationale="not registered in the local trusted-patch allowlist",
    )

    handle = backend.prepare(candidate)
    try:
        assert all(check.passed for check in backend.policy_checks(handle))
    finally:
        assert backend.cleanup(handle, rollback=False)
