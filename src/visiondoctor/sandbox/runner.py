from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False


class PythonSandboxRunner(Protocol):
    secure_for_untrusted: bool

    def run_script(
        self,
        worktree: Path,
        script: str,
        *,
        input_text: str | None,
        timeout_s: float,
    ) -> CommandResult: ...


class LocalPythonRunner:
    secure_for_untrusted = False

    def run_script(
        self,
        worktree: Path,
        script: str,
        *,
        input_text: str | None,
        timeout_s: float,
    ) -> CommandResult:
        started = time.perf_counter()
        try:
            process = subprocess.run(
                [sys.executable, "-I", script],
                cwd=worktree,
                input=input_text,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s,
                check=False,
                env=self._sanitized_environment(),
            )
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                returncode=124,
                stdout=exc.stdout or "",
                stderr=exc.stderr or str(exc),
                duration_s=time.perf_counter() - started,
                timed_out=True,
            )
        return CommandResult(
            returncode=process.returncode,
            stdout=process.stdout,
            stderr=process.stderr,
            duration_s=time.perf_counter() - started,
        )

    @staticmethod
    def _sanitized_environment() -> dict[str, str]:
        allowed = ("SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATH", "PATHEXT", "COMSPEC")
        environment = {key: os.environ[key] for key in allowed if key in os.environ}
        environment.update(
            {
                "PYTHONIOENCODING": "utf-8",
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "OPENBLAS_NUM_THREADS": "1",
            }
        )
        return environment


class DockerPythonRunner:
    secure_for_untrusted = True
    DEFAULT_IMAGE = "visiondoctor/sandbox-python:3.11-v1"

    def __init__(
        self,
        *,
        image: str = DEFAULT_IMAGE,
        docker_executable: str | None = None,
        memory: str = "512m",
        cpus: float = 1.0,
        pids_limit: int = 64,
    ) -> None:
        self.image = image
        self.docker_executable = docker_executable or self.find_docker()
        if self.docker_executable is None:
            raise RuntimeError("docker CLI is not installed")
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = pids_limit

    @staticmethod
    def find_docker() -> str | None:
        executable = shutil.which("docker")
        if executable:
            return executable
        candidate = Path("C:/Program Files/Docker/Docker/resources/bin/docker.exe")
        return str(candidate) if candidate.is_file() else None

    @classmethod
    def available(cls) -> bool:
        executable = cls.find_docker()
        if executable is None:
            return False
        try:
            process = subprocess.run(
                [executable, "info", "--format", "{{.ServerVersion}}"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
                env=cls._docker_environment(executable),
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return process.returncode == 0

    def image_exists(self) -> bool:
        process = subprocess.run(
            [self.docker_executable, "image", "inspect", self.image],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
            env=self._docker_environment(self.docker_executable),
        )
        return process.returncode == 0

    def build_image(self, dockerfile: Path, context: Path) -> None:
        process = subprocess.run(
            [
                self.docker_executable,
                "build",
                "--pull",
                "--file",
                str(dockerfile.resolve()),
                "--tag",
                self.image,
                str(context.resolve()),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=1800,
            check=False,
            env=self._docker_environment(self.docker_executable),
        )
        if process.returncode != 0:
            raise RuntimeError(f"Docker sandbox image build failed: {process.stderr}")

    def ensure_image(self, dockerfile: Path, context: Path) -> None:
        if not self.image_exists():
            self.build_image(dockerfile, context)

    def run_script(
        self,
        worktree: Path,
        script: str,
        *,
        input_text: str | None,
        timeout_s: float,
    ) -> CommandResult:
        container_name = f"visiondoctor-{uuid.uuid4().hex[:12]}"
        mount_source = str(worktree.resolve())
        command = [
            self.docker_executable,
            "run",
            "--name",
            container_name,
            "--rm",
            "--label",
            "visiondoctor.component=candidate-sandbox",
            "--interactive",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges=true",
            "--pids-limit",
            str(self.pids_limit),
            "--memory",
            self.memory,
            "--cpus",
            str(self.cpus),
            "--user",
            "65532:65532",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev,size=64m",
            "--mount",
            f"type=bind,src={mount_source},dst=/workspace,readonly",
            "--workdir",
            "/workspace",
            self.image,
            "python",
            "-I",
            script,
        ]
        started = time.perf_counter()
        try:
            process = subprocess.run(
                command,
                input=input_text,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s,
                check=False,
                env=self._docker_environment(self.docker_executable),
            )
        except subprocess.TimeoutExpired as exc:
            subprocess.run(
                [self.docker_executable, "rm", "--force", container_name],
                capture_output=True,
                timeout=15,
                check=False,
                env=self._docker_environment(self.docker_executable),
            )
            return CommandResult(
                returncode=124,
                stdout=exc.stdout or "",
                stderr=exc.stderr or str(exc),
                duration_s=time.perf_counter() - started,
                timed_out=True,
            )
        return CommandResult(
            returncode=process.returncode,
            stdout=process.stdout,
            stderr=process.stderr,
            duration_s=time.perf_counter() - started,
        )

    @staticmethod
    def _docker_environment(executable: str) -> dict[str, str]:
        environment = os.environ.copy()
        binary_directory = str(Path(executable).resolve().parent)
        environment["PATH"] = binary_directory + os.pathsep + environment.get("PATH", "")
        return environment
