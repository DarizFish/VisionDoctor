from __future__ import annotations

import json
import re
import statistics
import subprocess
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import numpy as np

from visiondoctor.adapters.base import RuntimeHandle
from visiondoctor.sandbox import (
    GitWorktreeSandbox,
    LocalPythonRunner,
    PatchPolicyEvaluator,
    PythonSandboxRunner,
    SandboxError,
)
from visiondoctor.schemas import (
    BackendType,
    CandidateVersion,
    CaseExecutionResult,
    ExecutionResult,
    ExecutionStatus,
    Incident,
    PolicyCheck,
    PoseTransform,
    RobotOutputs,
    TestCaseRef,
    UnitTestResult,
)
from visiondoctor.tasks import PreparedTaskCase, get_task_adapter


class DatasetExecutionBackend:
    """Runs target code in an isolated worktree and computes fixed-motion outputs."""

    def __init__(
        self,
        incident: Incident,
        sandbox: GitWorktreeSandbox,
        *,
        command_timeout_s: float = 60.0,
        runner: PythonSandboxRunner | None = None,
    ) -> None:
        self.incident = incident
        self.sandbox = sandbox
        self.command_timeout_s = command_timeout_s
        self.runner = runner or LocalPythonRunner()
        self.task_adapter = get_task_adapter(incident.task.kind)
        self._case_results: dict[str, dict[str, CaseExecutionResult]] = {}
        self._unit_tests: dict[str, UnitTestResult] = {}
        self._statuses: dict[str, ExecutionStatus] = {}
        self._started: dict[str, datetime] = {}
        self._completed: dict[str, datetime] = {}
        self._policy_checks: dict[str, tuple[PolicyCheck, ...]] = {}
        self._policy_blocked: set[str] = set()
        self._calibrations: dict[str, float] = {}
        self._normalized_ratios: dict[str, float] = {}

    def prepare(self, candidate: CandidateVersion) -> RuntimeHandle:
        if candidate.patch_text and not self.runner.secure_for_untrusted:
            raise SandboxError(
                "model-generated patches require the Docker sandbox; local execution is forbidden"
            )
        handle = self.sandbox.create(candidate)
        self._started[handle.handle_id] = datetime.now(UTC)
        if candidate.patch_text:
            checks = PatchPolicyEvaluator().evaluate(
                handle.worktree, candidate.base_commit, self.incident.allowed_patch_scope
            )
            changed = tuple(
                line.strip().replace("\\", "/")
                for line in subprocess.run(
                    [
                        "git",
                        "-C",
                        str(handle.worktree),
                        "diff",
                        "--name-only",
                        candidate.base_commit,
                        "--",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                ).stdout.splitlines()
                if line.strip()
            )
            checks = (
                *checks,
                PolicyCheck(
                    name="expected_change_set",
                    passed=set(changed) == set(candidate.expected_changed_files),
                    details=(f"expected={candidate.expected_changed_files}; actual={changed}"),
                ),
                PolicyCheck(
                    name="execution_contract_immutable",
                    passed=not set(changed).intersection(self._protected_contract_paths()),
                    details=(
                        "protected execution paths="
                        f"{tuple(sorted(self._protected_contract_paths()))}; actual={changed}"
                    ),
                ),
            )
        else:
            checks = (
                PolicyCheck(name="reference_version", passed=True, details=candidate.kind.value),
            )
        self._policy_checks[handle.handle_id] = checks
        if not all(check.passed for check in checks):
            self._policy_blocked.add(handle.handle_id)
        return handle

    def policy_checks(self, handle: RuntimeHandle) -> tuple[PolicyCheck, ...]:
        return self._policy_checks[handle.handle_id]

    def run_case(self, handle: RuntimeHandle, case: TestCaseRef) -> CaseExecutionResult:
        if handle.handle_id not in self._case_results:
            self._run_all_cases(handle)
        return self._case_results[handle.handle_id][case.case_id]

    def _run_all_cases(self, handle: RuntimeHandle) -> None:
        if handle.handle_id in self._policy_blocked:
            message = "candidate execution blocked by patch policy before code execution"
            self._unit_tests[handle.handle_id] = UnitTestResult(
                passed=False,
                tests_run=0,
                failures=0,
                errors=1,
                duration_s=0,
                stderr=message,
            )
            self._case_results[handle.handle_id] = {
                case.case_id: CaseExecutionResult(
                    case_id=case.case_id,
                    status=ExecutionStatus.FAILED,
                    stderr=message,
                )
                for case in self.incident.case_set
            }
            self._statuses[handle.handle_id] = ExecutionStatus.FAILED
            self._completed[handle.handle_id] = datetime.now(UTC)
            return
        unit_tests = self._run_unit_tests(handle.worktree)
        self._unit_tests[handle.handle_id] = unit_tests
        payload_cases = []
        prepared_cases: dict[str, PreparedTaskCase] = {}
        for case in self.incident.case_set:
            prepared = self.task_adapter.prepare_execution(case)
            prepared_cases[case.case_id] = prepared
            payload_cases.append(prepared.runner_payload)
        payload = json.dumps({"cases": payload_cases, "benchmark_repeats": 2000})
        process = self.runner.run_script(
            handle.worktree,
            self._contract_script("runner_script", "runner.py"),
            input_text=payload,
            timeout_s=self.command_timeout_s,
        )
        if process.timed_out:
            self._case_results[handle.handle_id] = {
                case.case_id: CaseExecutionResult(
                    case_id=case.case_id,
                    status=ExecutionStatus.TIMEOUT,
                    stdout=process.stdout,
                    stderr=process.stderr,
                )
                for case in self.incident.case_set
            }
            self._statuses[handle.handle_id] = ExecutionStatus.TIMEOUT
            self._completed[handle.handle_id] = datetime.now(UTC)
            return
        if process.returncode != 0:
            self._case_results[handle.handle_id] = {
                case.case_id: CaseExecutionResult(
                    case_id=case.case_id,
                    status=ExecutionStatus.FAILED,
                    stdout=process.stdout,
                    stderr=process.stderr,
                )
                for case in self.incident.case_set
            }
            self._statuses[handle.handle_id] = ExecutionStatus.FAILED
            self._completed[handle.handle_id] = datetime.now(UTC)
            return

        try:
            decoded = json.loads(process.stdout)
            by_id = {item["case_id"]: item for item in decoded["results"]}
            if set(by_id) != {case.case_id for case in self.incident.case_set}:
                raise ValueError("candidate output case set does not match the incident")
            calibration = float(decoded["calibration_latency_s"])
            if calibration <= 0:
                raise ValueError("benchmark calibration must be positive")
            benchmark_samples = tuple(float(value) for value in decoded["benchmark_samples_s"])
            calibration_samples = tuple(
                float(value) for value in decoded["calibration_samples_s"]
            )
            if (
                len(benchmark_samples) != 15
                or len(calibration_samples) != 15
                or any(value <= 0 for value in (*benchmark_samples, *calibration_samples))
            ):
                raise ValueError("benchmark samples must contain 15 positive pairs")
            self._calibrations[handle.handle_id] = calibration
            self._normalized_ratios[handle.handle_id] = statistics.median(
                candidate_sample / calibration_sample
                for candidate_sample, calibration_sample in zip(
                    benchmark_samples, calibration_samples, strict=True
                )
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            message = f"candidate emitted invalid JSON results: {exc}"
            self._case_results[handle.handle_id] = {
                case.case_id: CaseExecutionResult(
                    case_id=case.case_id,
                    status=ExecutionStatus.FAILED,
                    stdout=process.stdout,
                    stderr=message,
                )
                for case in self.incident.case_set
            }
            self._statuses[handle.handle_id] = ExecutionStatus.FAILED
            self._completed[handle.handle_id] = datetime.now(UTC)
            return
        results: dict[str, CaseExecutionResult] = {}
        try:
            for case in self.incident.case_set:
                results[case.case_id] = self.task_adapter.parse_execution(
                    case,
                    by_id[case.case_id],
                    prepared_cases[case.case_id],
                    self._fixed_motion,
                )
        except (KeyError, TypeError, ValueError) as exc:
            message = f"candidate emitted an invalid {self.incident.task.kind} result: {exc}"
            results = {
                case.case_id: CaseExecutionResult(
                    case_id=case.case_id,
                    status=ExecutionStatus.FAILED,
                    stdout=process.stdout,
                    stderr=message,
                )
                for case in self.incident.case_set
            }
            self._statuses[handle.handle_id] = ExecutionStatus.FAILED
        else:
            self._statuses[handle.handle_id] = ExecutionStatus.SUCCESS
        self._case_results[handle.handle_id] = results
        self._completed[handle.handle_id] = datetime.now(UTC)

    def _fixed_motion(self, case_id: str, predicted_t_base_object: np.ndarray) -> RobotOutputs:
        target = np.asarray(predicted_t_base_object, dtype=float).copy()
        target[:3, 3] += np.array([0.0, 0.0, 0.10])
        position = target[:3, 3]
        completed = bool(
            -0.9 <= position[0] <= 0.9 and -0.9 <= position[1] <= 0.9 and 0.1 <= position[2] <= 1.5
        )
        actual = target.copy()
        checksum = sum(case_id.encode("utf-8"))
        jitter = np.array([((checksum % 5) - 2), (((checksum // 5) % 5) - 2), 1], dtype=float)
        actual[:3, 3] += jitter * 0.00005
        return RobotOutputs(
            target_tcp=PoseTransform.from_array("base", "tcp", target),
            actual_tcp=PoseTransform.from_array("base", "tcp", actual),
            fixed_motion_completed=completed,
            motion_sequence=("HOME", "OBSERVATION_POSE", "VALIDATION_POSE", "HOME"),
            error_code=None if completed else "TARGET_OUTSIDE_WHITELIST_WORKSPACE",
            source="dataset_simulation",
        )

    def _run_unit_tests(self, worktree: Path) -> UnitTestResult:
        process = self.runner.run_script(
            worktree,
            self._contract_script("test_runner_script", "test_runner.py"),
            input_text=None,
            timeout_s=self.command_timeout_s,
        )
        combined = process.stdout + "\n" + process.stderr
        match = re.search(r"Ran (\d+) tests?", combined)
        tests_run = int(match.group(1)) if match else 0
        failures = len(re.findall(r"^FAIL:", combined, re.MULTILINE))
        errors = len(re.findall(r"^ERROR:", combined, re.MULTILINE))
        return UnitTestResult(
            passed=process.returncode == 0 and tests_run > 0,
            tests_run=tests_run,
            failures=failures,
            errors=errors,
            duration_s=process.duration_s,
            stdout=process.stdout,
            stderr=process.stderr,
        )

    def finalize(self, handle: RuntimeHandle) -> ExecutionResult:
        if handle.handle_id not in self._case_results:
            self._run_all_cases(handle)
        return ExecutionResult(
            execution_id=f"EXEC-{handle.candidate.candidate_id}",
            candidate_id=handle.candidate.candidate_id,
            backend_type=BackendType.DATASET,
            case_results=tuple(
                self._case_results[handle.handle_id][case.case_id]
                for case in self.incident.case_set
            ),
            unit_tests=self._unit_tests[handle.handle_id],
            status=self._statuses[handle.handle_id],
            benchmark_calibration_s=self._calibrations.get(handle.handle_id),
            benchmark_normalized_ratio=self._normalized_ratios.get(handle.handle_id),
            started_at=self._started[handle.handle_id],
            completed_at=self._completed[handle.handle_id],
        )

    def cleanup(self, handle: RuntimeHandle, *, rollback: bool) -> bool:
        return self.sandbox.cleanup(handle, rollback=rollback)

    def diff(self, handle: RuntimeHandle) -> str:
        return self.sandbox.diff(handle)

    def _contract_script(self, key: str, default: str) -> str:
        contract = self.incident.metadata.get("execution_contract", {})
        if not isinstance(contract, dict):
            raise ValueError("incident metadata.execution_contract must be an object")
        value = str(contract.get(key, default)).replace("\\", "/")
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or ".git" in path.parts:
            raise ValueError(f"unsafe execution contract path: {value}")
        if path.suffix.lower() != ".py":
            raise ValueError(
                "the current automatic execution backend supports only Python contract scripts"
            )
        return path.as_posix()

    def _protected_contract_paths(self) -> frozenset[str]:
        return frozenset(
            {
                self._contract_script("runner_script", "runner.py"),
                self._contract_script("test_runner_script", "test_runner.py"),
            }
        )

    @staticmethod
    def _sanitized_environment() -> dict[str, str]:
        """Compatibility helper for callers auditing the local runner environment."""
        return LocalPythonRunner._sanitized_environment()
