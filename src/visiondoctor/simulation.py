from __future__ import annotations

import json
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import numpy as np

from visiondoctor.adapters.gazebo import GazeboAdapter
from visiondoctor.adapters.gazebo_view import GazeboVisualAdapter
from visiondoctor.geometry import quaternion_xyzw_from_rotation
from visiondoctor.sandbox.runner import DockerPythonRunner
from visiondoctor.schemas import PoseTransform, TaskKind, TestCaseRef
from visiondoctor.tasks import get_task_adapter

SimulationAction = Literal[
    "start_gui",
    "run_motion",
    "run_project_observation",
    "capture_rgbd",
    "stop",
]


class SimulationService:
    """Owns asynchronous local demo operations for the official Gazebo backend."""

    def __init__(self, project_root: Path, data_root: Path) -> None:
        self.data_root = data_root.resolve()
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.visual = GazeboVisualAdapter(project_root, self.data_root / "gui")
        self._lock = threading.Lock()
        self._active: dict[str, Any] | None = None
        self._latest: dict[str, Any] | None = self._load_latest()

    def status(self) -> dict[str, Any]:
        with self._lock:
            active = dict(self._active) if self._active else None
            latest = dict(self._latest) if self._latest else None
        return {
            "visual": self.visual.status(),
            "active_operation": active,
            "latest_operation": latest,
        }

    def start(
        self,
        action: SimulationAction,
        *,
        case_id: str = "live-gazebo-001",
        project_observation: dict[str, str] | None = None,
    ) -> dict:
        if action == "stop":
            return self._stop()
        with self._lock:
            if self._active is not None:
                raise RuntimeError(
                    f"simulation operation {self._active['operation_id']} is already running"
                )
            operation = {
                "operation_id": f"SIM-{uuid.uuid4().hex[:12]}",
                "action": action,
                "status": "RUNNING",
                "case_id": (
                    case_id
                    if action in {"capture_rgbd", "run_project_observation"}
                    else None
                ),
                "project_observation": project_observation,
                "created_at": _now(),
                "completed_at": None,
                "result": None,
                "error": None,
            }
            self._active = operation
            self._persist(operation)
        thread = threading.Thread(
            target=self._execute,
            args=(dict(operation),),
            daemon=True,
            name=f"visiondoctor-{operation['operation_id']}",
        )
        thread.start()
        return dict(operation)

    def _execute(self, operation: dict[str, Any]) -> None:
        try:
            if operation["action"] == "start_gui":
                result = self.visual.start(run_motion=False)
            elif operation["action"] == "run_motion":
                result = self.visual.run_motion()
            elif operation["action"] == "run_project_observation":
                result = self._run_project_observation(operation)
            elif operation["action"] == "capture_rgbd":
                capture_root = (
                    self.data_root / "captures" / str(operation["operation_id"])
                ).resolve()
                contract = GazeboAdapter.run_rgbd_capture_contract(
                    self.visual.project_root,
                    capture_root,
                    case_id=str(operation["case_id"]),
                )
                result = contract.payload
                if not contract.success:
                    raise RuntimeError(str(result.get("error", "Gazebo RGB-D capture failed")))
            else:
                raise ValueError(f"unknown simulation action: {operation['action']}")
        except Exception as exc:
            completed = {
                **operation,
                "status": "FAILED",
                "completed_at": _now(),
                "error": str(exc),
            }
        else:
            completed = {
                **operation,
                "status": "SUCCEEDED",
                "completed_at": _now(),
                "result": result,
            }
        with self._lock:
            self._active = None
            self._latest = completed
            self._persist(completed)

    def _run_project_observation(self, operation: dict[str, Any]) -> dict[str, Any]:
        context = operation.get("project_observation")
        if not isinstance(context, dict):
            raise ValueError("project observation context is missing")
        repository = Path(str(context.get("repository_path", ""))).resolve()
        manifest_path = Path(str(context.get("manifest_path", ""))).resolve()
        runner_value = str(context.get("runner_script", "runner.py")).replace("\\", "/")
        runner_path = PurePosixPath(runner_value)
        if (
            not repository.is_dir()
            or runner_path.is_absolute()
            or ".." in runner_path.parts
            or runner_path.suffix.lower() != ".py"
            or not (repository / runner_path.as_posix()).is_file()
        ):
            raise ValueError("project observation runner is unavailable or unsafe")
        if not manifest_path.is_file():
            raise ValueError("captured simulation evidence is unavailable")

        case_id = str(context.get("case_id") or "live-project-observation")
        case = TestCaseRef(
            case_id=case_id,
            manifest_path=str(manifest_path),
            reference_path=str(manifest_path),
        )
        prepared = get_task_adapter(TaskKind.RGBD_POSE).prepare_execution(case)
        runner = DockerPythonRunner()
        project_root = self.visual.project_root
        runner.ensure_image(project_root / "docker" / "sandbox.Dockerfile", project_root)
        process = runner.run_script(
            repository,
            runner_path.as_posix(),
            input_text=json.dumps(
                {"cases": [prepared.runner_payload], "benchmark_repeats": 1},
                separators=(",", ":"),
            ),
            timeout_s=60.0,
        )
        if process.timed_out or process.returncode != 0:
            raise RuntimeError(
                "project observation failed in the isolated runner: "
                + (process.stderr.strip() or "no result")
            )
        try:
            payload = json.loads(process.stdout)
            results = payload["results"]
            if not isinstance(results, list) or len(results) != 1:
                raise ValueError("runner must return exactly one observation")
            item = results[0]
            if item.get("case_id") != case_id:
                raise ValueError("runner returned a different observation id")
            object_pose = PoseTransform.from_array(
                "base", "object", np.asarray(item["T_base_object"], dtype=float)
            ).as_array()
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"project emitted an invalid target pose: {exc}") from exc
        target = object_pose.copy()
        target[:3, 3] += np.array([0.0, 0.0, 0.10])
        target_tcp = {
            "position": target[:3, 3].tolist(),
            "quaternion_xyzw": quaternion_xyzw_from_rotation(target[:3, :3]).tolist(),
        }
        motion = self.visual.run_motion(target_tcp=target_tcp)
        return {
            **motion,
            "observation_source": "current_project",
            "case_id": case_id,
            "project_target_tcp": target_tcp,
            "project_t_base_object": object_pose.tolist(),
            "runner_script": runner_path.as_posix(),
        }

    def _stop(self) -> dict[str, Any]:
        result = self.visual.stop()
        operation = {
            "operation_id": f"SIM-{uuid.uuid4().hex[:12]}",
            "action": "stop",
            "status": "SUCCEEDED",
            "case_id": None,
            "created_at": _now(),
            "completed_at": _now(),
            "result": result,
            "error": None,
        }
        with self._lock:
            self._active = None
            self._latest = operation
            self._persist(operation)
        return operation

    def _persist(self, value: dict[str, Any]) -> None:
        path = self.data_root / "latest-operation.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def _load_latest(self) -> dict[str, Any] | None:
        path = self.data_root / "latest-operation.json"
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(value, dict):
            return None
        if value.get("status") == "RUNNING":
            value = {
                **value,
                "status": "FAILED",
                "completed_at": _now(),
                "error": "API restarted while the operation was running",
            }
        return value


def _now() -> str:
    return datetime.now(UTC).isoformat()
